"""Experiment: populate the (hidden, intention) subspace per code combination.

For each of 12 pure-behavior-label clips identified in
``data/rodent/rodent_labels.csv``, extract the clip's per-frame code sequence
from the checkpoint's segmenter and run N=100 walkers (same random frame=0
inits across all combinations, decoder-only mode ``z_e_scale=0``).  Collect
the GRU hidden state ``h_t`` and the encoder posterior mean ``z_e_mu``
(intention space) per walker per frame.

Two video products:

* ``grid_behavior.mp4`` — single grid of 3 ghost-rendered walkers per cell
  laid out in a 3×4 (rows×cols) configuration; the "behavior catalog".
* ``<cell>_3col.mp4`` — per-cell 3-column composite (motion ghost,
  hidden-state PCA-3D rotating, intention PCA-3D rotating) with 3 highlighted
  walker trajectories on top of 97 grey background dots.  Both PCAs share
  axes across cells via a joint PCA fit.

Pure-label clips are clips whose collapsed behavior-segment sequence equals
the cell's ``labels`` list exactly — no other behavior labels intervening.
The pool of n=100 random initial poses is sampled once at startup (clip
ids drawn uniformly with replacement, frame=0 each) and reused for every
cell.  This is intentional: keeping the inits constant attributes
cross-cell differences to the code sequence and not to init-pose noise.

Run::

    cd code2act
    python -m c2a.experiments.population.run_population_intention_dynamics

The output base directory and combination list are configured in
``c2a/experiments/configs/population_intention_dynamics.yaml``.
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Orbax 0.11.24 vs JAX 0.5.3 mismatch: Orbax calls
# ``jax.monitoring.record_scalar`` (telemetry, optional) from
# ``replica_slices.get_replica_slices`` during ``restore``.  Newer JAX has
# only ``record_event*`` — restore would AttributeError.  No-op the missing
# function before importing jax-using modules.
import jax.monitoring
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *_a, **_kw: None

import csv
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA

from c2a.paths import get_c2a_root, get_repo_root

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.codegen.code_sources import load_codes
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _abs(p) -> str:
    """Resolve ``p`` against the repo root if relative."""
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Pure-label clip mining
# ---------------------------------------------------------------------------


def _clip_to_index(clip_name: str) -> int:
    """Map ``clip_NNN.mp4`` → ``NNN`` int index.  Returns -1 if non-matching."""
    stem = Path(clip_name).stem
    if not stem.startswith("clip_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return -1


def _load_clip_label_sequences(
    csv_path: Path,
) -> dict[int, list[str]]:
    """Read the labels CSV and return ``{clip_idx: [behavior_label, ...]}``.

    Sequences are the collapsed list of consecutive behavior labels in
    Start_Frame order (consecutive duplicates folded into a single entry).
    Direction labels are ignored — the user requested behavior-only purity.
    Only clips whose ``Video`` is ``clip_NNN.mp4`` form are kept.
    """
    by_clip: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip = _clip_to_index(row["Video"])
            if clip < 0:
                continue
            if row.get("Track") != "behavior":
                continue
            by_clip[clip].append(
                {"label": row["Label"], "start": int(row["Start_Frame"])}
            )

    out: dict[int, list[str]] = {}
    for clip, segs in by_clip.items():
        segs.sort(key=lambda s: s["start"])
        seq: list[str] = []
        for s in segs:
            if not seq or seq[-1] != s["label"]:
                seq.append(s["label"])
        out[clip] = seq
    return out




# ---------------------------------------------------------------------------
# Random init pose pool
# ---------------------------------------------------------------------------


def _sample_init_clip_pool(
    n_walkers: int, n_clips: int, seed: int,
) -> np.ndarray:
    """Sample ``n_walkers`` clip indices for the random init pool.

    Each walker starts at frame=0 of its sampled clip — the standard "reset
    to the start of a randomly-picked reference clip" semantics that the env
    already supports via the ``clip_idx`` reset kwarg.  Sampling is uniform
    with replacement (n_walkers can exceed n_clips).
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_clips, size=n_walkers).astype(np.int32)


# ---------------------------------------------------------------------------
# Vmapped batched rollout (one cell)
# ---------------------------------------------------------------------------


def _make_vae_mean_inference_fn(
    ppo_networks: Any, normalizer_params: Any,
    z_e_scale: float = 0.0,
):
    """Custom inference fn that exposes ``vae_mean`` (a.k.a. distill_mean
    when ``use_pretrained_decoder=True``) in extras.

    The trained policy's actual *intention* — the value sampled into z and
    passed to the frozen MIMIC decoder — is ``vae_mean``.  The standard
    factory `make_inference_fn` exposes ``cont_mean`` (the encoder
    posterior) but ignores ``vae_mean``.  This patch unpacks the apply
    tuple's position 5 (``distill_mean`` when distillation+pretrained
    decoder are on, which equals ``vae_mean``) and returns it.
    """
    policy_network = ppo_networks.policy_network

    def policy(params, obs, hidden, key):
        # apply returns: (action_params, code_idx, mean, logvar,
        #                 new_hidden, distill_mean, distill_logvar)
        # With use_distillation_head=True AND use_pretrained_decoder=True
        # (the case for this ckpt), distill_mean == vae_mean.
        (
            action_params, code_idx, cont_mean, cont_logvar,
            new_hidden, distill_mean, distill_logvar,
        ) = policy_network.apply(
            *params, obs, hidden, key,
            deterministic=True,
            z_e_scale=z_e_scale,
        )
        # Deterministic action = distribution MODE (no sampling). Decoder-only
        # dynamics analysis requires this; stochastic actions inject per-step
        # noise. See c2a.experiments.shared.decoder_only.
        action_dist = ppo_networks.parametric_action_distribution
        raw = action_dist.mode(action_params)
        action = jnp.array(action_dist.postprocess(raw))
        extras = {
            "code_idx": code_idx,
            "vae_mean": distill_mean,    # the true intention
        }
        if cont_mean is not None:
            extras["cont_mean"] = cont_mean
        return action, extras, new_hidden

    return jax.jit(policy)


def _batched_rollout_for_sequence(
    env, inf_fn, params, ppo_networks,
    code_sequence: jnp.ndarray,        # (T_full, code_stack_size)
    clip_indices: jnp.ndarray,         # (N,)
    base_keys: jnp.ndarray,            # (N, 2)
    max_steps: int,
    code_stack_size: int,
):
    """Vmapped + scan rollout: N walkers fed the same code sequence.

    Same pattern as ``c2a/experiments/run_code_grid_rollouts.py::
    _batched_rollout_for_code`` but with a per-frame code lookup instead
    of a held constant code.  The recorded scan output includes the
    policy's ``vae_mean`` (intention to the frozen MIMIC decoder) per
    step — exposed via ``_make_vae_mean_inference_fn`` (vae_mean is
    ``distill_mean`` in the apply return tuple).

    Returns a dict of ``(N, T, ...)`` arrays.
    """
    def single_body(base_key, clip_idx):
        key, reset_key = jax.random.split(base_key)
        state0 = env.reset(reset_key, clip_idx=clip_idx)
        hidden0 = ppo_networks.policy_network.init_hidden(1)

        def step(carry, t):
            state, hidden, key = carry
            # Per-step code override: t -> code_sequence[t] (or clamp at end).
            t_clamped = jnp.minimum(t, code_sequence.shape[0] - 1)
            code_window = code_sequence[t_clamped]                 # (code_stack_size,)
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = code_window
            state = state.replace(obs=new_obs)

            key, subkey = jax.random.split(key)
            batched_obs = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inf_fn(params, batched_obs, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)

            vae_mean = extras.get("vae_mean", jnp.zeros((1,)))
            vae_mean = jnp.asarray(vae_mean).reshape(-1)

            out = {
                "hidden":    hidden[-1][0],
                "intention": vae_mean,           # vae_mean = the actual intent to frozen decoder
                "qpos":      state.data.qpos,
                "proprio":   state.obs["proprioception"],
                "code":      jnp.asarray(extras["code_idx"]).reshape(-1)[0].astype(jnp.int32),
            }
            new_state = env.step(state, action)
            out["reward"] = new_state.reward
            out["done"]   = new_state.done.astype(jnp.float32)
            return (new_state, hidden, key), out

        _, outs = jax.lax.scan(
            step, (state0, hidden0, key), jnp.arange(max_steps), length=max_steps,
        )
        return outs

    return jax.vmap(single_body)(base_keys, clip_indices)


def _build_code_sequence_window(
    codes_clip: np.ndarray, code_stack_size: int, max_steps: int,
) -> np.ndarray:
    """Build the (T, code_stack_size) code-override window for a clip.

    For each rollout step t, the obs window contains the codes at frames
    ``[t, t+1, ..., t+code_stack_size-1]`` of the source clip (with the
    clip-end clamped).  Same semantics as
    ``MoSeqImitation._get_obs.kpms_code``.
    """
    T_clip = int(codes_clip.shape[0])
    out = np.zeros((max_steps, code_stack_size), dtype=np.float32)
    for t in range(max_steps):
        for si in range(code_stack_size):
            idx = min(t + si, T_clip - 1)
            out[t, si] = float(codes_clip[idx])
    return out


# ---------------------------------------------------------------------------
# Highlight selection
# ---------------------------------------------------------------------------


def _pick_spread_highlights(
    hidden_pca: np.ndarray, n: int = 3,
) -> np.ndarray:
    """Pick ``n`` walker indices whose final-frame PCA positions are most spread.

    Greedy farthest-point selection on the final-frame positions
    (``hidden_pca[:, -1, :3]``).  Returns ``(n,)`` int indices into the
    walker axis.
    """
    final = hidden_pca[:, -1, :3]                                 # (N, 3)
    N = final.shape[0]
    if N <= n:
        return np.arange(N, dtype=np.int64)
    chosen = [int(np.argmax(np.linalg.norm(final - final.mean(0), axis=1)))]
    while len(chosen) < n:
        # Pick the point with the largest min-distance to already-chosen points.
        min_dist = np.min(
            np.linalg.norm(
                final[None, :, :] - final[chosen][:, None, :], axis=-1,
            ),
            axis=0,
        )
        nxt = int(np.argmax(min_dist))
        if nxt in chosen:
            break
        chosen.append(nxt)
    return np.asarray(chosen, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="configs",
    config_name="population_intention_dynamics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("=" * 80)
    log.info("Population intention dynamics experiment")
    log.info("=" * 80)

    out_dir = Path(_abs(cfg.output.base_dir))
    data_dir = out_dir / "data"
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    log.info("Outputs under %s", out_dir)

    # --- 1. Load checkpoint + build env -----------------------------------
    log.info("Loading checkpoint from %s", cfg.checkpoint.path)
    ckpt_cfg, normalizer_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path), step=cfg.checkpoint.get("step", None),
    )
    # Inference function expects `params` as the packed tuple of
    # (normalizer_state, policy_params) per the track-mjx convention.
    params = (normalizer_state, policy_params)

    network_cfg = ckpt_cfg.network_config
    code_stack_size = int(network_cfg.code_stack_size)
    num_codes = int(network_cfg.num_codes)
    use_rnn = bool(network_cfg.get("use_rnn_decoder", False))
    log.info(
        "  num_codes=%d, code_stack_size=%d, "
        "use_continuous_encoder=%s, use_rnn_decoder=%s",
        num_codes, code_stack_size,
        bool(network_cfg.get("use_continuous_encoder", False)),
        use_rnn,
    )

    # Build the env so we can find out N_clips and load reference clips.
    cfg_local, _, env_cfg_ml = utils.prepare_config(ckpt_cfg)
    ref_data_path = (
        REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"
    )
    if ref_data_path.exists():
        cfg_local.env_config.reference_data_path = str(ref_data_path)
        env_cfg_ml.reference_data_path = str(ref_data_path)

    splits = load_balanced_splits(_abs(cfg.paths.balanced_splits))
    balanced = splits["balanced"]
    train_indices = np.asarray(balanced["train_indices"], dtype=np.int64)
    test_indices = np.asarray(balanced["test_indices"], dtype=np.int64)
    log.info("  splits: train=%d, test=%d clips",
             len(train_indices), len(test_indices))

    # Load codes via the experiment-shared loader.  It returns codes
    # aligned to (train_indices, test_indices) in that order.
    codes_payload = load_codes(
        cfg_local, train_indices=train_indices, test_indices=test_indices,
        n_frames=int(cfg_local.env_config.clip_length),
    )
    train_codes = np.asarray(codes_payload.train_codes)
    test_codes = np.asarray(codes_payload.test_codes)
    log.info("  segmenter codes: train=%s, test=%s",
             train_codes.shape, test_codes.shape)

    # Train reference clips for the init pool.
    clips_train = ReferenceClips(
        data_path=str(env_cfg_ml.reference_data_path),
        n_frames_per_clip=int(env_cfg_ml.clip_length),
        keep_clips_idx=train_indices,
    )
    n_clips_train = int(clips_train.qpos.shape[0])
    log.info("  loaded %d train reference clips", n_clips_train)

    # Build a clip_idx_in_train → labels-csv-clip-idx map.  The CSV uses
    # original clip ids; the env's reset(clip_idx=...) is in the train-
    # subset positional index.  We need labels-csv-clip-id when we mine
    # pure-label clips and look them up by env-internal index.
    csv_to_train_pos: dict[int, int] = {
        int(orig): pos for pos, orig in enumerate(train_indices)
    }

    # Override start_frame_range = [0, 0] so every reset puts the walker
    # at the start of its clip — matches the user's "frame 0 only".
    env_cfg_ml_train = env_cfg_ml.copy_and_resolve_references()
    env_cfg_ml_train.start_frame_range = [0, 0]

    env = MoSeqImitation(
        config=env_cfg_ml_train,
        clips=clips_train,
        kpms_codes=jnp.asarray(train_codes, dtype=jnp.int32),
        code_stack_size=code_stack_size,
        c2c_adapter=None,
        c2c_reward_mode="off",
    )

    # Custom inference fn that exposes ``vae_mean`` (the policy's actual
    # intention to the frozen MIMIC decoder).  When the network has
    # use_distillation_head + use_pretrained_decoder both on (this ckpt's
    # case), distill_mean == vae_mean in the apply return tuple.
    from c2a.experiments.shared.decoder_only import (
        assert_decoder_only_settings, describe_decoder_only_mode,
        verify_encoder_severed_from_env,
    )
    _z_e_scale = float(cfg.rollout.z_e_scale)
    assert_decoder_only_settings(_z_e_scale, deterministic=True)
    log.info(describe_decoder_only_mode(ckpt_cfg.network_config))
    inf_fn = _make_vae_mean_inference_fn(
        ppo_networks, normalizer_state,
        z_e_scale=_z_e_scale,
    )
    # Behavioural guarantee: perturbing the encoder's private input (task_obs)
    # must not change the policy output. Confirms decoder-only end to end.
    verify_encoder_severed_from_env(
        inf_fn, params, ppo_networks, env, jax.random.PRNGKey(0),
    )

    # --- 2. Mine pure-label clips ----------------------------------------
    log.info("Mining pure-label clip combinations from %s",
             cfg.paths.labels_csv)
    clip_to_seq = _load_clip_label_sequences(Path(_abs(cfg.paths.labels_csv)))
    log.info("  %d clips have a behavior-label sequence", len(clip_to_seq))
    log.info("Picking the longest qualifying clip per combination:")
    picks_csv: dict[str, int] = {}
    for combo in cfg.combinations:
        name = str(combo["name"])
        target = tuple(str(x) for x in combo["labels"])
        # Find clips whose collapsed behavior sequence equals target AND that
        # are in the env-internal training subset.
        candidates = [
            int(c) for c, seq in clip_to_seq.items()
            if tuple(seq) == target and int(c) in csv_to_train_pos
        ]
        if not candidates:
            log.warning(
                "No train-subset pure-label match for %r — skipping. "
                "target=%s, %d corpus-wide matches with no train-subset clip",
                name, " → ".join(target),
                sum(1 for s in clip_to_seq.values() if tuple(s) == target),
            )
            continue
        picks_csv[name] = sorted(candidates)[0]
        log.info("  %-40s  → csv clip %4d  (%d train candidates)",
                 name, picks_csv[name], len(candidates))

    if not picks_csv:
        raise RuntimeError(
            "No combinations matched against train-subset pure-label clips. "
            "Check the labels CSV and the combinations list in the config."
        )

    # Convert each CSV clip id → env-internal train-position index.
    picks: dict[str, int] = {
        name: csv_to_train_pos[csv_idx] for name, csv_idx in picks_csv.items()
    }

    # --- 3. Per-cell init: ALL 100 walkers reset to the SOURCE clip's
    # frame 0 so the env's reference matches the code sequence being
    # fed.  Domain randomization + per-env qpos jitter give the spread
    # across the population.  This is the "source-clip-reference" fix
    # for the previous experiment where each walker tracked a random
    # clip whose reference disagreed with the codes.
    n_walkers = int(cfg.population.num_walkers)

    # --- 4. Vmap rollouts per cell ---------------------------------------
    max_steps = int(cfg.rollout.max_steps)

    # JIT once; reuse across cells (signature: code_sequence, init_clips, base_keys).
    @jax.jit
    def _run_one_cell(code_sequence, init_clips, base_keys):
        return _batched_rollout_for_sequence(
            env, inf_fn, params, ppo_networks,
            code_sequence, init_clips, base_keys,
            max_steps, code_stack_size,
        )

    log.info("Running %d cells × %d walkers × %d steps",
             len(picks), int(cfg.population.num_walkers), max_steps)
    cell_data: dict[str, dict[str, np.ndarray]] = {}
    for ci, (name, clip_idx) in enumerate(picks.items()):
        npz_path = data_dir / f"cell_{ci:02d}_{name.replace(' ', '_').replace('→', 'arrow')}.npz"
        if cfg.output.reuse_rollouts and npz_path.exists():
            log.info("  [%d/%d] %s — cached, reloading", ci + 1, len(picks), name)
            with np.load(npz_path) as d:
                cell_data[name] = {k: d[k] for k in d.files}
            continue

        codes_clip = train_codes[clip_idx]
        code_seq_window = _build_code_sequence_window(
            codes_clip, code_stack_size, max_steps,
        )
        log.info(
            "  [%d/%d] %s  (clip %d, codes head=%s)",
            ci + 1, len(picks), name, clip_idx, codes_clip[:5].tolist(),
        )
        # All 100 walkers reset to the source clip's train-pos index.
        init_clips = jnp.full((n_walkers,), int(clip_idx), dtype=jnp.int32)
        base_keys = jnp.stack([
            jax.random.PRNGKey(int(cfg.rollout.policy_seed) + 1_000_003 * (ci + 1) + k)
            for k in range(n_walkers)
        ])
        outs = _run_one_cell(jnp.asarray(code_seq_window), init_clips, base_keys)
        agg = {
            "hidden":    np.asarray(outs["hidden"]),
            "intention": np.asarray(outs["intention"]),
            "qpos":      np.asarray(outs["qpos"]),
            "proprio":   np.asarray(outs["proprio"]),
            "codes":     np.asarray(outs["code"]),
            "rewards":   np.asarray(outs["reward"]),
            "done":      np.asarray(outs["done"]),
            "init_clips": np.asarray(init_clips),
            "source_clip": np.int32(clip_idx),
            "source_code_seq": np.asarray(codes_clip[:max_steps]),
            "combination_name": np.array(name),
        }
        np.savez_compressed(npz_path, **agg)
        cell_data[name] = agg
        log.info(
            "    saved %s  (mean_reward=%.3f, survival_mean=%.1f frames)",
            npz_path.name, float(np.mean(agg["rewards"])),
            float(np.mean(np.argmax(agg["done"] > 0.5, axis=1).astype(np.float32)
                          if np.any(agg["done"] > 0.5) else np.full(agg["done"].shape[0], max_steps))),
        )

    # --- 5. Joint PCA fits ----------------------------------------------
    log.info("Joint PCA-3D fit on hidden + intention across all cells")
    hidden_all = np.concatenate(
        [v["hidden"].reshape(-1, v["hidden"].shape[-1]) for v in cell_data.values()],
        axis=0,
    )
    intention_all = np.concatenate(
        [v["intention"].reshape(-1, v["intention"].shape[-1]) for v in cell_data.values()],
        axis=0,
    )
    log.info("  hidden_all shape=%s, intention_all shape=%s",
             hidden_all.shape, intention_all.shape)
    pca_hidden = PCA(n_components=3).fit(hidden_all)
    pca_intention = PCA(n_components=3).fit(intention_all)
    np.savez_compressed(
        data_dir / "pca_basis.npz",
        hidden_components=pca_hidden.components_,
        hidden_mean=pca_hidden.mean_,
        hidden_variance=pca_hidden.explained_variance_,
        intention_components=pca_intention.components_,
        intention_mean=pca_intention.mean_,
        intention_variance=pca_intention.explained_variance_,
    )
    log.info(
        "  hidden  explained-var ratio (3D): %s",
        pca_hidden.explained_variance_ratio_,
    )
    log.info(
        "  intention explained-var ratio (3D): %s",
        pca_intention.explained_variance_ratio_,
    )

    # --- 6. Highlight picks + summary ------------------------------------
    log.info("Picking 3 highlighted walkers per cell via %s",
             cfg.highlight.strategy)
    highlights_per_cell: dict[str, np.ndarray] = {}
    for name, d in cell_data.items():
        hidden_pca = pca_hidden.transform(
            d["hidden"].reshape(-1, d["hidden"].shape[-1])
        ).reshape(*d["hidden"].shape[:-1], 3)
        highlights_per_cell[name] = _pick_spread_highlights(
            hidden_pca, n=int(cfg.highlight.n_highlight),
        )
        log.info("  %-40s  highlights=%s",
                 name, highlights_per_cell[name].tolist())

    np.savez_compressed(
        data_dir / "highlights.npz",
        **{k.replace(" ", "_").replace("→", "arrow"): v
           for k, v in highlights_per_cell.items()},
    )

    # --- 7. Save summary metadata ----------------------------------------
    summary = {
        "checkpoint": str(cfg.checkpoint.path),
        "num_walkers": int(cfg.population.num_walkers),
        "max_steps": max_steps,
        "z_e_scale": float(cfg.rollout.z_e_scale),
        "combinations": [
            {
                "name": name, "source_clip": int(picks[name]),
                "highlights": [int(h) for h in highlights_per_cell[name]],
            }
            for name in picks
        ],
        "pca_hidden_var_ratio":    pca_hidden.explained_variance_ratio_.tolist(),
        "pca_intention_var_ratio": pca_intention.explained_variance_ratio_.tolist(),
    }
    import json
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", out_dir / "summary.json")

    # --- 8. Render videos -----------------------------------------------
    if bool(cfg.video.data_only):
        log.info("video.data_only=true — skipping video render")
        return

    try:
        from c2a.experiments.shared.population_videos import (
            render_grid_behavior_video,
            render_per_cell_3col_video,
        )
    except ImportError:
        log.warning(
            "population_videos helpers not yet importable; skipping render. "
            "Run video rendering separately via the helper script."
        )
        return

    log.info("Rendering grid behavior video")
    try:
        render_grid_behavior_video(
            cell_data=cell_data,
            highlights_per_cell=highlights_per_cell,
            ckpt=None,
            env_cfg_ml=env_cfg_ml_train,
            clips=clips_train,
            cfg=cfg,
            out_path=video_dir / "grid_behavior.mp4",
        )
    except Exception:
        log.exception("Grid video render failed; npz already saved")

    log.info("Rendering per-cell 3-column videos")
    for name, d in cell_data.items():
        try:
            render_per_cell_3col_video(
                cell_name=name,
                cell_data=d,
                highlights=highlights_per_cell[name],
                pca_hidden=pca_hidden,
                pca_intention=pca_intention,
                ckpt=None,
                env_cfg_ml=env_cfg_ml_train,
                clips=clips_train,
                cfg=cfg,
                out_path=video_dir / f"{name.replace(' ', '_').replace('→', 'arrow')}.mp4",
            )
        except Exception:
            log.exception("Per-cell render failed for %r; continuing", name)

    log.info("Done.  Outputs at %s", out_dir)


if __name__ == "__main__":
    main()
