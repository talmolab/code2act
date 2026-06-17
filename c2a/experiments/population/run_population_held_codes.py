"""Population intention dynamics with HELD codes per behavior phase.

Companion to ``run_population_intention_dynamics`` that fixes three
problems with the natural-code-sequence version:

1. **Hold the code per phase, don't flicker.** Resolve each behavior
   label to its "agreement KPMS code" — the mode of KPMS codes across
   all frames labeled with that behavior in the corpus.  For a cell
   like "Imm → Walk → Imm", the rollout's per-frame code sequence is
   ``[code_imm] * 80 + [code_walk] * 80 + [code_imm] * 80`` so each
   syllable is held long enough for the policy to commit.
2. **z_e_scale=0** (decoder-only) so only ``[code_emb, proprio]``
   drives the GRU.  Matches the website's ``run_labeled_hidden_*``
   experiments.
3. **tab10 walker colors** for visually distinguishable highlights
   (the existing experiment used arbitrary red/blue/green).

Run::

    cd code2act
    python -m c2a.experiments.population.run_population_held_codes \\
        checkpoint.path=model_checkpoints/260407_031233_484020/
"""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.monitoring
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *_a, **_kw: None

import csv
import logging
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from c2a.paths import get_repo_root

REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.codegen.code_sources import load_codes
from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _clip_to_index(clip_name: str) -> int:
    stem = Path(clip_name).stem
    if not stem.startswith("clip_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# label → KPMS code resolution
# ---------------------------------------------------------------------------


def compute_label_to_code(
    labels_csv_path: Path,
    train_codes: np.ndarray,
    train_indices: np.ndarray,
    csv_to_h5_subsample: int = 2,
) -> dict[str, int]:
    """For each behavior label, return the mode KPMS code over all
    labeled frames in the training set.

    Subsampling: the labels CSV is at 2× the env's mocap_hz (30 fps vs
    15 fps usage), so frame indices are divided by ``csv_to_h5_subsample``.
    """
    csv_to_train_pos = {int(orig): pos for pos, orig in enumerate(train_indices)}
    label_to_codes: dict[str, list[int]] = defaultdict(list)
    T_codes = int(train_codes.shape[1])
    with open(labels_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Track") != "behavior":
                continue
            clip_id = _clip_to_index(row["Video"])
            if clip_id < 0 or clip_id not in csv_to_train_pos:
                continue
            train_pos = csv_to_train_pos[clip_id]
            start = int(row["Start_Frame"]) // csv_to_h5_subsample
            end = int(row["End_Frame"]) // csv_to_h5_subsample
            end = min(end, T_codes)
            if start >= end:
                continue
            codes = train_codes[train_pos, start:end]
            label_to_codes[row["Label"]].extend(codes.tolist())
    result: dict[str, int] = {}
    for label, codes in label_to_codes.items():
        if not codes:
            continue
        counter = Counter(codes)
        mode_code, mode_count = counter.most_common(1)[0]
        result[label] = int(mode_code)
        log.info(
            "label %-10s  agreement_code=%d  (mode_count=%d of %d total = %.1f%%, top-3=%s)",
            label, mode_code, mode_count, len(codes),
            100.0 * mode_count / len(codes),
            [(c, n) for c, n in counter.most_common(3)],
        )
    return result


# ---------------------------------------------------------------------------
# Held-code sequence construction
# ---------------------------------------------------------------------------


def build_hold_sequence(
    labels: list[str], label_to_code: dict[str, int],
    total_frames: int, code_stack_size: int,
) -> np.ndarray:
    """Build (T, code_stack_size) held-code sequence per cell.

    Each label holds for ``total_frames // len(labels)`` frames.
    Returns float32 because that's what the env's obs["kpms_code"] is.
    """
    n_phases = len(labels)
    hold = total_frames // n_phases
    blocks: list[np.ndarray] = []
    for lab in labels:
        code = label_to_code[lab]
        blocks.append(np.full(hold, code, dtype=np.int32))
    # If integer division left a remainder, pad with the last code.
    rem = total_frames - hold * n_phases
    if rem > 0:
        blocks.append(np.full(rem, label_to_code[labels[-1]], dtype=np.int32))
    codes_arr = np.concatenate(blocks)            # (total_frames,)
    # Build (T, code_stack_size) window — at each step t, look ahead by
    # code_stack_size positions (clamped at the end).
    T = int(codes_arr.shape[0])
    out = np.zeros((T, code_stack_size), dtype=np.float32)
    for t in range(T):
        for si in range(code_stack_size):
            idx = min(t + si, T - 1)
            out[t, si] = float(codes_arr[idx])
    return out


# ---------------------------------------------------------------------------
# vae_mean-exposing inference fn
# ---------------------------------------------------------------------------


def _make_vae_mean_inference_fn(
    ppo_networks: Any, z_e_scale: float = 0.0,
):
    """Custom inference fn that exposes vae_mean (== distill_mean when
    use_distillation_head AND use_pretrained_decoder are both True).
    """
    policy_network = ppo_networks.policy_network

    def policy(params, obs, hidden, key):
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
        extras = {"code_idx": code_idx, "vae_mean": distill_mean}
        if cont_mean is not None:
            extras["cont_mean"] = cont_mean
        return action, extras, new_hidden

    return jax.jit(policy)


# ---------------------------------------------------------------------------
# Vmapped rollout (vmap over walkers, scan over time)
# ---------------------------------------------------------------------------


def _batched_rollout(
    env, inf_fn, params, ppo_networks,
    code_sequence: jnp.ndarray,       # (T, code_stack_size)
    clip_indices: jnp.ndarray,        # (N,) — env-internal train pos
    base_keys: jnp.ndarray,           # (N, 2)
    total_frames: int,
):
    def single_body(base_key, clip_idx):
        key, reset_key = jax.random.split(base_key)
        state0 = env.reset(reset_key, clip_idx=clip_idx)
        hidden0 = ppo_networks.policy_network.init_hidden(1)

        def step(carry, t):
            state, hidden, key = carry
            t_clamped = jnp.minimum(t, code_sequence.shape[0] - 1)
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = code_sequence[t_clamped]
            state = state.replace(obs=new_obs)

            key, subkey = jax.random.split(key)
            batched = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inf_fn(params, batched, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)
            vae_mean = jnp.asarray(extras["vae_mean"]).reshape(-1)
            out = {
                "hidden":    hidden[-1][0],
                "intention": vae_mean,
                "qpos":      state.data.qpos,
                "code":      jnp.asarray(extras["code_idx"]).reshape(-1)[0].astype(jnp.int32),
            }
            new_state = env.step(state, action)
            out["reward"] = new_state.reward
            out["done"]   = new_state.done.astype(jnp.float32)
            return (new_state, hidden, key), out

        _, outs = jax.lax.scan(
            step, (state0, hidden0, key), jnp.arange(total_frames), length=total_frames,
        )
        return outs

    return jax.vmap(single_body)(base_keys, clip_indices)


# ---------------------------------------------------------------------------
# Pure-label clip mining (same as the natural-sequence experiment)
# ---------------------------------------------------------------------------


def _load_clip_label_sequences(csv_path: Path) -> dict[int, list[str]]:
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
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="configs",
    config_name="population_intention_dynamics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("=" * 80)
    log.info("Population HELD-code experiment (label-resolved sustained codes)")
    log.info("=" * 80)

    out_dir = Path(_abs(cfg.output.base_dir))
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info("Outputs under %s", out_dir)

    # --- 1. Ckpt + env ---------------------------------------------------
    log.info("Loading checkpoint from %s", cfg.checkpoint.path)
    ckpt_cfg, normalizer_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path), step=cfg.checkpoint.get("step", None),
    )
    params = (normalizer_state, policy_params)
    network_cfg = ckpt_cfg.network_config
    code_stack_size = int(network_cfg.code_stack_size)
    num_codes = int(network_cfg.num_codes)
    log.info("  num_codes=%d, code_stack_size=%d", num_codes, code_stack_size)

    cfg_local, _, env_cfg_ml = utils.prepare_config(ckpt_cfg)
    ref_data_path = REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"
    if ref_data_path.exists():
        cfg_local.env_config.reference_data_path = str(ref_data_path)
        env_cfg_ml.reference_data_path = str(ref_data_path)

    splits = load_balanced_splits(_abs(cfg.paths.balanced_splits))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    log.info("  splits: train=%d, test=%d clips", len(train_indices), len(test_indices))

    codes_payload = load_codes(
        cfg_local, train_indices=train_indices, test_indices=test_indices,
        n_frames=int(cfg_local.env_config.clip_length),
    )
    train_codes = np.asarray(codes_payload.train_codes)
    log.info("  segmenter codes train=%s", train_codes.shape)

    clips_train = ReferenceClips(
        data_path=str(env_cfg_ml.reference_data_path),
        n_frames_per_clip=int(env_cfg_ml.clip_length),
        keep_clips_idx=train_indices,
    )
    csv_to_train_pos = {int(orig): pos for pos, orig in enumerate(train_indices)}

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

    # --- 2. label → code mode lookup ------------------------------------
    log.info("Computing label → KPMS-code mode (agreement code) mapping")
    label_to_code = compute_label_to_code(
        Path(_abs(cfg.paths.labels_csv)), train_codes, train_indices,
    )

    # --- 3. Identify pure-label clip per combination --------------------
    clip_to_seq = _load_clip_label_sequences(Path(_abs(cfg.paths.labels_csv)))
    log.info("Picking source clip per cell (for the env reset pose)")
    picks: dict[str, tuple[int, list[str]]] = {}
    for combo in cfg.combinations:
        name = str(combo["name"])
        target = tuple(str(x) for x in combo["labels"])
        candidates = sorted(
            int(c) for c, seq in clip_to_seq.items()
            if tuple(seq) == target and int(c) in csv_to_train_pos
        )
        if not candidates:
            log.warning("No pure-label clip for %r — skipping", name)
            continue
        train_pos = csv_to_train_pos[candidates[0]]
        picks[name] = (train_pos, list(target))
        log.info("  %-40s  source clip %d  labels=%s",
                 name, candidates[0], list(target))

    # --- 4. Inference fn (decoder-only) ---------------------------------
    from c2a.experiments.shared.decoder_only import (
        assert_decoder_only_settings, describe_decoder_only_mode,
        verify_encoder_severed_from_env,
    )
    _z_e_scale = float(cfg.rollout.z_e_scale)
    assert_decoder_only_settings(_z_e_scale, deterministic=True)
    log.info(describe_decoder_only_mode(ckpt_cfg.network_config))
    inf_fn = _make_vae_mean_inference_fn(
        ppo_networks, z_e_scale=_z_e_scale,
    )
    # Behavioural guarantee: perturbing the encoder's private input (task_obs)
    # must not change the policy output. Confirms decoder-only end to end.
    verify_encoder_severed_from_env(
        inf_fn, params, ppo_networks, env, jax.random.PRNGKey(0),
    )

    # --- 5. Vmap rollouts per cell --------------------------------------
    n_walkers = int(cfg.population.num_walkers)
    total_frames = int(cfg.rollout.max_steps)
    log.info("Running %d cells × %d walkers × %d frames",
             len(picks), n_walkers, total_frames)

    @jax.jit
    def _run_one_cell(code_sequence, init_clips, base_keys):
        return _batched_rollout(
            env, inf_fn, params, ppo_networks,
            code_sequence, init_clips, base_keys, total_frames,
        )

    cell_data: dict[str, dict[str, np.ndarray]] = {}
    for ci, (name, (train_pos, labels)) in enumerate(picks.items()):
        npz_path = data_dir / f"cell_{ci:02d}_{name.replace(' ', '_').replace('→', 'arrow')}.npz"
        if cfg.output.reuse_rollouts and npz_path.exists():
            log.info("  [%d/%d] %s — cached", ci + 1, len(picks), name)
            with np.load(npz_path) as d:
                cell_data[name] = {k: d[k] for k in d.files}
            continue

        # Skip if any label not in the mapping (e.g., csv had labels but no codes).
        try:
            held_codes = [label_to_code[l] for l in labels]
        except KeyError as e:
            log.warning("  skipping %r: label %s has no KPMS code mapping", name, e)
            continue

        code_seq = build_hold_sequence(
            labels, label_to_code, total_frames, code_stack_size,
        )
        log.info(
            "  [%d/%d] %-40s  labels=%s  →  codes=%s  (hold %d frames each)",
            ci + 1, len(picks), name, labels, held_codes,
            total_frames // len(labels),
        )
        init_clips = jnp.full((n_walkers,), train_pos, dtype=jnp.int32)
        base_keys = jnp.stack([
            jax.random.PRNGKey(int(cfg.rollout.policy_seed) + 1_000_003 * (ci + 1) + k)
            for k in range(n_walkers)
        ])
        outs = _run_one_cell(jnp.asarray(code_seq), init_clips, base_keys)
        agg = {
            "hidden":    np.asarray(outs["hidden"]),
            "intention": np.asarray(outs["intention"]),
            "qpos":      np.asarray(outs["qpos"]),
            "codes":     np.asarray(outs["code"]),
            "rewards":   np.asarray(outs["reward"]),
            "done":      np.asarray(outs["done"]),
            "init_clips": np.asarray(init_clips),
            "source_clip_train_pos": np.int32(train_pos),
            "source_code_seq": code_seq[:, 0].astype(np.int32),
            "combination_name": np.array(name),
            "labels": np.array(labels),
            "held_codes": np.array(held_codes),
        }
        np.savez_compressed(npz_path, **agg)
        cell_data[name] = agg
        survival = (
            np.argmax(agg["done"] > 0.5, axis=1).astype(np.float32)
            if np.any(agg["done"] > 0.5)
            else np.full(agg["done"].shape[0], total_frames, dtype=np.float32)
        )
        log.info(
            "    saved (mean_reward=%.3f, survival_mean=%.1f frames)",
            float(np.mean(agg["rewards"])), float(np.mean(survival)),
        )

    # --- 6. Joint PCA across all cells ----------------------------------
    if cell_data:
        log.info("Joint PCA-3D fit")
        hidden_all = np.concatenate(
            [v["hidden"].reshape(-1, v["hidden"].shape[-1]) for v in cell_data.values()],
            axis=0,
        )
        intention_all = np.concatenate(
            [v["intention"].reshape(-1, v["intention"].shape[-1]) for v in cell_data.values()],
            axis=0,
        )
        pca_h = PCA(n_components=3).fit(hidden_all)
        pca_i = PCA(n_components=3).fit(intention_all)
        np.savez_compressed(
            data_dir / "pca_basis.npz",
            hidden_components=pca_h.components_, hidden_mean=pca_h.mean_,
            hidden_variance=pca_h.explained_variance_,
            intention_components=pca_i.components_, intention_mean=pca_i.mean_,
            intention_variance=pca_i.explained_variance_,
        )
        log.info("  hidden var ratio: %s", pca_h.explained_variance_ratio_)
        log.info("  intention var ratio: %s", pca_i.explained_variance_ratio_)

    # --- 7. Highlights via spread + tab10 colors ------------------------
    if cell_data:
        cmap = plt.colormaps["tab10"]
        tab10 = [list(cmap(i))[:3] for i in range(10)]  # RGB only, no alpha
        log.info("Picking 3 highlighted walkers per cell (max-spread on hidden PCA)")
        highlights: dict[str, np.ndarray] = {}
        for name, d in cell_data.items():
            h_pca = pca_h.transform(
                d["hidden"].reshape(-1, d["hidden"].shape[-1])
            ).reshape(*d["hidden"].shape[:-1], 3)
            final = h_pca[:, -1, :3]
            N = final.shape[0]
            chosen = [int(np.argmax(np.linalg.norm(final - final.mean(0), axis=1)))]
            while len(chosen) < 3 and len(chosen) < N:
                d_min = np.min(
                    np.linalg.norm(
                        final[None, :, :] - final[chosen][:, None, :], axis=-1,
                    ),
                    axis=0,
                )
                nxt = int(np.argmax(d_min))
                if nxt in chosen:
                    break
                chosen.append(nxt)
            highlights[name] = np.asarray(chosen, dtype=np.int64)
            log.info("  %-40s  highlights=%s", name, chosen)

        np.savez_compressed(
            data_dir / "highlights.npz",
            **{n.replace(" ", "_").replace("→", "arrow"): v
               for n, v in highlights.items()},
        )

    # --- 8. Save summary -------------------------------------------------
    import json
    summary = {
        "checkpoint": str(cfg.checkpoint.path),
        "num_walkers": n_walkers,
        "total_frames": total_frames,
        "z_e_scale": float(cfg.rollout.z_e_scale),
        "label_to_code": label_to_code,
        "combinations": [
            {
                "name": name, "source_clip_train_pos": int(picks[name][0]),
                "labels": picks[name][1],
                "held_codes": [label_to_code[l] for l in picks[name][1]],
            }
            for name in picks
        ],
    }
    if cell_data:
        summary["pca_hidden_var_ratio"] = pca_h.explained_variance_ratio_.tolist()
        summary["pca_intention_var_ratio"] = pca_i.explained_variance_ratio_.tolist()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
