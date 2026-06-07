"""Population label patterns: Format A (held dominant code) + Format B (pattern).

12 behavioral cells × 2 formats × 100 walkers each.

* **Format A — "hold dominant code"**: for each phase in the cell, hold the
  label's mode KPMS code for the full phase duration.  Simplest sustained-
  code instruction.
* **Format B — "pattern-recreated"**: use the in-label code statistics from
  the corpus (Walk's 12↔0 oscillation with 7-frame blocks, Rear sustained
  hold, etc.) to feed a more behaviorally-natural code stream.

Init pose per cell: find a clip whose FIRST labeled behavior matches the
cell's first phase; all 100 walkers reset to that clip's frame 0.

z_e_scale = 0.0 (decoder-only).

Run::

    python -m c2a.experiments.run_population_label_patterns
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA

from c2a.paths import get_repo_root

REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.codegen.code_sources import load_codes
from c2a.codegen.shared.label_alignment import (
    BEHAVIORS, BEHAVIOR_TO_INT, UNLABELED, load_label_frames,
)
from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _clip_to_index(name: str) -> int:
    stem = Path(name).stem
    if not stem.startswith("clip_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Per-cell init clip selection (first-phase-match)
# ---------------------------------------------------------------------------


def pick_walking_pose_init_pool(
    labels_csv: Path,
    train_indices: np.ndarray,
    n_walkers: int,
    probe_frame: int,
    probe_frame_max: int,
    n_frames: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Sample n_walkers (train_pos, frame) pairs where each pose is 'Walk'.

    Mirrors ``_pick_start_clips`` from
    ``run_labeled_hidden_dynamics_controlled``: scans probe_frame..max for
    the earliest frame at which ≥``n_walkers`` train clips have label
    ``Walk`` at that frame, then returns those clip indices (as train_pos)
    and the chosen frame.
    """
    arrays = load_label_frames(
        labels_csv, clip_range=(0, 841), n_frames=n_frames,
    )
    behavior_array = arrays.behavior   # (N_clips_in_range, T) ints
    target = BEHAVIOR_TO_INT["Walk"]
    in_range = np.asarray(
        [int(c) for c in train_indices if int(c) < behavior_array.shape[0]],
        dtype=np.int64,
    )
    matches = (
        behavior_array[in_range, probe_frame:probe_frame_max + 1] == target
    )                                                   # (n_in_range, n_scan)
    counts = matches.sum(axis=0)
    # Walk is rare in this corpus (~100-200 frames per clip total).  Pick
    # the frame with the largest pool of Walking train clips, then
    # bootstrap-sample n_walkers with REPLACEMENT.  Without replacement
    # would cap us at the pool size (often <100).
    best_idx = int(counts.argmax())
    chosen_frame = probe_frame + best_idx
    pool = in_range[matches[:, best_idx]]
    if len(pool) == 0:
        raise RuntimeError(
            f"No train clip in [{probe_frame}, {probe_frame_max}] has "
            f"Walk pose at any frame.  Widen the probe range."
        )
    rng = np.random.RandomState(seed)
    # Sample with replacement if pool is smaller than n_walkers, else
    # without (for diversity).
    replace = len(pool) < n_walkers
    sampled = rng.choice(pool, size=n_walkers, replace=replace)
    if replace:
        import logging as _l
        _l.getLogger(__name__).info(
            "walking-pose pool size %d < n_walkers %d → sampling with "
            "replacement (chosen frame %d had max count %d in the corpus)",
            len(pool), n_walkers, chosen_frame, int(counts[best_idx]),
        )
    # Map original-clip-id → train-position index
    csv_to_train_pos = {int(orig): pos for pos, orig in enumerate(train_indices)}
    train_pos = np.asarray(
        [csv_to_train_pos[int(c)] for c in sampled], dtype=np.int32,
    )
    return train_pos, int(chosen_frame)


def pick_init_clip_per_label(
    labels_csv: Path,
    train_indices: np.ndarray,
    csv_to_h5_subsample: int = 2,
) -> dict[str, int]:
    """For each behavior label, find a TRAIN clip whose FIRST labeled segment
    is that label.  Returns ``{label: train_pos_index}``.

    Falls back to "any clip containing this label" if no clip starts with it.
    """
    csv_to_train_pos = {int(orig): pos for pos, orig in enumerate(train_indices)}
    # Collect per-clip behavior segments in start order.
    per_clip: dict[int, list[tuple[int, str]]] = defaultdict(list)
    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Track") != "behavior":
                continue
            clip_id = _clip_to_index(row["Video"])
            if clip_id < 0 or clip_id not in csv_to_train_pos:
                continue
            per_clip[clip_id].append((int(row["Start_Frame"]), row["Label"]))
    label_to_clip: dict[str, int] = {}
    label_to_clip_loose: dict[str, int] = {}
    for clip_id, segs in per_clip.items():
        segs.sort(key=lambda x: x[0])
        first_label = segs[0][1]
        if first_label not in label_to_clip:
            label_to_clip[first_label] = csv_to_train_pos[clip_id]
        for _, lab in segs:
            if lab not in label_to_clip_loose:
                label_to_clip_loose[lab] = csv_to_train_pos[clip_id]
    # Fall back to loose match for labels not represented as first-segment.
    for lab, clip in label_to_clip_loose.items():
        label_to_clip.setdefault(lab, clip)
    return label_to_clip


# ---------------------------------------------------------------------------
# Per-frame code sequence builders for each format
# ---------------------------------------------------------------------------


def build_format_A_sequence(
    labels: list[str], label_to_mode: dict[str, int], total_frames: int,
) -> np.ndarray:
    """Format A: hold the dominant (mode) code for each phase."""
    n_phases = len(labels)
    hold = total_frames // n_phases
    blocks = [np.full(hold, int(label_to_mode[l]), dtype=np.int32) for l in labels]
    rem = total_frames - hold * n_phases
    if rem > 0:
        blocks.append(np.full(rem, int(label_to_mode[labels[-1]]), dtype=np.int32))
    return np.concatenate(blocks)


def build_format_B_sequence(
    labels: list[str], pattern_spec: dict[str, Any], total_frames: int,
) -> np.ndarray:
    """Format B: per phase, emit the pattern for that label."""
    n_phases = len(labels)
    hold = total_frames // n_phases
    out_blocks: list[np.ndarray] = []
    for i, label in enumerate(labels):
        # Last phase absorbs the remainder.
        phase_len = hold + (total_frames - hold * n_phases if i == n_phases - 1 else 0)
        spec = pattern_spec[label]
        codes = list(spec["codes"])
        mode = str(spec["mode"])
        if mode == "hold":
            block = np.full(phase_len, int(codes[0]), dtype=np.int32)
        elif mode == "oscillate":
            # Two codes alternating, block_frames each
            bf = int(spec["block_frames"])
            pair = np.concatenate([
                np.full(bf, int(codes[0]), dtype=np.int32),
                np.full(bf, int(codes[1]), dtype=np.int32),
            ])
            n_repeats = (phase_len + len(pair) - 1) // len(pair)
            block = np.tile(pair, n_repeats)[:phase_len]
        elif mode == "cycle":
            # Cycle through codes, block_frames each
            bf = int(spec["block_frames"])
            cycle = np.concatenate([
                np.full(bf, int(c), dtype=np.int32) for c in codes
            ])
            n_repeats = (phase_len + len(cycle) - 1) // len(cycle)
            block = np.tile(cycle, n_repeats)[:phase_len]
        else:
            raise ValueError(f"Unknown pattern mode: {mode}")
        out_blocks.append(block)
    return np.concatenate(out_blocks)


def build_code_window(
    flat_codes: np.ndarray, code_stack_size: int,
) -> np.ndarray:
    """Convert (T,) flat code sequence to (T, code_stack_size) lookahead window."""
    T = int(flat_codes.shape[0])
    out = np.zeros((T, code_stack_size), dtype=np.float32)
    for t in range(T):
        for si in range(code_stack_size):
            idx = min(t + si, T - 1)
            out[t, si] = float(flat_codes[idx])
    return out


# ---------------------------------------------------------------------------
# Inference fn exposing vae_mean
# ---------------------------------------------------------------------------


def _make_vae_mean_inference_fn(
    ppo_networks: Any, z_e_scale: float = 0.0, deterministic: bool = True,
):
    """Build inference fn that returns vae_mean as `extras["vae_mean"]`.

    ``deterministic=True`` (default) takes the action distribution's MODE
    instead of sampling — this matches the original eval-quality
    inference path used by ``run_labeled_hidden_dynamics_controlled``.
    Sampling on top of a converged policy adds permanent un-corrected
    Gaussian noise that destabilizes precision motor control.
    """
    policy_network = ppo_networks.policy_network

    def policy(params, obs, hidden, key):
        (
            action_params, code_idx, cont_mean, cont_logvar,
            new_hidden, distill_mean, distill_logvar,
        ) = policy_network.apply(
            *params, obs, hidden, key,
            deterministic=deterministic, z_e_scale=z_e_scale,
        )
        action_dist = ppo_networks.parametric_action_distribution
        if deterministic:
            raw = action_dist.mode(action_params)
        else:
            raw = action_dist.sample_no_postprocessing(action_params, key)
        action = jnp.array(action_dist.postprocess(raw))
        extras = {"code_idx": code_idx, "vae_mean": distill_mean}
        if cont_mean is not None:
            extras["cont_mean"] = cont_mean
        return action, extras, new_hidden

    return jax.jit(policy)


def _batched_rollout(
    env, inf_fn, params, ppo_networks,
    code_sequence: jnp.ndarray,       # (T, code_stack_size)
    clip_indices: jnp.ndarray,        # (N,)
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
            new_obs = dict(state.obs)
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
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="configs",
    config_name="population_label_patterns",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("=" * 80)
    log.info("Population label-patterns experiment (Format A + Format B)")
    log.info("=" * 80)

    out_dir = Path(_abs(cfg.output.base_dir))
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log.info("Outputs under %s", out_dir)

    # --- Ckpt + env -----------------------------------------------------
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

    # --- Init mode handling ---------------------------------------------
    init_mode = str(cfg.population.get("init_mode", "same_pos"))
    log.info("init_mode=%s", init_mode)

    if init_mode == "walking_pose":
        # All walkers start at the same frame F (=probe frame) but on
        # DIFFERENT train clips that each have a "Walk" pose at F.
        # Frame F is shared so env can be built with start_frame_range=[F, F].
        walking_train_pos, walking_frame = pick_walking_pose_init_pool(
            labels_csv=Path(_abs(cfg.paths.labels_csv)),
            train_indices=train_indices,
            n_walkers=int(cfg.population.num_walkers),
            probe_frame=int(cfg.population.walking_probe_frame),
            probe_frame_max=int(cfg.population.walking_probe_frame_max),
            n_frames=int(cfg_local.env_config.clip_length),
            seed=int(cfg.population.init_seed),
        )
        log.info(
            "walking-pose init: frame=%d, %d train clips selected",
            walking_frame, len(walking_train_pos),
        )
        env_start_frame = walking_frame
        # Per-cell init_clips will be FIXED at these walking_train_pos
        # regardless of cell name.  The pool is the same across cells.
    else:
        walking_train_pos = None
        walking_frame = 0
        env_start_frame = 0

    env_cfg_ml_train = env_cfg_ml.copy_and_resolve_references()
    env_cfg_ml_train.start_frame_range = [env_start_frame, env_start_frame]
    env = MoSeqImitation(
        config=env_cfg_ml_train,
        clips=clips_train,
        kpms_codes=jnp.asarray(train_codes, dtype=jnp.int32),
        code_stack_size=code_stack_size,
        c2c_adapter=None,
        c2c_reward_mode="off",
    )

    # --- Per-label init clip (only consulted in same_pos mode) ----------
    label_to_init_clip = pick_init_clip_per_label(
        Path(_abs(cfg.paths.labels_csv)), train_indices,
    )
    if init_mode == "same_pos":
        log.info("Per-label init clip (train pos):")
        for label, pos in label_to_init_clip.items():
            log.info("  %-10s  train_pos=%d", label, pos)

    # --- Inference fn ----------------------------------------------------
    deterministic = bool(cfg.rollout.get("deterministic", True))
    from c2a.experiments.shared.decoder_only import (
        assert_decoder_only_settings, describe_decoder_only_mode,
    )
    assert_decoder_only_settings(float(cfg.rollout.z_e_scale), deterministic)
    log.info(describe_decoder_only_mode(ckpt_cfg.network_config))
    log.info("Inference: deterministic=%s, z_e_scale=%.3f",
             deterministic, float(cfg.rollout.z_e_scale))
    inf_fn = _make_vae_mean_inference_fn(
        ppo_networks, z_e_scale=float(cfg.rollout.z_e_scale),
        deterministic=deterministic,
    )

    n_walkers = int(cfg.population.num_walkers)
    total_frames = int(cfg.rollout.max_steps)

    @jax.jit
    def _run(code_sequence, init_clips, base_keys):
        return _batched_rollout(
            env, inf_fn, params, ppo_networks,
            code_sequence, init_clips, base_keys, total_frames,
        )

    label_to_mode = {str(k): int(v) for k, v in cfg.label_to_mode_code.items()}
    pattern_per_label = {
        str(k): {
            "mode": str(v.mode),
            "codes": [int(c) for c in v.codes],
            "block_frames": int(v.get("block_frames", 7)),
        }
        for k, v in cfg.pattern_per_label.items()
    }

    cells = list(cfg.combinations)
    log.info("Running %d cells × 2 formats × %d walkers × %d frames",
             len(cells), n_walkers, total_frames)

    all_data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for ci, combo in enumerate(cells):
        name = str(combo.name)
        labels = [str(l) for l in combo.labels]
        first_label = labels[0]

        # Init clips: in same_pos mode, all 100 walkers reset to source-clip
        # frame 0 (varies per cell).  In walking_pose mode, the 100 walkers
        # are pre-sampled (same across all cells) at frame F where each clip
        # has a Walk pose at that frame.
        if init_mode == "walking_pose":
            assert walking_train_pos is not None
            init_clips = jnp.asarray(walking_train_pos, dtype=jnp.int32)
            init_pos_for_record = int(walking_train_pos[0])  # representative
        else:
            if first_label not in label_to_init_clip:
                log.warning(
                    "  cell %r: no init clip for first label %r; skipping",
                    name, first_label,
                )
                continue
            init_pos = label_to_init_clip[first_label]
            init_clips = jnp.full((n_walkers,), int(init_pos), dtype=jnp.int32)
            init_pos_for_record = int(init_pos)

        for fi, (fmt, seq_builder) in enumerate([
            ("A", lambda labs: build_format_A_sequence(labs, label_to_mode, total_frames)),
            ("B", lambda labs: build_format_B_sequence(labs, pattern_per_label, total_frames)),
        ]):
            npz_path = (
                data_dir
                / f"cell_{ci:02d}_{name.replace(' ', '_').replace('→', 'arrow')}_fmt{fmt}.npz"
            )
            if cfg.output.reuse_rollouts and npz_path.exists():
                log.info("  [%d/%d %s] %s  CACHED", ci + 1, len(cells), fmt, name)
                with np.load(npz_path) as d:
                    all_data[(name, fmt)] = {k: d[k] for k in d.files}
                continue
            try:
                flat = seq_builder(labels)
            except KeyError as e:
                log.warning("  [%d/%d %s] %s  missing label %s; skipping",
                            ci + 1, len(cells), fmt, name, e)
                continue
            code_window = build_code_window(flat, code_stack_size)
            log.info(
                "  [%d/%d %s] %-40s  labels=%s  unique_codes_in_seq=%s",
                ci + 1, len(cells), fmt, name, labels,
                sorted(set(flat.tolist())),
            )
            base_keys = jnp.stack([
                jax.random.PRNGKey(
                    int(cfg.rollout.policy_seed)
                    + 1_000_003 * (ci + 1)
                    + 100 * (0 if fmt == "A" else 1)
                    + k
                )
                for k in range(n_walkers)
            ])
            outs = _run(jnp.asarray(code_window), init_clips, base_keys)
            agg = {
                "hidden":    np.asarray(outs["hidden"]),
                "intention": np.asarray(outs["intention"]),
                "qpos":      np.asarray(outs["qpos"]),
                "codes":     np.asarray(outs["code"]),
                "rewards":   np.asarray(outs["reward"]),
                "done":      np.asarray(outs["done"]),
                "init_clip_train_pos": np.int32(init_pos_for_record),
                "init_clips_all": np.asarray(init_clips),
                "init_mode": np.array(init_mode),
                "fed_code_seq": flat.astype(np.int32),
                "combination_name": np.array(name),
                "format": np.array(fmt),
                "labels": np.array(labels),
            }
            np.savez_compressed(npz_path, **agg)
            all_data[(name, fmt)] = agg
            survival = (
                np.argmax(agg["done"] > 0.5, axis=1).astype(np.float32)
                if np.any(agg["done"] > 0.5)
                else np.full(agg["done"].shape[0], total_frames, dtype=np.float32)
            )
            log.info(
                "    saved (mean_reward=%.3f, survival_mean=%.1f)",
                float(np.mean(agg["rewards"])), float(np.mean(survival)),
            )

    # --- Joint PCA across all cells & formats ----------------------------
    if all_data:
        log.info("Joint PCA-3D fit on all cells × both formats")
        hidden_all = np.concatenate(
            [v["hidden"].reshape(-1, v["hidden"].shape[-1]) for v in all_data.values()],
            axis=0,
        )
        intention_all = np.concatenate(
            [v["intention"].reshape(-1, v["intention"].shape[-1]) for v in all_data.values()],
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

        # Highlights via max-spread
        highlights: dict[str, np.ndarray] = {}
        for (name, fmt), d in all_data.items():
            key = f"{name.replace(' ', '_').replace('→', 'arrow')}_fmt{fmt}"
            h_pca = pca_h.transform(
                d["hidden"].reshape(-1, d["hidden"].shape[-1])
            ).reshape(*d["hidden"].shape[:-1], 3)
            final = h_pca[:, -1, :3]
            N = final.shape[0]
            chosen = [int(np.argmax(np.linalg.norm(final - final.mean(0), axis=1)))]
            while len(chosen) < int(cfg.highlight.n_highlight) and len(chosen) < N:
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
            highlights[key] = np.asarray(chosen, dtype=np.int64)
        np.savez_compressed(data_dir / "highlights.npz", **highlights)

    # --- Summary --------------------------------------------------------
    import json
    summary = {
        "checkpoint": str(cfg.checkpoint.path),
        "num_walkers": n_walkers,
        "total_frames": total_frames,
        "z_e_scale": float(cfg.rollout.z_e_scale),
        "label_to_mode_code": label_to_mode,
        "pattern_per_label": pattern_per_label,
        "label_to_init_clip_train_pos": label_to_init_clip,
        "cells": [
            {"name": str(c.name), "labels": [str(l) for l in c.labels]}
            for c in cells
        ],
    }
    if all_data:
        summary["pca_hidden_var_ratio"] = pca_h.explained_variance_ratio_.tolist()
        summary["pca_intention_var_ratio"] = pca_i.explained_variance_ratio_.tolist()
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", out_dir / "summary.json")
    log.info("Done.")


if __name__ == "__main__":
    main()
