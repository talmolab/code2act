"""Experiment: per-code rollouts from controlled poses, rendered as 5x5 grids.

For each tokenizer code (top-N by usage) and each starting pose (walk / rear),
roll out K bodies that hold that single code for ``hold_frames`` steps in
decoder-only mode. Output:

* Raw npz rollouts per (pose, code).
* Four 5x5 grid videos (codes 0..24, 25..49, per pose) — each panel shows
  the K walkers under one code.
* Per-pose MMD merge of codes by behavioral similarity (UPGMA + elbow cut).
* Per-pose "merged" grid videos: one panel per cluster, walkers colored
  by their original code.

Companion script: ``run_code_grid_dynamics.py`` runs DSA + rSLDS on the
merged-cluster hidden state trajectories. See its module docstring.

Usage:
    cd code2act
    python -m c2a.experiments.run_code_grid_rollouts
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

# Workaround for venv's orbax-checkpoint (0.11.24) calling jax.monitoring.record_scalar,
# which was removed from JAX. Same patch as c2a/train_moseq_decoder.py — needed before
# any Orbax restore happens. Pure telemetry call; stubbing is safe.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

from c2a.paths import get_c2a_root, get_repo_root

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.codegen.code_sources import load_codes
from c2a.codegen.shared.label_alignment import (
    BEHAVIORS, BEHAVIOR_TO_INT, UNLABELED, load_label_frames,
)
from c2a.experiments.run_hidden_dynamics import run_rollout_with_hidden
from c2a.experiments.run_labeled_hidden_dynamics_controlled import (
    _pick_start_clips, _build_pose_env,
)
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint, make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.code_grid import (
    make_code_colors, rank_codes_by_usage,
    compute_mmd_distance_matrix, cluster_codes,
    render_panel_frames, add_banner, stack_panels_into_grid, write_mp4,
    save_clusters_json, build_uniform_walkers,
)
from c2a.experiments.shared.code_labels_from_rollouts import (
    cluster_majority_label_rollout as cluster_majority_label,
    label_based_clusters_rollout as label_based_clusters,
    label_codes_from_rollouts_dir,
    load_code_labels_json,
    save_code_labels_json,
)

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Rollout phase
# ---------------------------------------------------------------------------


def _rollouts_for_pose(
    env, inf_fn, params, ppo_networks,
    pose_clips: np.ndarray, code_ids: list[int],
    hold_frames: int, seed: int,
    jit_reset, jit_step,
    pose_name: str, data_dir: Path,
    reuse: bool,
) -> dict[int, dict[str, np.ndarray]]:
    """For each code id, K rollouts holding that code for ``hold_frames``.

    Returns ``{code_id: agg_dict}`` where each agg has keys
    ``hidden, qpos, proprio, codes_actual, rewards, start_clips``.

    NB: ``code_override`` is broadcast to ``hold_frames + code_stack_size``
    steps internally by run_rollout_with_hidden — we only need an array of
    length ``hold_frames`` (the helper indexes off the end safely).
    """
    K = int(pose_clips.shape[0])
    results: dict[int, dict[str, np.ndarray]] = {}
    for ci, code in enumerate(code_ids):
        npz_path = data_dir / f"rollout_{pose_name}_code{int(code):03d}.npz"
        if reuse and npz_path.exists():
            d = np.load(npz_path)
            results[int(code)] = {k: d[k] for k in d.files}
            continue

        codes_arr = np.full(hold_frames, int(code), dtype=np.int32)

        hidden_all, qpos_all, proprio_all = [], [], []
        codes_actual_all, rewards_all = [], []

        for k in range(K):
            key = jax.random.PRNGKey(seed + 1_000_003 * (ci + 1) + k)
            result = run_rollout_with_hidden(
                env, inf_fn, params, ppo_networks, key,
                max_steps=hold_frames,
                code_override=codes_arr,
                reset_clip_idx=int(k),
                jit_reset=jit_reset, jit_step=jit_step,
            )

            def _rect(a: np.ndarray) -> np.ndarray:
                if a.shape[0] >= hold_frames:
                    return a[:hold_frames]
                pad = np.zeros(
                    (hold_frames - a.shape[0],) + a.shape[1:], dtype=a.dtype,
                )
                return np.concatenate([a, pad], axis=0)

            hidden_all.append(_rect(result["hidden_states"]))
            qpos_all.append(_rect(result["qpos"]))
            proprio_all.append(_rect(result["proprio"]))
            codes_actual_all.append(_rect(result["code_indices"]))
            rewards_all.append(_rect(result["rewards"]))

        agg = {
            "hidden": np.stack(hidden_all, axis=0),
            "qpos": np.stack(qpos_all, axis=0),
            "proprio": np.stack(proprio_all, axis=0),
            "codes_actual": np.stack(codes_actual_all, axis=0),
            "rewards": np.stack(rewards_all, axis=0),
            "start_clips": np.asarray(pose_clips, dtype=np.int64),
            "code": np.int32(int(code)),
            "pose": pose_name,
        }
        np.savez_compressed(npz_path, **agg)
        log.info(
            f"  [{pose_name}] code {int(code):>3d}: saved {npz_path.name} "
            f"(K={K}, hold={hold_frames}, mean_reward="
            f"{float(np.mean(agg['rewards'])):.3f})"
        )
        results[int(code)] = agg
    return results


# ---------------------------------------------------------------------------
# Raw grid render: one ghost_model per pose, K=K_per_panel ghosts.
# ---------------------------------------------------------------------------


def _solid_ghost_colors(rgba: np.ndarray, n: int) -> list[list[float]]:
    return [rgba.tolist() for _ in range(n)]


def _render_raw_grid_for_pose(
    env, pose_name: str, top_codes_in_pose: list[int],
    rollouts: dict[int, dict[str, np.ndarray]],
    code_colors: np.ndarray, usage: np.ndarray,
    cfg: DictConfig, video_dir: Path,
    code_labels: dict[int, dict[str, Any]] | None = None,
) -> None:
    """One grid video per (pose, chunk-of-codes). Each panel = one code."""
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    panels_per_grid = rows * cols
    K = int(cfg.K_per_panel)
    panel_w, panel_h = int(cfg.panel_w), int(cfg.panel_h)
    panel_banner_h = int(cfg.panel_banner_h)
    grid_banner_h = int(cfg.grid_banner_h)
    fps = int(cfg.render_fps)

    chunks = [
        top_codes_in_pose[i:i + panels_per_grid]
        for i in range(0, len(top_codes_in_pose), panels_per_grid)
    ]
    for gi, chunk in enumerate(chunks):
        log.info(
            f"  [{pose_name}] raw grid {gi}: codes {chunk[0]}..{chunk[-1]} "
            f"({len(chunk)} panels)"
        )
        panel_frames_list: list[np.ndarray] = []
        for rank_idx, code in enumerate(chunk):
            agg = rollouts[int(code)]
            qpos = agg["qpos"]  # (K, T, nq)
            traj = [qpos[k] for k in range(min(K, qpos.shape[0]))]
            # Per-panel ghost model: K walkers all in the same code color
            # (base body recolored to match ghosts via build_uniform_walkers).
            color = code_colors[int(code)]
            walker_colors = [color.tolist()] * len(traj)
            ghost_model, base_nq = build_uniform_walkers(
                env, walker_colors=walker_colors,
                camera_distance=float(cfg.camera_distance),
                camera_elevation=float(cfg.camera_elevation),
                camera_azimuth=float(cfg.camera_azimuth),
            )
            raw_frames = render_panel_frames(
                ghost_model, base_nq, traj,
                camera_name="divergent_cam",
                width=panel_w, height=panel_h,
            )
            color_dot = "● "
            lab = (
                str(code_labels[int(code)]["label"])
                if code_labels is not None and int(code) in code_labels
                else None
            )
            lab_str = f" [{lab}]" if lab and lab != "unlabeled" else ""
            title = (
                f"{color_dot}code {int(code):>3d}{lab_str} "
                f"rank={gi * panels_per_grid + rank_idx + 1} "
                f"use={int(usage[int(code)])}"
            )
            # Per-panel banner stacked above the rendered frames.
            framed = np.stack([
                add_banner(
                    f, title, banner_h=panel_banner_h,
                    bg=(245, 245, 245),
                    fg=tuple(int(255 * c * 0.65) for c in color[:3]),
                    font_size=12,
                )
                for f in raw_frames
            ], axis=0)
            panel_frames_list.append(framed)

        grid = stack_panels_into_grid(
            panel_frames_list, rows=rows, cols=cols,
            grid_title=f"{pose_name.upper()} pose - raw codes "
                       f"{chunk[0]}..{chunk[-1]} (z_e=0)",
            grid_banner_h=grid_banner_h,
        )
        out_path = video_dir / f"{pose_name}_raw_codes_{chunk[0]:03d}-{chunk[-1]:03d}.mp4"
        write_mp4(grid, out_path, fps=fps)
        log.info(f"    Wrote {out_path}")


# ---------------------------------------------------------------------------
# Merged grid render: one ghost_model per (pose, cluster), colored by code.
# ---------------------------------------------------------------------------


def _render_merged_grid_for_pose(
    env, pose_name: str,
    clusters: list[list[int]],
    rollouts: dict[int, dict[str, np.ndarray]],
    code_colors: np.ndarray, usage: np.ndarray,
    cfg: DictConfig, video_dir: Path,
    code_labels: dict[int, dict[str, Any]] | None = None,
    filename_prefix: str = "merged_clusters",
    title_kind: str = "merged clusters",
) -> None:
    """One merged-grid video per chunk of clusters. Each panel = one cluster.

    Within a panel: K_per_panel walkers per sub-code in the cluster, each
    walker colored by its sub-code (so a 3-code cluster shows 3 colors).
    """
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    panels_per_grid = rows * cols
    K = int(cfg.K_per_panel)
    panel_w, panel_h = int(cfg.panel_w), int(cfg.panel_h)
    panel_banner_h = int(cfg.panel_banner_h)
    grid_banner_h = int(cfg.grid_banner_h)
    fps = int(cfg.render_fps)

    chunks = [
        clusters[i:i + panels_per_grid]
        for i in range(0, len(clusters), panels_per_grid)
    ]

    for gi, chunk in enumerate(chunks):
        panel_frames_list: list[np.ndarray] = []
        for ci, cluster in enumerate(chunk):
            # Per-cluster ghost model: K * |cluster| walkers; each walker
            # is colored by its originating sub-code. The base body is
            # recolored to match the first sub-code so all walkers in the
            # cluster panel are uniformly visible.
            walker_colors: list[list[float]] = []
            for sub_code in cluster:
                walker_colors += [code_colors[int(sub_code)].tolist()] * K
            ghost_model, base_nq = build_uniform_walkers(
                env, walker_colors=walker_colors,
                camera_distance=float(cfg.camera_distance),
                camera_elevation=float(cfg.camera_elevation),
                camera_azimuth=float(cfg.camera_azimuth),
            )
            traj: list[np.ndarray] = []
            for sub_code in cluster:
                qpos = rollouts[int(sub_code)]["qpos"]
                for k in range(K):
                    traj.append(qpos[k % qpos.shape[0]])
            raw_frames = render_panel_frames(
                ghost_model, base_nq, traj,
                camera_name="divergent_cam",
                width=panel_w, height=panel_h,
            )

            code_str = ",".join(f"{c}" for c in cluster[:6])
            if len(cluster) > 6:
                code_str += "+"
            cluster_lab = (
                cluster_majority_label(cluster, code_labels)
                if code_labels is not None else "unlabeled"
            )
            lab_str = f" [{cluster_lab}]" if cluster_lab and cluster_lab != "unlabeled" else ""
            title = (
                f"cluster {gi * panels_per_grid + ci}{lab_str} [{code_str}]"
            )
            framed = np.stack([
                add_banner(
                    f, title, banner_h=panel_banner_h,
                    bg=(245, 245, 245),
                    fg=(35, 35, 35),
                    font_size=12,
                )
                for f in raw_frames
            ], axis=0)
            panel_frames_list.append(framed)

        if not panel_frames_list:
            continue
        # Force all panels to the same T (clip to min).
        T_min = min(p.shape[0] for p in panel_frames_list)
        panel_frames_list = [p[:T_min] for p in panel_frames_list]

        grid = stack_panels_into_grid(
            panel_frames_list, rows=rows, cols=cols,
            grid_title=f"{pose_name.upper()} pose - {title_kind} "
                       f"{gi * panels_per_grid}..{gi * panels_per_grid + len(chunk) - 1}",
            grid_banner_h=grid_banner_h,
        )
        out_path = (
            video_dir
            / f"{pose_name}_{filename_prefix}_g{gi:02d}.mp4"
        )
        write_mp4(grid, out_path, fps=fps)
        log.info(f"  [{pose_name}] {filename_prefix} grid {gi}: wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_grid_rollouts_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Grid Rollouts experiment ===")

    out_base = Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    data_dir = out_base / "data"
    videos_raw_dir = out_base / "videos_raw"
    merged_dir = out_base / "merged"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_raw_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint + inference fn ---
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path),
    )
    if not bool(ckpt_cfg.network_config.get("use_rnn_decoder", False)):
        raise ValueError(
            "This experiment requires an RNN decoder checkpoint."
        )
    # Override the checkpoint's saved codes_path if requested. This handles
    # the case where the tokenizer files have moved since the ckpt was
    # written.
    codes_override = cfg.checkpoint.get("codes_path_override", None)
    if codes_override is not None:
        ckpt_cfg.kpms_config.codes_path = _abs(codes_override)
        log.info(
            f"Overriding ckpt codes_path → {ckpt_cfg.kpms_config.codes_path}"
        )
    z_e_scale = float(cfg.get("z_e_scale", 0.0))
    inf_fn = make_inference_fn(
        ppo_networks, use_rnn=True,
        deterministic=True, z_e_scale=z_e_scale,
    )
    params = (norm_state, policy_params)
    log.info(f"Inference fn: deterministic=True, z_e_scale={z_e_scale}")

    # --- Codes + balanced split ---
    splits = load_balanced_splits(_abs(cfg.data.balanced_split_path))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    clip_len = int(ckpt_cfg.env_config.clip_length)
    codes_payload = load_codes(ckpt_cfg, train_indices, test_indices, clip_len)
    num_codes = int(codes_payload.num_codes)
    log.info(
        f"Codes: source={codes_payload.code_source}, num_codes={num_codes}"
    )

    # --- Rank codes by usage ---
    ranked_codes, usage = rank_codes_by_usage(
        codes_payload.train_codes, codes_payload.test_codes, num_codes,
    )
    top_n = min(num_codes, int(cfg.top_n_codes))
    top_codes = ranked_codes[:top_n].tolist()
    log.info(
        f"Top-{top_n} codes by usage (first 10 shown): {top_codes[:10]} "
        f"(usage range {int(usage[top_codes[0]])}..{int(usage[top_codes[-1]])})"
    )
    # Persist ranking for downstream interpretation.
    np.savez(
        data_dir / "usage_rank.npz",
        ranked_codes=ranked_codes, usage=usage, top_codes=np.asarray(top_codes),
    )

    # --- Pose-conditioned start clips ---
    label_arrays = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    behavior_arr = label_arrays.behavior

    # Per-code labels are computed AFTER rollouts (they require the
    # generated qpos), so this block lives further down. The labels CSV is
    # still loaded above because `_pick_start_clips` uses `behavior_arr` to
    # find clips matching the requested start pose.

    K = int(cfg.K_per_panel)
    probe_frame = int(cfg.start_pose.probe_frame)
    probe_frame_max = cfg.start_pose.get("probe_frame_max", None)
    if probe_frame_max is not None:
        probe_frame_max = int(probe_frame_max)
    seed = int(cfg.seed)

    walk_clips, walk_probe = _pick_start_clips(
        cfg.start_pose.walking_label, test_indices, behavior_arr,
        probe_frame=probe_frame, probe_frame_max=probe_frame_max,
        K=K, seed=seed,
    )
    rear_clips, rear_probe = _pick_start_clips(
        cfg.start_pose.rearing_label, test_indices, behavior_arr,
        probe_frame=probe_frame, probe_frame_max=probe_frame_max,
        K=K, seed=seed + 1,
    )
    log.info(
        f"Pose clips — walk @ frame {walk_probe}: {walk_clips.tolist()}"
    )
    log.info(
        f"Pose clips — rear @ frame {rear_probe}: {rear_clips.tolist()}"
    )

    # --- Build per-pose envs ---
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

    def _codes_for_pose(pose_clips: np.ndarray) -> np.ndarray:
        test_pos_lookup = {int(c): i for i, c in enumerate(test_indices)}
        rows = []
        for clip_id in pose_clips:
            pos = test_pos_lookup[int(clip_id)]
            rows.append(codes_payload.test_codes[pos])
        return np.stack(rows, axis=0)

    codes_walk = _codes_for_pose(walk_clips)
    codes_rear = _codes_for_pose(rear_clips)

    log.info("Building walking-pose env...")
    env_walk = _build_pose_env(
        ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
        keep_clips_idx=walk_clips, test_codes=codes_walk,
        code_stack_size=code_stack_size, probe_frame=walk_probe,
    )
    jit_reset_w = jax.jit(env_walk.reset)
    jit_step_w = jax.jit(env_walk.step)

    log.info("Building rearing-pose env...")
    env_rear = _build_pose_env(
        ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
        keep_clips_idx=rear_clips, test_codes=codes_rear,
        code_stack_size=code_stack_size, probe_frame=rear_probe,
    )
    jit_reset_r = jax.jit(env_rear.reset)
    jit_step_r = jax.jit(env_rear.step)

    # --- Rollouts (top_n × 2 poses × K bodies) ---
    hold_frames = int(cfg.hold_frames)
    reuse = bool(cfg.get("reuse_rollouts", False))
    log.info(
        f"\nRollouts: {top_n} codes × 2 poses × {K} bodies × {hold_frames} "
        f"frames (reuse_rollouts={reuse})"
    )
    log.info("  Walking pose rollouts ...")
    rollouts_walk = _rollouts_for_pose(
        env_walk, inf_fn, params, ppo_networks,
        walk_clips, top_codes, hold_frames, seed,
        jit_reset_w, jit_step_w, "walk", data_dir, reuse,
    )
    log.info("  Rearing pose rollouts ...")
    rollouts_rear = _rollouts_for_pose(
        env_rear, inf_fn, params, ppo_networks,
        rear_clips, top_codes, hold_frames, seed + 2000,
        jit_reset_r, jit_step_r, "rear", data_dir, reuse,
    )

    # --- Per-code labels from ROLLED-OUT qpos (TopoMimic auto-labeler) ---
    # Each (code, body) rollout's longest behavior segment + majority
    # direction within that span votes for the code's modal label. Pools
    # bodies from both walk and rear poses. Replaces the prior
    # corpus-frame-majority labeling which collapsed onto a handful of
    # over-frequent classes (Immobile_S / Rear_S).
    labels_dir = out_base / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    code_labels_path = labels_dir / "code_labels.json"
    reuse_labels = bool(cfg.get("reuse_labels", True))
    if reuse_labels and code_labels_path.exists():
        code_labels = load_code_labels_json(code_labels_path)
        log.info(f"Loaded cached code labels from {code_labels_path}")
    else:
        log.info(
            "Computing per-code rollout-derived labels via TopoMimic auto-labeler ..."
        )
        code_labels = label_codes_from_rollouts_dir(
            data_dir, top_codes, num_codes=num_codes,
            poses=("walk", "rear"),
            src_fps=100.0,
        )
        save_code_labels_json(
            code_labels_path, code_labels,
            source=f"rollouts:{data_dir}",
        )
        log.info(f"  Saved {code_labels_path}")
    top_label_preview = [
        (c, code_labels.get(int(c), {}).get("label", "unlabeled"))
        for c in top_codes[:10]
    ]
    log.info(f"  Top-10 code labels: {top_label_preview}")

    # --- Code colors (shared across raw and merged for visual continuity) ---
    code_colors = make_code_colors(num_codes)

    # --- Raw grid rendering (4 mp4s) ---
    if bool(cfg.get("render_video", True)):
        log.info("\n=== Rendering raw grids ===")
        _render_raw_grid_for_pose(
            env_walk, "walk", top_codes, rollouts_walk,
            code_colors, usage, cfg, videos_raw_dir,
            code_labels=code_labels,
        )
        _render_raw_grid_for_pose(
            env_rear, "rear", top_codes, rollouts_rear,
            code_colors, usage, cfg, videos_raw_dir,
            code_labels=code_labels,
        )

    # --- MMD merge per pose ---
    log.info("\n=== Computing MMD merge ===")
    reuse_merge = bool(cfg.get("reuse_merge", False))
    merge_cfg = cfg.merge
    pose_rollouts = {"walk": rollouts_walk, "rear": rollouts_rear}
    pose_clusters: dict[str, list[list[int]]] = {}
    pose_envs = {"walk": env_walk, "rear": env_rear}

    for pose_name, rollouts_pose in pose_rollouts.items():
        dist_path = merged_dir / f"distance_{pose_name}.npz"
        if reuse_merge and dist_path.exists():
            d = np.load(dist_path, allow_pickle=True)
            D = d["distance"]
            code_ids = list(d["code_ids"].tolist())
            sigma = float(d["sigma"])
            log.info(
                f"  [{pose_name}] reused cached MMD: D={D.shape}, sigma={sigma:.4g}"
            )
        else:
            qpos_per_code = {
                int(c): rollouts_pose[int(c)]["qpos"] for c in top_codes
            }
            D, code_ids, sigma = compute_mmd_distance_matrix(
                qpos_per_code,
                strip_root_pos=bool(merge_cfg.get("strip_root_pos", True)),
                kernel_sigma=merge_cfg.get("kernel_sigma", "median"),
            )
            np.savez(
                dist_path, distance=D, code_ids=np.asarray(code_ids),
                sigma=np.float32(sigma),
            )
            log.info(
                f"  [{pose_name}] MMD: D={D.shape}, sigma={sigma:.4g}, "
                f"median(D)={float(np.median(D)):.4f}"
            )

        clusters, info = cluster_codes(
            D, code_ids,
            cutoff=str(merge_cfg.get("cutoff", "elbow")),
            fixed_k=int(merge_cfg.get("fixed_k", 10)),
            frac_median=float(merge_cfg.get("frac_median", 0.5)),
        )
        save_clusters_json(
            merged_dir / f"clusters_{pose_name}.json",
            pose_name, clusters, info, usage,
        )
        # Augment clusters JSON with per-cluster majority label so the
        # downstream dynamics module can read it without recomputing.
        _augment_clusters_with_labels(
            merged_dir / f"clusters_{pose_name}.json", clusters, code_labels,
        )
        log.info(
            f"  [{pose_name}] {len(clusters)} clusters: sizes "
            f"{[len(c) for c in clusters[:10]]}..."
        )
        pose_clusters[pose_name] = clusters

    # --- Label-based clustering (pose-agnostic: codes with the same
    # majority behavior+direction tuple form one cluster regardless of
    # which pose the rollout starts from). We still write per-pose JSONs
    # so the downstream dynamics module can consume them with the same
    # naming convention as the MMD clusters; the partitions are identical. ---
    log.info("\n=== Label-based clustering ===")
    lbl_clusters, lbl_info = label_based_clusters(top_codes, code_labels)
    log.info(
        f"  label-clusters: "
        f"{[(lbl_info['labels'][i], len(c)) for i, c in enumerate(lbl_clusters)]}"
    )
    pose_label_clusters: dict[str, list[list[int]]] = {}
    for pose_name in ("walk", "rear"):
        save_clusters_json(
            merged_dir / f"clusters_by_label_{pose_name}.json",
            pose_name, lbl_clusters, lbl_info, usage,
        )
        _augment_clusters_with_labels(
            merged_dir / f"clusters_by_label_{pose_name}.json",
            lbl_clusters, code_labels,
        )
        pose_label_clusters[pose_name] = lbl_clusters

    # --- Merged grid rendering ---
    if bool(cfg.get("render_video", True)):
        log.info("\n=== Rendering MMD-merged grids ===")
        for pose_name in ("walk", "rear"):
            _render_merged_grid_for_pose(
                pose_envs[pose_name], pose_name,
                pose_clusters[pose_name], pose_rollouts[pose_name],
                code_colors, usage, cfg, merged_dir,
                code_labels=code_labels,
                filename_prefix="merged_clusters",
                title_kind="MMD merged clusters",
            )
        log.info("\n=== Rendering label-merged grids ===")
        for pose_name in ("walk", "rear"):
            _render_merged_grid_for_pose(
                pose_envs[pose_name], pose_name,
                pose_label_clusters[pose_name], pose_rollouts[pose_name],
                code_colors, usage, cfg, merged_dir,
                code_labels=code_labels,
                filename_prefix="merged_by_label",
                title_kind="label-merged clusters",
            )

    log.info(f"\n=== Done. Artifacts under {out_base}/ ===")
    log.info(f"  Rollouts:        {data_dir}/")
    log.info(f"  Raw grids (mp4): {videos_raw_dir}/")
    log.info(f"  Merged:          {merged_dir}/")
    log.info(f"  Labels:          {labels_dir}/")


def _augment_clusters_with_labels(
    path: Path,
    clusters: list[list[int]],
    code_labels: dict[int, dict[str, Any]],
) -> None:
    """Add ``label`` field to each cluster entry in a clusters JSON in place."""
    import json
    with path.open() as f:
        payload = json.load(f)
    for i, cl in enumerate(clusters):
        if i < len(payload.get("clusters", [])):
            payload["clusters"][i]["label"] = cluster_majority_label(
                cl, code_labels,
            )
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
