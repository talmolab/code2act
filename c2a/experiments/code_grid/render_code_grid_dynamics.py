"""Render the 3 code-grid hidden-dynamics composite videos.

Companion to ``run_code_grid_dynamics.py``. Reads the cached **walk** rollouts
+ cluster JSONs from ``run_code_grid_rollouts`` and produces three
ghost-bodies + rotating-hidden-PCA composites (one shared 3-D hidden PCA basis
so all three share axes):

1. ``mmd.mp4``   — groups = MMD-merged clusters. 5 highlighted walkers/cluster
   (top-usage codes, body 0), ghost-rendered + full-color hidden trajectories,
   colored by cluster; the rest of each cluster's walkers as a faint per-cluster
   hidden point cloud.
2. ``label.mp4`` — same, with label-majority clusters.
3. ``raw.mp4``   — groups = codes. One walker per code (body 0), colored per
   code, legend **beneath** the video (code id + behavior label); the rest of
   each code's walkers as a faint per-code cloud.

Outputs under ``dynamics_output_base_dir/<run_name>/videos/``.

Usage:
    cd code2act
    python -m c2a.experiments.code_grid.render_code_grid_dynamics \
        output.run_name=best_vae \
        checkpoint.path=$C2A_ROOT/model_checkpoints/260603_180852_140475
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import logging
from pathlib import Path

import hydra
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

from c2a.codegen.code_sources import load_codes
from c2a.codegen.shared.label_alignment import load_label_frames
from c2a.experiments.code_grid.run_code_grid_dynamics import _load_rollouts
from c2a.experiments.code_grid.run_code_grid_rollouts import _pick_random_clips
from c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled import (
    _pick_start_clips, _build_pose_env,
)
from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.code_labels import load_code_labels_json
from c2a.experiments.shared.rotating_pca_video import (
    fit_joint_pca, _compute_pca_limits, render_grouped_rotating_pca_video,
)
from c2a.paths import get_repo_root

REPO_ROOT = get_repo_root()
log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _top_codes(cluster: list[int], usage: np.ndarray, n: int) -> list[int]:
    return sorted((int(c) for c in cluster), key=lambda c: -float(usage[c]))[:n]


def _build_groups_for_clusters(
    clusters: list[list[int]],
    cluster_labels: list[str],
    rollouts: dict[int, dict[str, np.ndarray]],
    usage: np.ndarray,
    walkers_per_group: int,
):
    """For cluster groups: highlighted = top-usage codes, EACH from a distinct
    start clip (the j-th highlighted walker uses body j, so they begin at their
    true different init positions, not all stacked at body 0); bg = the rest."""
    g_qpos, g_hidden, g_bg, g_names = [], [], [], []
    cmap = plt.get_cmap("tab20")
    g_colors = []
    for gi, cluster in enumerate(clusters):
        sel = _top_codes(cluster, usage, walkers_per_group)
        hi_h, hi_q, hi_pairs = [], [], set()
        for j, code in enumerate(sel):
            h = rollouts[int(code)]["hidden"]
            b = j % h.shape[0]  # distinct start clip per highlighted walker
            hi_h.append(h[b])
            hi_q.append(rollouts[int(code)]["qpos"][b])
            hi_pairs.add((int(code), b))
        # Background: every (code, body) in the cluster except the highlighted.
        bg = []
        for code in cluster:
            h = rollouts[int(code)]["hidden"]  # (K, T, H)
            for b in range(h.shape[0]):
                if (int(code), b) in hi_pairs:
                    continue
                bg.append(h[b].reshape(-1, h.shape[-1]))
        g_qpos.append(np.stack(hi_q, axis=0))
        g_hidden.append(np.stack(hi_h, axis=0))
        g_bg.append(np.concatenate(bg, axis=0) if bg else np.zeros((0, hi_h[0].shape[-1])))
        g_names.append(f"c{gi} [{cluster_labels[gi]}] (n={len(cluster)})")
        g_colors.append(cmap(gi % 20))
    return g_qpos, g_hidden, g_bg, g_colors, g_names


def _build_groups_for_codes(
    top_codes: list[int],
    code_labels: dict[int, dict],
    rollouts: dict[int, dict[str, np.ndarray]],
):
    """Raw video: one group per code. The highlighted walker for code i uses a
    DISTINCT start clip (body i % K), so the 25 codes' walkers begin at their
    true different init positions rather than all stacked at body 0; bg = that
    code's other bodies."""
    g_qpos, g_hidden, g_bg, g_names, g_colors = [], [], [], [], []
    cmap = plt.get_cmap("gist_ncar")
    n = len(top_codes)
    for i, code in enumerate(top_codes):
        h = rollouts[int(code)]["hidden"]  # (K, T, H)
        q = rollouts[int(code)]["qpos"]
        b = i % h.shape[0]  # distinct start clip per code
        g_qpos.append(q[b:b + 1])
        g_hidden.append(h[b:b + 1])
        bg = (np.delete(h, b, axis=0).reshape(-1, h.shape[-1])
              if h.shape[0] > 1 else np.zeros((0, h.shape[-1])))
        g_bg.append(bg)
        lab = code_labels.get(int(code), {}).get("label", "unlabeled")
        g_names.append(f"{int(code)}:{lab}")
        g_colors.append(cmap(i / max(n - 1, 1)))
    return g_qpos, g_hidden, g_bg, g_colors, g_names


@hydra.main(version_base=None, config_path="configs",
            config_name="code_grid_rollouts_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Render Code-Grid Hidden-Dynamics Composites ===")
    pose = str(cfg.get("poses", ["walk"])[0])

    rollouts_base = Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    data_dir = rollouts_base / "data"
    merged_dir = rollouts_base / "merged"
    labels_dir = rollouts_base / "labels"
    if not data_dir.exists():
        raise RuntimeError(f"No rollouts under {rollouts_base}. Run rollouts first.")

    usage_data = np.load(data_dir / "usage_rank.npz")
    top_codes = [int(c) for c in usage_data["top_codes"].tolist()]
    usage = usage_data["usage"]
    code_labels = load_code_labels_json(labels_dir / "code_labels.json")

    out_dir = (
        Path(_abs(cfg.get("dynamics_output_base_dir",
                          "./outputs/code_grid/code_grid_dynamics")))
        / str(cfg.output.run_name) / "videos"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rcfg = cfg.get("render", {})
    walkers_per_group = int(rcfg.get("walkers_per_group", 5))
    frame_stride = int(rcfg.get("frame_stride", 4))
    fps = int(rcfg.get("fps", 30))
    panel_w = int(rcfg.get("panel_w", 640))
    panel_h = int(rcfg.get("panel_h", 480))

    # --- Cached walk rollouts ---
    log.info(f"Loading {len(top_codes)} cached {pose} rollouts ...")
    rollouts = _load_rollouts(data_dir, pose, top_codes)

    # --- Shared 3-D hidden PCA across all walk rollouts ---
    hidden_list = [rollouts[int(c)]["hidden"] for c in top_codes]
    pca, var = fit_joint_pca(hidden_list, n_components=3)
    emb_all = pca.transform(
        np.concatenate([h.reshape(-1, h.shape[-1]) for h in hidden_list], axis=0)
    )
    xlim, ylim, zlim = _compute_pca_limits(emb_all)
    log.info(f"Hidden PCA var explained: {var[:3]}")

    # --- Build the ghost env (walk pose), mirroring run_code_grid_rollouts ---
    ckpt_cfg, _norm, _pp, _nets = load_moseq_checkpoint(_abs(cfg.checkpoint.path))
    codes_override = cfg.checkpoint.get("codes_path_override", None)
    if codes_override is not None:
        ckpt_cfg.kpms_config.codes_path = _abs(codes_override)
    splits = load_balanced_splits(_abs(cfg.data.balanced_split_path))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    clip_len = int(ckpt_cfg.env_config.clip_length)
    codes_payload = load_codes(ckpt_cfg, train_indices, test_indices, clip_len)
    label_arrays = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    K = int(cfg.K_per_panel)
    # Reproduce the same start clips the rollouts used (same seed) so the env
    # matches. The env is only used to build the ghost body model.
    if pose == "random":
        env_clips, env_probe = _pick_random_clips(
            test_indices, K=K, seed=int(cfg.seed) + 4000,
            probe_frame=int(cfg.start_pose.probe_frame),
        )
    else:
        label = (cfg.start_pose.walking_label if pose == "walk"
                 else cfg.start_pose.rearing_label)
        env_clips, env_probe = _pick_start_clips(
            label, test_indices, label_arrays.behavior,
            probe_frame=int(cfg.start_pose.probe_frame),
            probe_frame_max=cfg.start_pose.get("probe_frame_max", None),
            K=K, seed=int(cfg.seed) + (0 if pose == "walk" else 1),
        )
    test_pos = {int(c): i for i, c in enumerate(test_indices)}
    codes_env = np.stack([codes_payload.test_codes[test_pos[int(c)]]
                          for c in env_clips], axis=0)
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    log.info(f"Building {pose}-pose ghost env ...")
    env = _build_pose_env(
        ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
        keep_clips_idx=env_clips, test_codes=codes_env,
        code_stack_size=code_stack_size, probe_frame=env_probe,
    )

    common = dict(
        pca=pca, var_explained=var, xlim=xlim, ylim=ylim, zlim=zlim,
        ghost_env=env, fps=fps, panel_w=panel_w, panel_h=panel_h,
        frame_stride=frame_stride,
    )

    # ── Video 1 + 2: MMD / label merged clusters ───────────────────────────
    for strategy, fname, vid in [
        ("MMD", f"clusters_{pose}.json", "mmd.mp4"),
        ("label", f"clusters_by_label_{pose}.json", "label.mp4"),
    ]:
        path = merged_dir / fname
        if not path.exists():
            log.warning(f"  {strategy}: {path} missing; skipping {vid}.")
            continue
        with path.open() as f:
            payload = json.load(f)
        clusters = [c["codes"] for c in payload["clusters"]]
        labels = [c.get("label", "?") for c in payload["clusters"]]
        gq, gh, gbg, gc, gn = _build_groups_for_clusters(
            clusters, labels, rollouts, usage, walkers_per_group)
        log.info(f"Rendering {vid}: {len(clusters)} clusters, "
                 f"{sum(x.shape[0] for x in gq)} highlighted bodies ...")
        render_grouped_rotating_pca_video(
            group_qpos=gq, group_hidden=gh, group_bg_hidden=gbg,
            group_colors=gc, group_names=gn, output_path=out_dir / vid,
            legend_below=True,
            title=f"{strategy}-merged clusters — walk", **common,
        )

    # ── Video 3: raw (one walker per code) ──────────────────────────────────
    gq, gh, gbg, gc, gn = _build_groups_for_codes(top_codes, code_labels, rollouts)
    log.info(f"Rendering raw.mp4: {len(top_codes)} codes (1 walker each) ...")
    render_grouped_rotating_pca_video(
        group_qpos=gq, group_hidden=gh, group_bg_hidden=gbg,
        group_colors=gc, group_names=gn, output_path=out_dir / "raw.mp4",
        legend_below=True, title="All codes (one walker/code) — walk", **common,
    )

    log.info(f"\n=== Done. Videos under {out_dir}/ ===")


if __name__ == "__main__":
    main()
