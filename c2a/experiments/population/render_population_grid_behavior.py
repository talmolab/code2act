"""Render the behavior-grid video for ``run_population_intention_dynamics``.

Reads the cached per-cell npz files in
``outputs/population_intention_dynamics/data/`` and renders a 3×4 grid
video: each cell shows 3 ghost-rendered walkers (the 3 spread-by-PCA
highlights) moving simultaneously, all panels in sync per frame.

This is the "behavior catalog" video the user asked for in the
population-dynamics design — companion to the PCA scatter videos that
were already rendered.

Run::

    cd code2act
    python -m c2a.experiments.population.render_population_grid_behavior
"""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Orbax 0.11.24 vs JAX 0.5.3 mismatch — see run_population_intention_dynamics.
import jax.monitoring
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *_a, **_kw: None

import logging
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from c2a.paths import get_c2a_root, get_repo_root

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.code_grid import (
    build_uniform_walkers,
    render_panel_frames,
    add_banner,
    stack_panels_into_grid,
    write_mp4,
)
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


@hydra.main(
    config_path="configs",
    config_name="population_intention_dynamics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("Render population grid-behavior video")

    out_dir = Path(_abs(cfg.output.base_dir))
    data_dir = out_dir / "data"
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load ckpt + env (for ghost model and rendering) -------------
    log.info("Loading checkpoint from %s", cfg.checkpoint.path)
    ckpt_cfg, _, _, _ = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path), step=cfg.checkpoint.get("step", None),
    )

    cfg_local, _, env_cfg_ml = utils.prepare_config(ckpt_cfg)
    ref_data_path = REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"
    if ref_data_path.exists():
        cfg_local.env_config.reference_data_path = str(ref_data_path)
        env_cfg_ml.reference_data_path = str(ref_data_path)

    splits = load_balanced_splits(_abs(cfg.paths.balanced_splits))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)

    clips_train = ReferenceClips(
        data_path=str(env_cfg_ml.reference_data_path),
        n_frames_per_clip=int(env_cfg_ml.clip_length),
        keep_clips_idx=train_indices,
    )

    # Codes get a dummy zero array — we don't use them in this render, but
    # MoSeqImitation requires the same shape as the segmenter output.
    code_stack_size = int(ckpt_cfg.network_config.code_stack_size)
    dummy_codes = jnp.zeros(
        (clips_train.qpos.shape[0], clips_train.qpos.shape[1]),
        dtype=jnp.int32,
    )

    env_cfg_ml_train = env_cfg_ml.copy_and_resolve_references()
    env_cfg_ml_train.start_frame_range = [0, 0]
    env = MoSeqImitation(
        config=env_cfg_ml_train,
        clips=clips_train,
        kpms_codes=dummy_codes,
        code_stack_size=code_stack_size,
        c2c_adapter=None,
        c2c_reward_mode="off",
    )

    # --- 2. Build ghost model (3 walkers with our highlight colors) -----
    highlight_colors = [list(c) + [1.0] for c in cfg.highlight.highlight_colors]
    ghost_model, base_nq = build_uniform_walkers(
        env, walker_colors=highlight_colors,
        camera_distance=0.8, camera_elevation=-30.0, camera_azimuth=135.0,
    )
    log.info("ghost model built: %d walkers, base_nq=%d",
             len(highlight_colors), base_nq)

    # --- 3. For each cell, render the 3 highlighted walkers --------------
    panel_w = int(cfg.video.grid.cell_w)
    panel_h = int(cfg.video.grid.cell_h)
    banner_h = int(cfg.video.grid.panel_banner_h)
    grid_banner_h = int(cfg.video.grid.grid_banner_h)

    # The "close_profile" camera in the ckpt config doesn't exist in the
    # ghost model.  Use the same camera that `run_code_grid_rollouts`
    # uses for its grid videos.
    camera_name = "divergent_cam"

    panel_frames_list: list[np.ndarray] = []
    cell_order: list[str] = []

    # The npz files are named cell_NN_{name_with_arrow}.npz; iterate in
    # sorted-index order to match the config's combination ordering.
    npz_files = sorted(data_dir.glob("cell_*.npz"))
    log.info("Found %d cell npz files", len(npz_files))

    for npz in npz_files:
        with np.load(npz, allow_pickle=True) as d:
            qpos = d["qpos"]                     # (N, T, nq)
            highlights = None
            # Try to recover cell name from the saved string.
            cell_name = (
                str(d["combination_name"])
                if "combination_name" in d.files
                else npz.stem.split("_", 2)[2].replace("arrow", "→").replace("_", " ")
            )
        # Load the highlights from the global highlights.npz.
        with np.load(data_dir / "highlights.npz") as h:
            key = cell_name.replace(" ", "_").replace("→", "arrow")
            if key in h.files:
                highlights = np.asarray(h[key])
            else:
                highlights = np.arange(3)
        log.info("rendering cell %r  (highlights=%s)", cell_name, highlights.tolist())

        T = qpos.shape[1]
        nq = qpos.shape[-1]
        trajectories_qpos = [np.asarray(qpos[int(h)]) for h in highlights[:3]]

        frames = render_panel_frames(
            ghost_model, base_nq, trajectories_qpos,
            camera_name=camera_name,
            width=panel_w, height=panel_h,
        )
        # Add banner with cell name.
        banner_frames = np.stack(
            [add_banner(frames[t], cell_name, banner_h=banner_h)
             for t in range(frames.shape[0])],
            axis=0,
        )
        panel_frames_list.append(banner_frames)
        cell_order.append(cell_name)

    # --- 4. Stack into grid + write ------------------------------------
    if not panel_frames_list:
        log.error("No panels rendered; aborting")
        return

    rows = int(cfg.grid.rows)
    cols = int(cfg.grid.cols)
    log.info("Stacking %d panels into a %d×%d grid", len(panel_frames_list), rows, cols)

    # All panels must have the same shape; pad if needed.
    target_T = min(p.shape[0] for p in panel_frames_list)
    padded = [p[:target_T] for p in panel_frames_list]

    grid_frames = stack_panels_into_grid(
        padded, rows=rows, cols=cols,
        grid_title="Population intention dynamics — 3 walkers per behavior cell",
        grid_banner_h=grid_banner_h,
    )

    out_path = video_dir / "grid_behavior_motion.mp4"
    write_mp4(grid_frames, out_path, fps=int(cfg.video.fps))
    log.info("wrote %s  (%d frames, %s)",
             out_path, grid_frames.shape[0], grid_frames.shape[1:])


if __name__ == "__main__":
    main()
