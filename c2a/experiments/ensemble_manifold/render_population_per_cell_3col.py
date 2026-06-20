"""Render per-cell 3-column composite videos for the population-dynamics
experiment.

For each of the 12 cells: ``[motion ghost | hidden PCA-3D | intention
PCA-3D]`` composited side-by-side, 3 walkers ghost-rendered + 3
highlighted PCA trajectories + 97 grey-dot background per panel.  Reads
cached per-cell qpos + hidden + intention npz from
``outputs/population_intention_dynamics/data/`` and the joint PCA basis
fitted by ``run_population_intention_dynamics``.

Run::

    cd code2act
    python -m c2a.experiments.ensemble_manifold.render_population_per_cell_3col
"""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c2a.paths import get_repo_root

REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.code_grid import (
    build_uniform_walkers,
    render_panel_frames,
    add_banner,
    write_mp4,
)
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# PCA panel rendering — return (T, H, W, 3) uint8 numpy
# ---------------------------------------------------------------------------


def _pca_limits(arr: np.ndarray, pad: float = 0.1) -> tuple:
    lo, hi = float(arr.min()), float(arr.max())
    span = max(hi - lo, 1e-3)
    pad_abs = span * pad
    return (lo - pad_abs, hi + pad_abs)


def render_pca_panel_frames(
    pca_traj: np.ndarray,           # (N, T, 3) — all walkers' PCA-3D trajectory
    highlights: np.ndarray,         # (n_hi,) walker indices
    highlight_colors: list[list[float]],
    title: str,
    height: int, width: int,
    rotation_degrees: float = 360.0,
    elev: float = 20.0,
) -> np.ndarray:
    """Render a per-frame PCA-3D scatter into ``(T, H, W, 3)`` uint8.

    Background = all non-highlighted walkers' positions at frame t (grey
    dots).  Foreground = highlighted walkers' trajectories from t=0 to t,
    plus a marker at their current position.  Azimuth rotates linearly
    from 0 → ``rotation_degrees`` over the T frames.
    """
    N, T, _ = pca_traj.shape
    bg_mask = np.ones(N, dtype=bool)
    bg_mask[highlights] = False

    xlim = _pca_limits(pca_traj[:, :, 0])
    ylim = _pca_limits(pca_traj[:, :, 1])
    zlim = _pca_limits(pca_traj[:, :, 2])

    dpi = 100
    figsize = (width / dpi, height / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    frames = np.zeros((T, height, width, 3), dtype=np.uint8)
    for t in range(T):
        ax.clear()
        azim = float(t) / max(T - 1, 1) * rotation_degrees
        # Background grey dots at frame t.
        ax.scatter(
            pca_traj[bg_mask, t, 0],
            pca_traj[bg_mask, t, 1],
            pca_traj[bg_mask, t, 2],
            c=[[0.6, 0.6, 0.6, 0.25]], s=8, edgecolors="none",
        )
        # Highlighted walker trajectories (0..t).
        for hi, color in zip(highlights[:len(highlight_colors)], highlight_colors):
            traj = pca_traj[hi, :t + 1, :]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=color, linewidth=2.0, alpha=0.95)
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                       color=color, s=40, edgecolors="black", linewidths=0.8)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        # Buffer comes back at canvas size — should match (height, width).
        if buf.shape[0] != height or buf.shape[1] != width:
            # Defensive resize if matplotlib's dpi math gives off-by-one.
            from PIL import Image
            buf = np.asarray(
                Image.fromarray(buf).resize((width, height)),
                dtype=np.uint8,
            )
        frames[t] = buf
    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="configs",
    config_name="population_intention_dynamics",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("Render population per-cell 3-column composites")

    out_dir = Path(_abs(cfg.output.base_dir))
    data_dir = out_dir / "data"
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ckpt + env (for ghost model) -------------------------------
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

    highlight_colors = [list(c) for c in cfg.highlight.highlight_colors]
    highlight_colors_rgba = [c + [1.0] for c in highlight_colors]
    ghost_model, base_nq = build_uniform_walkers(
        env, walker_colors=highlight_colors_rgba,
        camera_distance=0.8, camera_elevation=-30.0, camera_azimuth=135.0,
    )

    # --- Load PCA basis ---------------------------------------------------
    pca_basis = np.load(data_dir / "pca_basis.npz")
    pca_h_components = pca_basis["hidden_components"]    # (3, H_dim)
    pca_h_mean = pca_basis["hidden_mean"]                # (H_dim,)
    pca_i_components = pca_basis["intention_components"] # (3, I_dim)
    pca_i_mean = pca_basis["intention_mean"]             # (I_dim,)

    def project(x: np.ndarray, comps: np.ndarray, mean: np.ndarray) -> np.ndarray:
        # x: (N, T, D) → (N, T, 3)
        N, T, D = x.shape
        return ((x.reshape(-1, D) - mean) @ comps.T).reshape(N, T, 3)

    # --- Per-cell rendering ----------------------------------------------
    # Use larger panels for "high resolution" — the user asked for this.
    panel_w = 480
    panel_h = 480
    banner_h = 36
    fps = int(cfg.video.fps)
    rot_deg = float(cfg.video.pca_rotation_degrees)

    camera_name = "divergent_cam"

    npz_files = sorted(data_dir.glob("cell_*.npz"))
    log.info("Found %d cell npz files", len(npz_files))

    highlights_npz = np.load(data_dir / "highlights.npz")

    for npz in npz_files:
        with np.load(npz, allow_pickle=True) as d:
            qpos = np.asarray(d["qpos"])              # (N, T, nq)
            hidden = np.asarray(d["hidden"])          # (N, T, H_dim)
            intention = np.asarray(d["intention"])    # (N, T, I_dim)
            cell_name = (
                str(d["combination_name"])
                if "combination_name" in d.files
                else npz.stem.split("_", 2)[2].replace("arrow", "→").replace("_", " ")
            )
        key = cell_name.replace(" ", "_").replace("→", "arrow")
        highlights = (
            np.asarray(highlights_npz[key])
            if key in highlights_npz.files
            else np.arange(3)
        )
        log.info("rendering cell %r  (highlights=%s)", cell_name, highlights.tolist())

        T = qpos.shape[1]
        n_hi = min(len(highlight_colors), len(highlights))

        # Motion ghost frames — base body uses walker_colors[0], ghosts use [1:].
        motion_frames = render_panel_frames(
            ghost_model, base_nq,
            [qpos[int(h)] for h in highlights[:n_hi]],
            camera_name=camera_name,
            width=panel_w, height=panel_h,
        )  # (T, H, W, 3)

        # Project hidden + intention.
        hidden_pca = project(hidden, pca_h_components, pca_h_mean)         # (N, T, 3)
        intention_pca = project(intention, pca_i_components, pca_i_mean)   # (N, T, 3)

        # PCA panel frames.
        hidden_frames = render_pca_panel_frames(
            hidden_pca, highlights, highlight_colors,
            title="hidden h_t (PCA-3D)",
            height=panel_h, width=panel_w,
            rotation_degrees=rot_deg,
        )
        intention_frames = render_pca_panel_frames(
            intention_pca, highlights, highlight_colors,
            title="intention z_e μ (PCA-3D)",
            height=panel_h, width=panel_w,
            rotation_degrees=rot_deg,
        )

        T_common = min(motion_frames.shape[0], hidden_frames.shape[0], intention_frames.shape[0])

        # Concatenate horizontally per frame.
        composite = np.concatenate(
            [motion_frames[:T_common], hidden_frames[:T_common], intention_frames[:T_common]],
            axis=2,
        )  # (T_common, H, 3*W, 3)

        # Add banner per frame with cell name + frame counter.
        banner_frames = np.stack(
            [add_banner(composite[t],
                        f"{cell_name}   frame {t}/{T_common}",
                        banner_h=banner_h, font_size=16)
             for t in range(T_common)],
            axis=0,
        )

        out_path = video_dir / f"{key}_3col.mp4"
        write_mp4(banner_frames, out_path, fps=fps)
        log.info("  wrote %s", out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
