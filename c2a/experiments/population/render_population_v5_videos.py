"""Render v5 video products: per-cell × per-format 4-panel composite.

Layout per frame (top to bottom):
  [motion ghost | hidden PCA-3D rot | intention PCA-3D rot]   (3 panels)
  [code timeline bar — full width]                              (training-style)

3 highlighted walkers via tab10[0..2] (blue/orange/green), 97 grey points
in the PCA panels, ghost-rendered walkers in the motion panel with frontal
camera close-up (matches run_single_code's framing).

Run::

    python -m c2a.experiments.render_population_v5_videos
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
from PIL import Image, ImageDraw, ImageFont

from c2a.paths import get_repo_root

REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.code_grid import (
    build_uniform_walkers, render_panel_frames, add_banner, write_mp4,
)
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Code timeline bar — training-style color-coded per-frame code strip
# ---------------------------------------------------------------------------


def _build_code_color_map(num_codes: int) -> np.ndarray:
    """One distinct RGB color per code id, deterministically.

    Uses a hash-based hue mapping so the color is stable across runs.
    Returns ``(num_codes, 3)`` uint8.
    """
    cmap = plt.colormaps["hsv"]
    colors = np.zeros((num_codes, 3), dtype=np.uint8)
    for c in range(num_codes):
        hue = (c * 0.618033988749895) % 1.0   # golden-ratio hue stepping
        rgba = cmap(hue)
        colors[c] = (np.array(rgba[:3]) * 255).astype(np.uint8)
    return colors


def render_code_bar(
    fed_codes: np.ndarray, total_frames: int, num_codes: int,
    bar_w: int, bar_h: int,
    cursor_frame: int | None = None,
) -> np.ndarray:
    """Render one (bar_h, bar_w, 3) RGB frame of the code timeline.

    ``fed_codes`` has shape ``(T,)`` (the per-frame code being fed).
    If ``cursor_frame`` is provided, paint a vertical cursor line.
    """
    color_map = _build_code_color_map(num_codes)
    out = np.full((bar_h, bar_w, 3), 245, dtype=np.uint8)
    # Map each frame to a pixel column.
    px_per_frame = bar_w / max(total_frames, 1)
    for t in range(total_frames):
        c = int(fed_codes[min(t, len(fed_codes) - 1)])
        x0 = int(t * px_per_frame)
        x1 = int((t + 1) * px_per_frame)
        x1 = max(x1, x0 + 1)
        out[2:bar_h - 2, x0:x1] = color_map[c]
    # Cursor.
    if cursor_frame is not None:
        cx = int(cursor_frame * px_per_frame)
        cx = max(0, min(bar_w - 2, cx))
        out[:, cx:cx + 2] = (10, 10, 10)
    # Code legend strip at the top — small dots per unique code.
    used = sorted(set(int(c) for c in fed_codes.tolist()))
    # Draw small swatches at the top-right.
    swatch_h = 4
    dot_w = 8
    for i, c in enumerate(used[:30]):
        x = bar_w - (i + 1) * (dot_w + 2) - 4
        if x < 0:
            break
        out[1:1 + swatch_h, x:x + dot_w] = color_map[c]
    return out


# ---------------------------------------------------------------------------
# PCA-3D scatter panel
# ---------------------------------------------------------------------------


def render_pca_panel_frames(
    pca_traj: np.ndarray,           # (N, T, 3)
    highlights: np.ndarray,         # (n_hi,)
    highlight_colors: list[list[float]],
    title: str,
    height: int, width: int,
    rotation_degrees: float = 360.0,
    elev: float = 20.0,
) -> np.ndarray:
    N, T, _ = pca_traj.shape
    bg_mask = np.ones(N, dtype=bool)
    bg_mask[highlights] = False

    def _lim(a, pad=0.1):
        lo, hi = float(a.min()), float(a.max())
        span = max(hi - lo, 1e-3)
        return (lo - span * pad, hi + span * pad)
    xlim = _lim(pca_traj[:, :, 0]); ylim = _lim(pca_traj[:, :, 1]); zlim = _lim(pca_traj[:, :, 2])

    # STATIC structural cloud: all non-highlighted walkers × all time
    # frames, flattened to (N_bg × T, 3).  Drawn once per frame as the
    # population manifold — shows the PCA structure that the population
    # traces out across the full rollout, not just where they happen to
    # be at frame t.  The 3 highlighted walkers' trajectories evolve on
    # top of this static backdrop.
    bg_cloud = pca_traj[bg_mask, :, :].reshape(-1, 3)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    frames = np.zeros((T, height, width, 3), dtype=np.uint8)
    for t in range(T):
        ax.clear()
        azim = float(t) / max(T - 1, 1) * rotation_degrees
        # Static structural cloud — low alpha because there are many
        # points (~24k for 97 walkers × 250 frames).
        ax.scatter(
            bg_cloud[:, 0], bg_cloud[:, 1], bg_cloud[:, 2],
            c=[[0.6, 0.6, 0.6, 0.05]], s=4, edgecolors="none",
        )
        for hi, color in zip(highlights[:len(highlight_colors)], highlight_colors):
            traj = pca_traj[int(hi), :t + 1, :]
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
        if buf.shape[0] != height or buf.shape[1] != width:
            buf = np.asarray(
                Image.fromarray(buf).resize((width, height)), dtype=np.uint8,
            )
        frames[t] = buf
    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="configs",
    config_name="population_label_patterns",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info("v5 renderer — per-cell × per-format 4-panel composite")

    out_dir = Path(_abs(cfg.output.base_dir))
    data_dir = out_dir / "data"
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ckpt + env ------------------------------------------------
    ckpt_cfg, _, _, _ = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path), step=cfg.checkpoint.get("step", None),
    )
    cfg_local, _, env_cfg_ml = utils.prepare_config(ckpt_cfg)
    ref_path = REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"
    if ref_path.exists():
        cfg_local.env_config.reference_data_path = str(ref_path)
        env_cfg_ml.reference_data_path = str(ref_path)
    splits = load_balanced_splits(_abs(cfg.paths.balanced_splits))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    clips_train = ReferenceClips(
        data_path=str(env_cfg_ml.reference_data_path),
        n_frames_per_clip=int(env_cfg_ml.clip_length),
        keep_clips_idx=train_indices,
    )
    code_stack_size = int(ckpt_cfg.network_config.code_stack_size)
    num_codes = int(ckpt_cfg.network_config.num_codes)
    dummy_codes = jnp.zeros(
        (clips_train.qpos.shape[0], clips_train.qpos.shape[1]),
        dtype=jnp.int32,
    )
    env_cfg_ml_train = env_cfg_ml.copy_and_resolve_references()
    env_cfg_ml_train.start_frame_range = [0, 0]
    env = MoSeqImitation(
        config=env_cfg_ml_train, clips=clips_train, kpms_codes=dummy_codes,
        code_stack_size=code_stack_size, c2c_adapter=None, c2c_reward_mode="off",
    )

    # tab10 first 3 colors with full alpha.
    cam = cfg.video.camera
    highlight_colors_rgba = [list(c) + [1.0] for c in cfg.highlight.highlight_colors]
    ghost_model, base_nq = build_uniform_walkers(
        env, walker_colors=highlight_colors_rgba,
        camera_distance=float(cam.distance),
        camera_elevation=float(cam.elevation),
        camera_azimuth=float(cam.azimuth),
    )

    # --- PCA basis ------------------------------------------------------
    pca_basis = np.load(data_dir / "pca_basis.npz")
    pca_h_c = pca_basis["hidden_components"]; pca_h_mean = pca_basis["hidden_mean"]
    pca_i_c = pca_basis["intention_components"]; pca_i_mean = pca_basis["intention_mean"]

    def project(x: np.ndarray, comps: np.ndarray, mean: np.ndarray) -> np.ndarray:
        N, T, D = x.shape
        return ((x.reshape(-1, D) - mean) @ comps.T).reshape(N, T, 3)

    panel_w = int(cfg.video.panel_w); panel_h = int(cfg.video.panel_h)
    code_bar_h = int(cfg.video.code_bar_h)
    banner_h = int(cfg.video.banner_h)
    fps = int(cfg.video.fps)
    rot = float(cfg.video.pca_rotation_degrees)

    # --- Iterate over npz files (cells × formats) -----------------------
    highlights_npz = np.load(data_dir / "highlights.npz")
    highlight_colors = [list(c) for c in cfg.highlight.highlight_colors]
    npz_files = sorted(data_dir.glob("cell_*_fmt*.npz"))
    log.info("Found %d (cell × format) npz files", len(npz_files))

    for npz in npz_files:
        with np.load(npz, allow_pickle=True) as d:
            qpos = np.asarray(d["qpos"])         # (N, T, nq)
            hidden = np.asarray(d["hidden"])     # (N, T, H)
            intention = np.asarray(d["intention"])  # (N, T, I)
            fed_codes = np.asarray(d["fed_code_seq"])
            cell_name = str(d["combination_name"])
            fmt = str(d["format"])
        key = f"{cell_name.replace(' ', '_').replace('→', 'arrow')}_fmt{fmt}"
        # Skip if already rendered — allows clean resume after a crash.
        out_path_check = video_dir / f"{key}.mp4"
        if out_path_check.exists() and out_path_check.stat().st_size > 0:
            log.info("  skip %s (already exists)", out_path_check.name)
            continue
        highlights = (
            np.asarray(highlights_npz[key])
            if key in highlights_npz.files
            else np.arange(min(3, qpos.shape[0]))
        )
        log.info("rendering %s (Format %s)  highlights=%s",
                 cell_name, fmt, highlights.tolist())

        T = qpos.shape[1]
        n_hi = min(len(highlight_colors), len(highlights))

        # 1) motion ghost frames — 3 walkers, frontal camera close-up
        motion_frames = render_panel_frames(
            ghost_model, base_nq,
            [qpos[int(h)] for h in highlights[:n_hi]],
            camera_name="divergent_cam",
            width=panel_w, height=panel_h,
        )

        # 2) hidden PCA
        h_pca = project(hidden, pca_h_c, pca_h_mean)
        hidden_frames = render_pca_panel_frames(
            h_pca, highlights, highlight_colors,
            title="hidden h_t (PCA-3D)",
            height=panel_h, width=panel_w, rotation_degrees=rot,
        )

        # 3) intention PCA
        i_pca = project(intention, pca_i_c, pca_i_mean)
        intention_frames = render_pca_panel_frames(
            i_pca, highlights, highlight_colors,
            title="intention vae_mean (PCA-3D)",
            height=panel_h, width=panel_w, rotation_degrees=rot,
        )

        T_common = min(motion_frames.shape[0], hidden_frames.shape[0], intention_frames.shape[0])
        bar_w = 3 * panel_w  # full row width

        # 4) per-frame code timeline bar.  num_codes coming from the
        # base config may be smaller than max(fed_codes)+1 when the data
        # was produced with a tokenizer that has more codes than the
        # default ckpt (e.g. K=25 VAE data rendered against the K=20
        # default config).  Bump it locally to fit so the color-map
        # lookup doesn't IndexError.
        bar_num_codes = max(int(num_codes), int(np.max(fed_codes)) + 1)
        code_bars = np.stack([
            render_code_bar(
                fed_codes, total_frames=T_common, num_codes=bar_num_codes,
                bar_w=bar_w, bar_h=code_bar_h, cursor_frame=t,
            )
            for t in range(T_common)
        ], axis=0)

        # Compose: top row = motion | hidden | intention
        top_row = np.concatenate(
            [motion_frames[:T_common], hidden_frames[:T_common], intention_frames[:T_common]],
            axis=2,
        )                                                     # (T, panel_h, 3*panel_w, 3)
        composite = np.concatenate(
            [top_row, code_bars[:, :, :3 * panel_w, :]], axis=1,
        )                                                     # (T, panel_h + code_bar_h, 3*panel_w, 3)

        banner_text = f"{cell_name}   Format {fmt}"
        banner_frames = np.stack(
            [add_banner(composite[t],
                        f"{banner_text}   frame {t}/{T_common}",
                        banner_h=banner_h, font_size=16)
             for t in range(T_common)],
            axis=0,
        )

        out_path = video_dir / f"{key}.mp4"
        write_mp4(banner_frames, out_path, fps=fps)
        log.info("  wrote %s", out_path)

    log.info("Done.")


if __name__ == "__main__":
    main()
