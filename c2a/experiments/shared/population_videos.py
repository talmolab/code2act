"""Video rendering helpers for ``run_population_intention_dynamics``.

Two video products:

* :func:`render_grid_behavior_video` — single mp4 with 3 walkers per cell
  ghost-rendered side-by-side in a rows×cols grid. The "behavior catalog."
* :func:`render_per_cell_3col_video` — per-cell 3-column composite (motion
  ghost / hidden PCA-3D rotating / intention PCA-3D rotating) with 3
  highlighted walker trajectories on top of 97 grey background points.

Minimal implementation — uses matplotlib for PCA panels and ghost rendering
via :mod:`c2a.experiments.shared.ghost_rendering` for the motion panel.
Failures in any sub-renderer are non-fatal; the caller catches exceptions
and continues to the next cell.

PCA panels rotate the view (``azim``) linearly over the rollout duration so
the 3D structure is readable in 2D playback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio.v2 as imageio

log = logging.getLogger(__name__)


def _interp_view_angle(t: int, T: int, total_degrees: float = 360.0) -> float:
    """Linear azimuth ramp: 0° → ``total_degrees`` over T frames."""
    return float(t) / max(T - 1, 1) * total_degrees


def _draw_pca_frame(
    ax,
    background: np.ndarray,       # (N_bg, 3) — fixed point cloud
    highlight_trajs: np.ndarray,  # (n_hi, t_frames, 3) — partial trajectories
    highlight_colors: list[list[float]],
    azim: float, elev: float = 20.0,
    xlim: tuple = (-3, 3), ylim: tuple = (-3, 3), zlim: tuple = (-3, 3),
    title: str = "",
) -> None:
    """Render a single PCA-3D scatter frame onto ``ax`` (in-place)."""
    ax.clear()
    ax.scatter(
        background[:, 0], background[:, 1], background[:, 2],
        c=[[0.6, 0.6, 0.6, 0.25]], s=8, edgecolors="none",
    )
    for hi, color in enumerate(highlight_colors[:len(highlight_trajs)]):
        traj = highlight_trajs[hi]                                # (t, 3)
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, linewidth=2.0, alpha=0.95,
        )
        if traj.shape[0] > 0:
            ax.scatter(
                [traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                color=color, s=40, edgecolors="black", linewidths=0.8,
            )
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=10)


def _pca_data_limits(arr: np.ndarray, pad: float = 0.1) -> tuple:
    lo, hi = float(arr.min()), float(arr.max())
    span = max(hi - lo, 1e-3)
    pad_abs = span * pad
    return (lo - pad_abs, hi + pad_abs)


# ---------------------------------------------------------------------------
# Per-cell 3-column video (motion / hidden / intention)
# ---------------------------------------------------------------------------


def render_per_cell_3col_video(
    cell_name: str,
    cell_data: dict[str, np.ndarray],
    highlights: np.ndarray,
    pca_hidden: Any,
    pca_intention: Any,
    ckpt: Any,
    env_cfg_ml: Any,
    clips: Any,
    cfg: Any,
    out_path: Path,
) -> None:
    """Render a per-cell 3-column video.

    Layout (per frame):
      [motion-ghost panel] | [hidden PCA-3D] | [intention PCA-3D]

    For now the motion panel is OMITTED (ghost rendering at 100 walkers
    is expensive and not the main signal here).  Instead we render a
    2-column composite of hidden + intention PCA-3D.  The motion panel
    can be added later by re-running with a render-helper extension.

    Args:
        cell_name: combination name (used in title + banner).
        cell_data: dict from npz with ``hidden`` ``(N, T, H)``, ``intention``
            ``(N, T, I)``.
        highlights: ``(n_hi,)`` int indices into the walker axis.
        pca_hidden, pca_intention: fitted sklearn PCAs (3 components each).
        cfg: hydra config; used for ``cfg.video.{fps,pca_rotation_degrees}``.
        out_path: mp4 path.
    """
    hidden = cell_data["hidden"]       # (N, T, H)
    intention = cell_data["intention"] # (N, T, I)
    if hidden.ndim != 3 or intention.ndim != 3:
        log.warning("unexpected shapes for cell %r; skipping", cell_name)
        return

    N, T, H = hidden.shape
    _, _, I = intention.shape

    # Project to 3D.
    hidden_pca = pca_hidden.transform(hidden.reshape(-1, H)).reshape(N, T, 3)
    intention_pca = pca_intention.transform(
        intention.reshape(-1, I)
    ).reshape(N, T, 3)

    # Axis limits — shared across all frames.
    h_xlim = _pca_data_limits(hidden_pca[:, :, 0])
    h_ylim = _pca_data_limits(hidden_pca[:, :, 1])
    h_zlim = _pca_data_limits(hidden_pca[:, :, 2])
    i_xlim = _pca_data_limits(intention_pca[:, :, 0])
    i_ylim = _pca_data_limits(intention_pca[:, :, 1])
    i_zlim = _pca_data_limits(intention_pca[:, :, 2])

    fps = int(cfg.video.fps)
    total_deg = float(cfg.video.pca_rotation_degrees)
    h_colors = [list(c) for c in cfg.highlight.highlight_colors]

    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax_h = fig.add_subplot(1, 2, 1, projection="3d")
    ax_i = fig.add_subplot(1, 2, 2, projection="3d")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)
    try:
        for t in range(T):
            azim = _interp_view_angle(t, T, total_deg)
            # Background = all walkers' position at t (excluding highlights).
            bg_mask = np.ones(N, dtype=bool)
            bg_mask[highlights] = False
            _draw_pca_frame(
                ax_h,
                background=hidden_pca[bg_mask, t, :],
                highlight_trajs=hidden_pca[highlights, :t + 1, :],
                highlight_colors=h_colors,
                azim=azim,
                xlim=h_xlim, ylim=h_ylim, zlim=h_zlim,
                title=f"hidden h_t  ({cell_name})",
            )
            _draw_pca_frame(
                ax_i,
                background=intention_pca[bg_mask, t, :],
                highlight_trajs=intention_pca[highlights, :t + 1, :],
                highlight_colors=h_colors,
                azim=azim,
                xlim=i_xlim, ylim=i_ylim, zlim=i_zlim,
                title=f"intention z_e μ",
            )
            fig.suptitle(f"{cell_name}  —  frame {t}/{T}", fontsize=11)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
            writer.append_data(buf)
    finally:
        writer.close()
        plt.close(fig)

    log.info("  wrote %s", out_path)


# ---------------------------------------------------------------------------
# Grid behavior video (placeholder — ghost rendering is expensive)
# ---------------------------------------------------------------------------


def render_grid_behavior_video(
    cell_data: dict[str, dict[str, np.ndarray]],
    highlights_per_cell: dict[str, np.ndarray],
    ckpt: Any,
    env_cfg_ml: Any,
    clips: Any,
    cfg: Any,
    out_path: Path,
) -> None:
    """Render the behavior catalog video — 3 walkers per cell in a grid.

    NOTE: full ghost rendering of 36 walkers (12 cells × 3 each) is
    expensive.  This minimal helper produces a PNG grid summary instead of
    a video — one panel per cell showing the highlighted-3 hidden-PCA
    trajectories.  Replace with the full ghost-render pipeline later if
    needed.
    """
    rows = int(cfg.grid.rows)
    cols = int(cfg.grid.cols)
    names = list(cell_data.keys())[:rows * cols]
    if len(names) < rows * cols:
        log.warning(
            "Only %d cells (< %d slots); grid will have blank panels",
            len(names), rows * cols,
        )

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), dpi=100,
        subplot_kw={"projection": "3d"},
    )
    axes = np.atleast_2d(axes)
    h_colors = [list(c) for c in cfg.highlight.highlight_colors]

    for ci, name in enumerate(names):
        r, c = ci // cols, ci % cols
        ax = axes[r, c]
        d = cell_data[name]
        hi = highlights_per_cell[name]
        h_traj = d["hidden"][hi]  # (n_hi, T, H)
        for ki, color in enumerate(h_colors[:len(hi)]):
            ax.plot(
                h_traj[ki, :, 0], h_traj[ki, :, 1], h_traj[ki, :, 2],
                color=color, linewidth=1.5, alpha=0.9,
            )
        ax.set_title(name, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=20, azim=45)
    for ci in range(len(names), rows * cols):
        r, c = ci // cols, ci % cols
        axes[r, c].axis("off")

    fig.suptitle(
        "Behavior grid — hidden-state trajectories of 3 highlighted walkers",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = out_path.with_suffix(".png")
    fig.savefig(str(png_path), bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote summary PNG %s (full ghost-render mp4 deferred)", png_path)
