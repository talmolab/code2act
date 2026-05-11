"""Side-by-side ghost-render + rotating-PCA video compositor.

Extracted from ``c2a.figures.plot_hidden_trajectory.render_trajectory_video``
so the controlled-rollout experiments can reuse the same composite layout
without re-implementing it.

The output frame layout::

    +-------- phase banner (full width) -------+
    |                                          |
    +-------- ghost (K bodies) ----+ PCA ------+
    |                              |           |
    |  K mujoco bodies overlaid    |  rotating |
    |  on close_profile camera     |  3D scatter|
    |  with code-bar timeline at   |  with     |
    |  the bottom (existing helper)|  trajectory|
    +------------------------------+-----------+

The ghost panel is the standard ``mujoco.Renderer`` output of an
``MoSeqImitation`` env with ``K-1`` ghost bodies. The PCA panel is a
matplotlib 3-D scatter of all ``K`` trajectories in PCA-space, drawn up
to the current frame, with a camera azimuth that linearly interpolates
from ``azim_start`` to ``azim_end`` over the full video. The phase
banner at the top shows which code-segment we are currently in.

PCA is fit OUTSIDE this function (the caller supplies a fit
``sklearn.decomposition.PCA``) so the joint-basis-across-videos
requirement of the controlled experiment is satisfied — every video
renders the same PCA axes.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from c2a.experiments.shared.ghost_rendering import build_ghost_model
from c2a.rendering_utils import add_multi_line_overlay

log = logging.getLogger(__name__)


def _compute_pca_limits(emb_all: np.ndarray, pad: float = 0.05) -> tuple:
    """Symmetric per-axis limits for a 3-D PCA scatter with padding."""
    lims = []
    for d in range(3):
        lo, hi = float(emb_all[:, d].min()), float(emb_all[:, d].max())
        span = max(hi - lo, 1e-6)
        lims.append((lo - pad * span, hi + pad * span))
    return lims[0], lims[1], lims[2]


def _style_pca_axes(
    ax, var_explained: np.ndarray,
    xlim: tuple, ylim: tuple, zlim: tuple,
) -> None:
    """Apply a clean, low-clutter look to a 3-D PCA axes."""
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.0f}%)", fontsize=6, labelpad=-8)
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.0f}%)", fontsize=6, labelpad=-8)
    ax.set_zlabel(f"PC3 ({var_explained[2] * 100:.0f}%)", fontsize=6, labelpad=-8)
    ax.grid(True, alpha=0.2, linewidth=0.3)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.04)


def _draw_phase_banner(
    width: int, height: int,
    current_frame: int, total_frames: int,
    phase_boundaries: list[int],
    phase_names: list[str],
    phase_colors_rgb: list[tuple[int, int, int]],
) -> np.ndarray:
    """Render the top phase-progression banner.

    The banner is divided into ``len(phase_names)`` colored segments
    proportional to the phase durations. A white playhead marks the
    current frame, and the active phase name is centered in big text.
    """
    banner = np.ones((height, width, 3), dtype=np.uint8) * 245

    # Color each phase segment.
    for i in range(len(phase_names)):
        x0 = int(phase_boundaries[i] * width / total_frames)
        x1 = int(phase_boundaries[i + 1] * width / total_frames)
        banner[:, x0:x1] = phase_colors_rgb[i]

    # Thin separator lines between phases.
    for i in range(1, len(phase_names)):
        x = int(phase_boundaries[i] * width / total_frames)
        banner[:, max(0, x - 1):x + 1] = [255, 255, 255]

    # Playhead.
    px = int(current_frame * width / max(total_frames - 1, 1))
    px = min(max(px, 1), width - 2)
    banner[:, px - 1:px + 1] = [30, 30, 30]
    banner[:, px:px + 1] = [255, 255, 255]

    # Determine active phase.
    active = 0
    for i in range(len(phase_names)):
        if phase_boundaries[i] <= current_frame < phase_boundaries[i + 1]:
            active = i
            break

    label = f"Phase {active + 1}/{len(phase_names)}: {phase_names[active]}"
    banner = add_multi_line_overlay(
        banner, [label],
        start_position=(12, max(2, height // 2 - 9)),
        font_size=max(13, height // 2 + 2),
    )
    return banner


def _phase_colors_palette(n: int) -> list[tuple[int, int, int]]:
    """Distinct phase colors via matplotlib's tab10."""
    cmap = plt.get_cmap("tab10")
    return [tuple(int(255 * c) for c in cmap(i % 10)[:3]) for i in range(n)]


def fit_joint_pca(
    hidden_list: list[np.ndarray],
    n_components: int = 3,
) -> tuple[PCA, np.ndarray]:
    """Fit one PCA on the concatenation of all hidden trajectories.

    Args:
        hidden_list: list of ``(K_i, T_i, hidden_dim)`` arrays — one per
            rollout. Different K_i / T_i across rollouts are fine; we
            flatten to ``(total_samples, hidden_dim)`` before fit.
        n_components: PCA dimensionality (3 for 3-D visualisation).

    Returns:
        ``(pca, var_explained_ratio)`` — the fit object and the per-axis
        variance-explained fractions.
    """
    hd = hidden_list[0].shape[-1]
    flat = np.concatenate([h.reshape(-1, hd) for h in hidden_list], axis=0)
    pca = PCA(n_components=n_components).fit(flat)
    return pca, pca.explained_variance_ratio_


def render_rotating_pca_video(
    hidden: np.ndarray,
    qpos: np.ndarray,
    phase_boundaries: list[int],
    phase_names: list[str],
    output_path: Path,
    *,
    pca: PCA,
    var_explained: np.ndarray,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
    ghost_env,
    K_visible: int | None = None,
    fps: int = 50,
    panel_w: int = 640,
    panel_h: int = 360,
    banner_h: int = 44,
    azim_start: float = 140.0,
    azim_end: float = 260.0,
    ghost_camera_distance: float = 0.8,
    ghost_camera_elevation: float = -25.0,
    ghost_camera_azimuth: float = 135.0,
    ghost_camera_fovy: float = 50.0,
    trajectory_alpha: float = 0.35,
    title_left: str = "Hidden trajectory (joint PCA)",
    title_right: str = "Ghost rollout",
) -> Path:
    """Render the ghost + rotating-PCA composite video.

    Args:
        hidden: ``(K, T, hidden_dim)`` — per-body hidden trajectories.
        qpos: ``(K, T, nq)`` — per-body qpos trajectories.
        phase_boundaries: monotonically increasing ints, length
            ``len(phase_names) + 1``. Phase ``i`` spans frames
            ``[phase_boundaries[i], phase_boundaries[i + 1])``.
        phase_names: human-readable name per phase (e.g. ``"Walk"``).
        output_path: target ``.mp4`` path.
        pca: pre-fit PCA on hidden — caller supplies for joint basis.
        var_explained: ``pca.explained_variance_ratio_`` (axis labels).
        xlim / ylim / zlim: pre-computed PCA-space limits (so all videos
            share identical axes).
        ghost_env: ``MoSeqImitation`` env used to build the ghost
            mujoco model. We only call ``build_ghost_model(env, ...)``
            on it; the env's state is NOT advanced inside this function.
        K_visible: render only the first ``K_visible`` bodies (default
            ``K``). Useful if you want fewer ghost bodies than data
            trajectories.
        fps: video framerate.
        panel_w / panel_h: per-panel pixel size. Composite output size
            is ``(2 * panel_w, panel_h + banner_h)``.
        banner_h: phase-banner height in pixels.
        azim_start / azim_end: 3-D camera azimuth at frame 0 and final
            frame. Linear interpolation in between.

    Returns:
        ``output_path`` on success.
    """
    K, T, _ = hidden.shape
    K_show = K if K_visible is None else min(K_visible, K)

    if len(phase_boundaries) != len(phase_names) + 1:
        raise ValueError(
            f"phase_boundaries (len {len(phase_boundaries)}) must be "
            f"len(phase_names) + 1 (= {len(phase_names) + 1})."
        )
    if phase_boundaries[0] != 0 or phase_boundaries[-1] != T:
        raise ValueError(
            f"phase_boundaries must start at 0 and end at T={T}; got "
            f"{phase_boundaries}."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Rendering composite video ({K_show}/{K} bodies, T={T}, "
        f"fps={fps}) → {output_path}"
    )

    # PCA embeddings: (K, T, 3).
    hidden_dim = hidden.shape[-1]
    emb = pca.transform(hidden.reshape(-1, hidden_dim)).reshape(K, T, 3)

    # Phase colors for the banner + per-frame trajectory segment coloring.
    phase_colors_rgb = _phase_colors_palette(len(phase_names))
    phase_colors_norm = [tuple(c / 255.0 for c in col) for col in phase_colors_rgb]
    # Per-frame phase index.
    phase_of_frame = np.zeros(T, dtype=np.int64)
    for i in range(len(phase_names)):
        phase_of_frame[phase_boundaries[i]:phase_boundaries[i + 1]] = i

    # Body colors for the K trajectories in PCA (tab10).
    cmap_bodies = plt.get_cmap("tab10")
    body_colors = [cmap_bodies(i % 10) for i in range(K)]

    # ── Ghost model ────────────────────────────────────────────────────
    ghost_model, base_nq = build_ghost_model(
        ghost_env,
        num_ghosts=K_show - 1,
        ghost_colors=[list(body_colors[i]) for i in range(1, K_show)],
        camera_distance=ghost_camera_distance,
        camera_elevation=ghost_camera_elevation,
        camera_azimuth=ghost_camera_azimuth,
        camera_fovy=ghost_camera_fovy,
    )
    ghost_model.vis.global_.offwidth = panel_w
    ghost_model.vis.global_.offheight = panel_h
    mj_data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=panel_h, width=panel_w)

    # Center qpos around per-frame mean-xy so the K bodies stay in-frame.
    min_len = min(qpos.shape[1], T)
    stacked_xy = qpos[:K_show, :min_len, :2]
    mean_xy = stacked_xy.mean(axis=0)
    centered_qpos = qpos[:K_show, :min_len].copy()
    centered_qpos[..., 0] -= mean_xy[:, 0][None, :]
    centered_qpos[..., 1] -= mean_xy[:, 1][None, :]

    # ── Matplotlib figure for PCA ──────────────────────────────────────
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

    # ── Video writer (mp4v intermediate then ffmpeg re-encode to H.264) ─
    total_w = panel_w * 2
    total_h = panel_h + banner_h
    total_w = ((total_w + 15) // 16) * 16
    total_h = ((total_h + 15) // 16) * 16

    tmp_path = output_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (total_w, total_h))

    log_every = max(1, min_len // 10)

    for t in range(min_len):
        combined = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

        # ── Banner ────────────────────────────────────────────────────
        banner = _draw_phase_banner(
            total_w, banner_h, t, min_len,
            phase_boundaries, phase_names, phase_colors_rgb,
        )
        combined[:banner_h, :] = banner

        # ── PCA panel ────────────────────────────────────────────────
        ax.cla()
        azim = azim_start + (azim_end - azim_start) * t / max(min_len - 1, 1)

        for ki in range(K):
            traj = emb[ki, :t + 1]
            if t > 0:
                # Color trajectory tail by phase, segment-wise.
                # Implementation: plot a single line then color via
                # collection? Simpler: one Line per phase segment.
                for ph_i in range(len(phase_names)):
                    seg_lo = max(phase_boundaries[ph_i], 0)
                    seg_hi = min(phase_boundaries[ph_i + 1], t + 1)
                    if seg_hi - seg_lo < 2:
                        continue
                    seg = emb[ki, seg_lo:seg_hi]
                    ax.plot(
                        seg[:, 0], seg[:, 1], seg[:, 2],
                        color=phase_colors_norm[ph_i],
                        linewidth=0.6, alpha=trajectory_alpha,
                    )
            # Current-position marker per body.
            ax.scatter(
                [traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                color=body_colors[ki], s=18,
                edgecolors="black", linewidths=0.4,
                zorder=10, depthshade=False,
            )

        _style_pca_axes(ax, var_explained, xlim, ylim, zlim)
        ax.view_init(elev=22, azim=azim)
        ax.set_title(title_left, fontsize=8, fontweight="bold", pad=-2)

        # Phase legend (small, top-right of PCA panel).
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=phase_colors_norm[i], markersize=4,
                   markeredgewidth=0, label=phase_names[i], linestyle="None")
            for i in range(len(phase_names))
        ]
        ax.legend(
            handles=legend_handles, loc="upper left",
            frameon=True, framealpha=0.92, edgecolor="none",
            fontsize=5, handletextpad=0.2, borderpad=0.3,
        )

        fig.canvas.draw()
        pca_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        pca_img = cv2.resize(pca_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        combined[banner_h:banner_h + panel_h, :panel_w] = pca_img

        # ── Ghost panel ───────────────────────────────────────────────
        t_clip = min(t, min_len - 1)
        mj_data.qpos[:base_nq] = centered_qpos[0, t_clip]
        for gi in range(1, K_show):
            qs = base_nq + (gi - 1) * base_nq
            qe = qs + base_nq
            mj_data.qpos[qs:qe] = centered_qpos[gi, t_clip]
        mujoco.mj_forward(ghost_model, mj_data)
        renderer.update_scene(mj_data, camera="close_profile")
        frame_ghost = renderer.render().copy()
        if frame_ghost.shape[:2] != (panel_h, panel_w):
            frame_ghost = cv2.resize(
                frame_ghost, (panel_w, panel_h), interpolation=cv2.INTER_AREA,
            )
        combined[banner_h:banner_h + panel_h, panel_w:panel_w * 2] = frame_ghost

        # ── Compact text labels on each panel ────────────────────────
        combined = add_multi_line_overlay(
            combined, [f"t={t}/{min_len}"],
            start_position=(panel_w + 8, banner_h + 4),
            font_size=11,
        )

        writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if (t + 1) % log_every == 0:
            log.info(f"  frame {t + 1}/{min_len}")

    writer.release()
    renderer.close()
    plt.close(fig)

    # Re-encode to H.264 (mp4v plays in fewer viewers).
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(tmp_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-movflags", "+faststart", str(output_path),
        ],
        check=True, capture_output=True,
    )
    tmp_path.unlink()

    log.info(f"Saved video: {output_path}")
    return output_path
