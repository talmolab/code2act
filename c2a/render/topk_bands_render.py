"""Shared figure/video plumbing for top-K band visualizations.

:mod:`c2a.codegen.kpms.render_topk_bands` and the kinematics-VAE family's
renderers produce the same kind of artifact:

  * Per-clip figure: a 3D-skeleton filmstrip (8 snapshots) on top, with a
    ground-truth behavior band + K model band rows underneath.
  * Per-clip video: an animated 3D skeleton + the same band rows with a
    moving red cursor.
  * Combined 3-column variant of each (one column per selected clip).

The only thing that differs between callers is **where the per-clip
integer label arrays come from** (KPMS reads ``results.h5`` inside each
fit's project dir; the kinematics-VAE family reads ``best_codes.npz``).
This module owns everything else.

All helpers operate on the same ``model_rows`` shape:

    list[tuple[label: str, row: np.ndarray (T,), n_classes: int]]

so a "model" can be any per-frame integer code sequence.

Behaviour-band coloring uses ``c2a.codegen.shared.label_alignment.{BEHAVIORS, UNLABELED}``
because all pipelines compare against the same manual rodent labels.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from c2a.codegen.shared.label_alignment import BEHAVIORS, UNLABELED

log = logging.getLogger(__name__)


# 4 perceptually distinct colors for the rodent label vocabulary
# (Immobile, Rear, Turn, Walk — see c2a.codegen.shared.label_alignment.BEHAVIORS).
# Held stable across figures so plots are directly comparable across tokenizers.
BEHAVIOR_COLORS = {
    "Immobile": "#377eb8",
    "Rear":     "#984ea3",
    "Turn":     "#ff7f00",
    "Walk":     "#a65628",
}
UNLABELED_COLOR = "#dddddd"


def _resolve_skeleton_edges(
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
) -> list[tuple[int, int]]:
    """Translate skeleton name pairs to integer index pairs into ``kp_names``.

    Edges referencing absent keypoints are silently dropped.
    """
    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    return [
        (name_to_idx[a], name_to_idx[b])
        for a, b in skeleton
        if a in name_to_idx and b in name_to_idx
    ]


def _draw_gt_row(ax, row: np.ndarray, n_frames: int, show_xticks: bool = True) -> None:
    """Draw ground-truth behavior bands with named labels.

    ``show_xticks=False`` hides the x-tick labels — for stacked rows
    where only the bottom row should show frame numbers.
    """
    if row.size == 0:
        ax.set_xlim(0, n_frames)
        ax.set_yticks([])
        return
    changes = np.where(np.diff(row) != 0)[0] + 1
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [row.size]])

    for s, e in zip(segment_starts, segment_ends):
        val = int(row[s])
        if val == UNLABELED:
            color = UNLABELED_COLOR
            label = ""
        else:
            label = BEHAVIORS[val] if val < len(BEHAVIORS) else ""
            color = BEHAVIOR_COLORS.get(label, "#999999")
        ax.axvspan(s, e, color=color, alpha=0.85, linewidth=0)
        if label and (e - s) >= 12:
            ax.text(
                (s + e) / 2, 0.5, label,
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlim(0, n_frames)
    ax.set_yticks([])
    ax.set_ylabel("GT", fontsize=9, labelpad=8, rotation=0, ha="right", va="center")
    if not show_xticks:
        ax.tick_params(axis="x", labelbottom=False, length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_model_row(
    ax,
    row: np.ndarray,
    n_frames: int,
    n_classes: int,
    cmap,
    title: str,
    show_xticks: bool = True,
) -> None:
    """Draw discrete-class bands colored by class id.

    Works for any per-frame integer code sequence (e.g. KPMS syllable IDs).

    ``show_xticks=False`` hides the x-tick labels — useful for stacked
    band rows where only the bottom row should show frame numbers.
    """
    if row.size == 0:
        ax.set_xlim(0, n_frames)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=7, rotation=0, labelpad=8, ha="right", va="center")
        return
    changes = np.where(np.diff(row) != 0)[0] + 1
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [row.size]])

    denom = max(n_classes - 1, 1)
    for s, e in zip(segment_starts, segment_ends):
        cls = int(row[s])
        ax.axvspan(s, e, color=cmap(cls / denom), alpha=0.9, linewidth=0)

    ax.set_xlim(0, n_frames)
    ax.set_yticks([])
    ax.set_ylabel(title, fontsize=7, rotation=0, labelpad=8, ha="right", va="center")
    if not show_xticks:
        ax.tick_params(axis="x", labelbottom=False, length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_filmstrip_row(
    fig,
    outer_gridspec,
    keypoint_traj: np.ndarray,
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    n_frames: int,
    n_snapshots: int = 8,
) -> None:
    """Draw a row of 3D skeleton snapshots at evenly-spaced frames.

    Each snapshot's view is centered on its own COM at a shared zoom
    (median body extent across snapshots × 1.35) so motion reads as
    translation across the strip.
    """
    T = min(keypoint_traj.shape[0], n_frames)
    frame_idx = np.linspace(0, T - 1, n_snapshots).astype(int)

    pts_all = keypoint_traj[frame_idx]
    per_extent = np.ptp(pts_all.reshape(n_snapshots, -1, 3), axis=1)
    zoom = float(np.median(per_extent.max(axis=1)))
    zoom = max(zoom, 1e-3) * 1.35

    edges = _resolve_skeleton_edges(kp_names, skeleton)

    strip_gs = GridSpecFromSubplotSpec(
        1, n_snapshots, subplot_spec=outer_gridspec, wspace=0.02,
    )
    cmap = plt.get_cmap("viridis")

    for j, f in enumerate(frame_idx):
        ax = fig.add_subplot(strip_gs[0, j], projection="3d")
        pts = keypoint_traj[f]
        color = cmap(j / max(n_snapshots - 1, 1))
        com = pts.mean(axis=0)

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=10)
        for ia, ib in edges:
            ax.plot(
                [pts[ia, 0], pts[ib, 0]],
                [pts[ia, 1], pts[ib, 1]],
                [pts[ia, 2], pts[ib, 2]],
                color=color, linewidth=1.8, alpha=0.95,
            )

        half = zoom / 2
        ax.set_xlim(com[0] - half, com[0] + half)
        ax.set_ylim(com[1] - half, com[1] + half)
        ax.set_zlim(com[2] - half, com[2] + half)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(
            f"t={f}\n(x={com[0]:.2f}, y={com[1]:.2f})", fontsize=7,
        )
        ax.view_init(elev=15, azim=-75)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("w")


def render_clip_figure(
    clip_idx: int,
    gt_row: np.ndarray,
    model_rows: list[tuple[str, np.ndarray, int]],
    keypoint_traj: np.ndarray,
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    output_path: Path,
    n_frames: int,
    n_snapshots: int = 8,
    title_prefix: str = "Top-K models",
) -> None:
    """Render one per-clip PNG: filmstrip + 1 GT + K model rows."""
    k = len(model_rows)

    # Per-band height generous so labels never crowd; hspace breathing
    # room between rows.
    fig = plt.figure(figsize=(17, 3.6 + 1.0 * (k + 1)))
    outer = GridSpec(
        2, 1, height_ratios=[3.0, 1.0 * (k + 1)],
        hspace=0.32, figure=fig,
        left=0.20, right=0.99, top=0.95, bottom=0.06,
    )
    _draw_filmstrip_row(
        fig, outer[0], keypoint_traj, kp_names, skeleton,
        n_frames, n_snapshots,
    )

    band_gs = GridSpecFromSubplotSpec(
        k + 1, 1, subplot_spec=outer[1], hspace=0.95,
    )
    ax_gt = fig.add_subplot(band_gs[0])
    _draw_gt_row(ax_gt, gt_row, n_frames, show_xticks=False)
    ax_gt.set_title(
        f"Clip {clip_idx} — Ground-truth behavior vs {title_prefix}",
        fontsize=11,
    )

    cmap = plt.get_cmap("tab20")
    ax_last = None
    n_total = len(model_rows)
    for i, (label, row, n_cls) in enumerate(model_rows, start=1):
        ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
        _draw_model_row(
            ax_m, row[:n_frames], n_frames, n_cls, cmap, title=label,
            show_xticks=(i == n_total),
        )
        ax_last = ax_m

    if ax_last is not None:
        ax_last.set_xlabel("Frame")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {output_path}")


def render_combined_figure(
    clip_indices: list[int],
    gt_rows: list[np.ndarray],
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]],
    keypoint_trajs: list[np.ndarray],
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    output_path: Path,
    n_frames: int,
    n_snapshots: int = 8,
) -> None:
    """Render 3-column side-by-side combined PNG (filmstrip + bands)."""
    if len(clip_indices) != 3:
        raise ValueError(f"Expected 3 clips for combined figure, got {len(clip_indices)}.")
    k = len(model_rows_per_clip[0])

    fig = plt.figure(figsize=(3 * 14, 3.6 + 1.0 * (k + 1)))
    outer = GridSpec(
        2, 3, height_ratios=[3.0, 1.0 * (k + 1)],
        hspace=0.32, wspace=0.18, figure=fig,
        left=0.06, right=0.99, top=0.95, bottom=0.06,
    )

    cmap = plt.get_cmap("tab20")
    n_total = k
    for col in range(3):
        _draw_filmstrip_row(
            fig, outer[0, col], keypoint_trajs[col], kp_names, skeleton,
            n_frames, n_snapshots,
        )

        band_gs = GridSpecFromSubplotSpec(
            k + 1, 1, subplot_spec=outer[1, col], hspace=0.95,
        )
        ax_gt = fig.add_subplot(band_gs[0])
        _draw_gt_row(ax_gt, gt_rows[col], n_frames, show_xticks=False)
        ax_gt.set_title(f"Clip {clip_indices[col]}", fontsize=11)

        ax_last = None
        for i, (label, row, n_cls) in enumerate(model_rows_per_clip[col], start=1):
            ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
            _draw_model_row(
                ax_m, row[:n_frames], n_frames, n_cls, cmap, title=label,
                show_xticks=(i == n_total),
            )
            ax_last = ax_m
        if ax_last is not None:
            ax_last.set_xlabel("Frame")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {output_path}")


def render_clip_video(
    clip_idx: int,
    gt_row: np.ndarray,
    model_rows: list[tuple[str, np.ndarray, int]],
    keypoint_traj: np.ndarray,
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    output_path: Path,
    n_frames: int,
    fps: int = 30,
) -> None:
    """Render mp4: animated 3D skeleton (top) + static band rows + moving cursor."""
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    k = len(model_rows)
    T = int(min(keypoint_traj.shape[0], n_frames))
    edges = _resolve_skeleton_edges(kp_names, skeleton)

    fig = plt.figure(figsize=(16, 7.0 + 0.95 * (k + 1)))
    outer = GridSpec(
        2, 1, height_ratios=[6.0, 0.95 * (k + 1)],
        hspace=0.28, figure=fig,
        left=0.21, right=0.99, top=0.96, bottom=0.06,
    )
    ax3d = fig.add_subplot(outer[0], projection="3d")
    band_gs = GridSpecFromSubplotSpec(
        k + 1, 1, subplot_spec=outer[1], hspace=1.0,
    )

    coms = keypoint_traj[:T].mean(axis=1)
    per_frame_ext = np.ptp(keypoint_traj[:T].reshape(T, -1, 3), axis=1)
    zoom = float(np.median(per_frame_ext.max(axis=1))) * 2.2
    zoom = max(zoom, 1e-2)

    pts0 = keypoint_traj[0]
    (pts_artist,) = ax3d.plot(
        pts0[:, 0], pts0[:, 1], pts0[:, 2],
        "o", color="steelblue", ms=5,
    )
    line_artists = [
        ax3d.plot(
            [pts0[ia, 0], pts0[ib, 0]],
            [pts0[ia, 1], pts0[ib, 1]],
            [pts0[ia, 2], pts0[ib, 2]],
            "-", color="gray", linewidth=2,
        )[0]
        for ia, ib in edges
    ]
    frame_text = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes, fontsize=10)
    ax3d.set_title(f"Clip {clip_idx}", fontsize=12)
    ax3d.view_init(elev=18, azim=-65)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False

    cmap = plt.get_cmap("tab20")
    n_total = len(model_rows)
    ax_gt = fig.add_subplot(band_gs[0])
    _draw_gt_row(ax_gt, gt_row, n_frames, show_xticks=False)
    cursors = [ax_gt.axvline(0, color="red", linewidth=1.6, zorder=10)]

    ax_last = ax_gt
    for i, (label, row, n_cls) in enumerate(model_rows, start=1):
        ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
        _draw_model_row(
            ax_m, row[:n_frames], n_frames, n_cls, cmap, title=label,
            show_xticks=(i == n_total),
        )
        cursors.append(ax_m.axvline(0, color="red", linewidth=1.6, zorder=10))
        ax_last = ax_m
    ax_last.set_xlabel("Frame")

    def _update_skeleton_bounds(t: int) -> None:
        c = coms[t]
        half = zoom / 2
        ax3d.set_xlim(c[0] - half, c[0] + half)
        ax3d.set_ylim(c[1] - half, c[1] + half)
        ax3d.set_zlim(c[2] - half, c[2] + half)

    _update_skeleton_bounds(0)

    def update(t: int):
        pts = keypoint_traj[t]
        pts_artist.set_data(pts[:, 0], pts[:, 1])
        pts_artist.set_3d_properties(pts[:, 2])
        for ln, (ia, ib) in zip(line_artists, edges):
            ln.set_data([pts[ia, 0], pts[ib, 0]], [pts[ia, 1], pts[ib, 1]])
            ln.set_3d_properties([pts[ia, 2], pts[ib, 2]])
        _update_skeleton_bounds(t)
        frame_text.set_text(f"Frame {t}/{T - 1}")
        for cur in cursors:
            cur.set_xdata([t, t])
        return [pts_artist, *line_artists, frame_text, *cursors]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=3200)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    log.info(f"Saved video {output_path}")


def render_combined_video(
    clip_indices: list[int],
    gt_rows: list[np.ndarray],
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]],
    keypoint_trajs: list[np.ndarray],
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    output_path: Path,
    n_frames: int,
    fps: int = 30,
) -> None:
    """Render a 3-panel mp4: each column is one clip (skeleton + bands)."""
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    if len(clip_indices) != 3:
        raise ValueError(f"Expected 3 clips, got {len(clip_indices)}.")

    k = len(model_rows_per_clip[0])
    T = int(min(min(kp.shape[0] for kp in keypoint_trajs), n_frames))
    edges = _resolve_skeleton_edges(kp_names, skeleton)

    fig = plt.figure(figsize=(3 * 12, 7.0 + 0.95 * (k + 1)))
    outer = GridSpec(
        2, 3, height_ratios=[6.0, 0.95 * (k + 1)],
        hspace=0.28, wspace=0.18, figure=fig,
        left=0.07, right=0.99, top=0.96, bottom=0.06,
    )

    cmap = plt.get_cmap("tab20")
    per_col = []

    for col in range(3):
        kp_traj = keypoint_trajs[col][:T]
        coms = kp_traj.mean(axis=1)
        per_ext = np.ptp(kp_traj.reshape(T, -1, 3), axis=1)
        zoom = max(float(np.median(per_ext.max(axis=1))) * 2.2, 1e-2)

        ax3d = fig.add_subplot(outer[0, col], projection="3d")
        pts0 = kp_traj[0]
        (pts_artist,) = ax3d.plot(
            pts0[:, 0], pts0[:, 1], pts0[:, 2],
            "o", color="steelblue", ms=5,
        )
        line_artists = [
            ax3d.plot(
                [pts0[ia, 0], pts0[ib, 0]],
                [pts0[ia, 1], pts0[ib, 1]],
                [pts0[ia, 2], pts0[ib, 2]],
                "-", color="gray", linewidth=2,
            )[0]
            for ia, ib in edges
        ]
        frame_text = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes, fontsize=10)
        ax3d.set_title(f"Clip {clip_indices[col]}", fontsize=12)
        ax3d.view_init(elev=18, azim=-65)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False

        band_gs = GridSpecFromSubplotSpec(
            k + 1, 1, subplot_spec=outer[1, col], hspace=1.0,
        )
        ax_gt = fig.add_subplot(band_gs[0])
        _draw_gt_row(ax_gt, gt_rows[col], n_frames, show_xticks=False)
        cursors = [ax_gt.axvline(0, color="red", linewidth=1.6, zorder=10)]

        ax_last = ax_gt
        n_total_col = len(model_rows_per_clip[col])
        for i, (label, row, n_cls) in enumerate(model_rows_per_clip[col], start=1):
            ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
            _draw_model_row(
                ax_m, row[:n_frames], n_frames, n_cls, cmap, title=label,
                show_xticks=(i == n_total_col),
            )
            cursors.append(ax_m.axvline(0, color="red", linewidth=1.6, zorder=10))
            ax_last = ax_m
        ax_last.set_xlabel("Frame")

        per_col.append((ax3d, pts_artist, line_artists, frame_text, coms, zoom, cursors))

    for ax3d, _, _, _, coms, zoom, _ in per_col:
        c = coms[0]
        h = zoom / 2
        ax3d.set_xlim(c[0] - h, c[0] + h)
        ax3d.set_ylim(c[1] - h, c[1] + h)
        ax3d.set_zlim(c[2] - h, c[2] + h)

    def update(t: int):
        artists = []
        for col, (ax3d, pts_artist, line_artists, frame_text, coms, zoom, cursors) in enumerate(per_col):
            pts = keypoint_trajs[col][t]
            pts_artist.set_data(pts[:, 0], pts[:, 1])
            pts_artist.set_3d_properties(pts[:, 2])
            for ln, (ia, ib) in zip(line_artists, edges):
                ln.set_data([pts[ia, 0], pts[ib, 0]], [pts[ia, 1], pts[ib, 1]])
                ln.set_3d_properties([pts[ia, 2], pts[ib, 2]])
            c = coms[t]
            h = zoom / 2
            ax3d.set_xlim(c[0] - h, c[0] + h)
            ax3d.set_ylim(c[1] - h, c[1] + h)
            ax3d.set_zlim(c[2] - h, c[2] + h)
            frame_text.set_text(f"Frame {t}/{T - 1}")
            for cur in cursors:
                cur.set_xdata([t, t])
            artists.extend([pts_artist, *line_artists, frame_text, *cursors])
        return artists

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=4800)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    log.info(f"Saved video {output_path}")


def auto_select_clips(
    frame_labels: np.ndarray,
    labeled_clip_indices: np.ndarray,
    intersect_mask: np.ndarray,
    n_clips: int = 3,
) -> list[int]:
    """Pick ``n_clips`` clips with the most varied behavior content.

    Generic clip-picker shared by all top-K band renderers.  Primary
    key: number of distinct (non-``UNLABELED``) behaviors in the clip.
    Tie-breakers: transition count, then labeled-frame count.
    """
    candidate_pos = np.where(intersect_mask)[0]
    if candidate_pos.size == 0:
        raise ValueError("No labeled clip intersects the fit set — cannot render bands.")

    scored: list[tuple[int, int, int, int]] = []
    for fi in candidate_pos:
        row = frame_labels[fi]
        labeled = row[row != UNLABELED]
        if labeled.size == 0:
            continue
        n_distinct = int(np.unique(labeled).size)
        if n_distinct < 2:
            continue
        n_transitions = int((np.diff(row) != 0).sum())
        scored.append((
            -n_distinct, -n_transitions, -int(labeled.size),
            int(labeled_clip_indices[fi]),
        ))

    if not scored:
        log.warning(
            "No multi-behavior labeled clip found in the fit set — falling "
            "back to clips with the most labeled frames."
        )
        fallback = sorted(
            [
                (-int((frame_labels[fi] != UNLABELED).sum()),
                 int(labeled_clip_indices[fi]))
                for fi in candidate_pos
            ]
        )
        return [ci for _, ci in fallback[:n_clips]]

    scored.sort()
    chosen = [s[-1] for s in scored[:n_clips]]
    for s in scored[:n_clips]:
        log.info(
            f"Auto-selected clip {s[-1]}: "
            f"{-s[0]} distinct behaviors, {-s[1]} transitions, "
            f"{-s[2]} labeled frames."
        )
    return chosen
