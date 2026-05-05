"""Post-hoc top-K syllable-band figure for a KPMS sweep.

Reads ``sweep_results.json``, loads each top-K model's ``results.h5``, and
renders a stacked band figure for each of three selected labeled clips:

* Row 0 — ground-truth human behavior bands, labeled with behavior names.
* Rows 1..K — each top-K model's syllable bands, colored by syllable ID.

Produces one tall figure per clip plus a combined 3-column side-by-side.

Usage::

    python -m c2a.sweep.render_topk_bands --sweep-dir outputs/kpms_sweep
    python -m c2a.sweep.render_topk_bands --sweep-dir outputs/kpms_sweep \
        --rank nmi --top-k 5
    python -m c2a.sweep.render_topk_bands --sweep-dir outputs/kpms_sweep \
        --clips 700 720 735

Defaults: ``--top-k 5``, ``--rank mse``. Clip selection heuristic picks one
clip each dominated by Walking / Grooming / any Rear among the labeled
clips that intersect the fit set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from c2a.kpms.label_alignment import (
    BEHAVIORS,
    UNLABELED,
    load_behavior_frames,
)
from c2a.paths import get_repo_root
from c2a.sweep.run_sweep import RODENT_SKELETON, _load_keypoints

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 6 perceptually distinct colors for the 6 behavior classes (Grooming,
# Immobility, Orienting, Supported Rear, Unsupported Rear, Walking) plus a
# neutral gray for UNLABELED. Keep stable across figures.
BEHAVIOR_COLORS = {
    "Grooming": "#e41a1c",
    "Immobility": "#377eb8",
    "Orienting": "#4daf4a",
    "Supported Rear": "#984ea3",
    "Unsupported Rear": "#ff7f00",
    "Walking": "#a65628",
}
UNLABELED_COLOR = "#dddddd"


def _load_balanced_split(balanced_split_path: Path) -> np.ndarray:
    """Return the sorted union of train+test balanced clip indices."""
    with balanced_split_path.open() as f:
        splits = json.load(f)
    all_idx = sorted(
        set(splits["balanced"]["train_indices"])
        | set(splits["balanced"]["test_indices"])
    )
    return np.array(all_idx, dtype=np.int32)


def _load_syllables_for_fit(project_dir: Path, model_name: str) -> list[np.ndarray]:
    """Load per-clip syllable arrays from a KPMS fit's ``results.h5``.

    Returns the syllable array for each recording, sorted by recording name
    (zero-padded so alphabetical == numerical order — matches the sort
    semantics in ``c2a/kpms/fit_kpms.py:162``).
    """
    results_h5 = project_dir / model_name / "results.h5"
    if not results_h5.exists():
        raise FileNotFoundError(f"Results HDF5 not found: {results_h5}")
    with h5py.File(results_h5, "r") as f:
        rec_names = sorted(f.keys())
        return [np.asarray(f[rn]["syllable"][:], dtype=np.int32) for rn in rec_names]


def _select_topk(
    all_results: list[dict],
    rank: str,
    top_k: int,
) -> list[dict]:
    """Pick top-K valid results by the requested ranking."""
    valid = [r for r in all_results if "error" not in r]
    if rank == "mse":
        valid.sort(key=lambda r: r["reconstruction_mse"])
    elif rank == "nmi":
        valid.sort(key=lambda r: -r.get("nmi_global", float("-inf")))
    else:
        raise ValueError(f"Unknown rank={rank!r}; choose 'mse' or 'nmi'.")
    return valid[:top_k]


def _auto_select_clips(
    frame_labels: np.ndarray,
    labeled_clip_indices: np.ndarray,
    intersect_mask: np.ndarray,
    n_clips: int = 3,
) -> list[int]:
    """Pick ``n_clips`` clips with the most varied behavior content.

    Primary key: number of distinct (non-``UNLABELED``) behaviors in the
    clip — we want transitions to compare against the model bands. Clips
    with a single behavior are rejected outright. Tie-breakers: transition
    count (more = more movement to look at), labeled-frame count (more =
    more ground truth to compare).

    Only considers clips that are both labeled and in the fit set.
    """
    candidate_pos = np.where(intersect_mask)[0]
    if candidate_pos.size == 0:
        raise ValueError("No labeled clip intersects the fit set — cannot render bands.")

    scored: list[tuple[int, int, int, int]] = []  # (-n_distinct, -n_transitions, -n_labeled, clip_idx)
    for fi in candidate_pos:
        row = frame_labels[fi]
        labeled = row[row != UNLABELED]
        if labeled.size == 0:
            continue
        n_distinct = int(np.unique(labeled).size)
        if n_distinct < 2:
            continue  # single-behavior clips are not visually informative
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
        # Fallback: just take the clips with the most labeled frames.
        fallback = sorted(
            [
                (-int((frame_labels[fi] != UNLABELED).sum()),
                 int(labeled_clip_indices[fi]))
                for fi in candidate_pos
            ]
        )
        return [ci for _, ci in fallback[:n_clips]]

    scored.sort()  # ascending on negatives → descending on (distinct, trans, labeled)
    chosen = [s[-1] for s in scored[:n_clips]]
    for s in scored[:n_clips]:
        log.info(
            f"Auto-selected clip {s[-1]}: "
            f"{-s[0]} distinct behaviors, {-s[1]} transitions, "
            f"{-s[2]} labeled frames."
        )
    return chosen


def _resolve_skeleton_edges(kp_names: list[str]) -> list[tuple[int, int]]:
    """Translate ``RODENT_SKELETON`` name pairs to integer index pairs."""
    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    return [
        (name_to_idx[a], name_to_idx[b])
        for a, b in RODENT_SKELETON
        if a in name_to_idx and b in name_to_idx
    ]


def _render_clip_video(
    clip_idx: int,
    gt_row: np.ndarray,
    model_rows: list[tuple[str, np.ndarray, int]],
    keypoint_traj: np.ndarray,
    kp_names: list[str],
    output_path: Path,
    n_frames: int,
    fps: int = 30,
) -> None:
    """Render mp4: animated 3D skeleton (top) + static band rows + moving cursor.

    Band rows are drawn once (static coloring); a red vertical cursor line
    moves across all rows in lockstep with the skeleton animation.
    """
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    k = len(model_rows)
    T = int(min(keypoint_traj.shape[0], n_frames))
    edges = _resolve_skeleton_edges(kp_names)

    # Figure: skeleton (square, big) on top, bands (stacked) below.
    fig = plt.figure(figsize=(14, 6.5 + 0.55 * (k + 1)))
    outer = GridSpec(
        2, 1, height_ratios=[6.0, 0.55 * (k + 1)],
        hspace=0.25, figure=fig,
    )
    ax3d = fig.add_subplot(outer[0], projection="3d")
    band_gs = GridSpecFromSubplotSpec(
        k + 1, 1, subplot_spec=outer[1], hspace=0.55,
    )

    # Follow-cam: center on per-frame COM with a fixed zoom = body-length * 2.
    coms = keypoint_traj[:T].mean(axis=1)  # [T, 3]
    per_frame_ext = keypoint_traj[:T].reshape(T, -1, 3).ptp(axis=1)
    zoom = float(np.median(per_frame_ext.max(axis=1))) * 2.2
    zoom = max(zoom, 1e-2)

    # Initial skeleton artists (frame 0).
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

    # Band rows (drawn once, static) + per-row cursor.
    cmap = plt.get_cmap("tab20")
    ax_gt = fig.add_subplot(band_gs[0])
    _draw_gt_row(ax_gt, gt_row, n_frames)
    cursors = [ax_gt.axvline(0, color="red", linewidth=1.6, zorder=10)]

    ax_last = ax_gt
    for i, (label, row, n_syll) in enumerate(model_rows, start=1):
        ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
        _draw_model_row(ax_m, row[:n_frames], n_frames, n_syll, cmap, title=label)
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


def _render_combined_video(
    clip_indices: list[int],
    gt_rows: list[np.ndarray],
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]],
    keypoint_trajs: list[np.ndarray],
    kp_names: list[str],
    output_path: Path,
    n_frames: int,
    fps: int = 30,
) -> None:
    """Render a 3-panel mp4: each column is one clip (skeleton on top, bands below)."""
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    if len(clip_indices) != 3:
        raise ValueError(f"Expected 3 clips, got {len(clip_indices)}.")

    k = len(model_rows_per_clip[0])
    T = int(min(min(kp.shape[0] for kp in keypoint_trajs), n_frames))
    edges = _resolve_skeleton_edges(kp_names)

    fig = plt.figure(figsize=(3 * 11, 6.5 + 0.55 * (k + 1)))
    outer = GridSpec(
        2, 3, height_ratios=[6.0, 0.55 * (k + 1)],
        hspace=0.25, wspace=0.1, figure=fig,
    )

    cmap = plt.get_cmap("tab20")
    per_col = []  # (ax3d, pts_artist, line_artists, frame_text, coms, zoom, cursors)

    for col in range(3):
        kp_traj = keypoint_trajs[col][:T]
        coms = kp_traj.mean(axis=1)
        per_ext = kp_traj.reshape(T, -1, 3).ptp(axis=1)
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
            k + 1, 1, subplot_spec=outer[1, col], hspace=0.55,
        )
        ax_gt = fig.add_subplot(band_gs[0])
        _draw_gt_row(ax_gt, gt_rows[col], n_frames)
        cursors = [ax_gt.axvline(0, color="red", linewidth=1.6, zorder=10)]

        ax_last = ax_gt
        for i, (label, row, n_syll) in enumerate(model_rows_per_clip[col], start=1):
            ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
            _draw_model_row(ax_m, row[:n_frames], n_frames, n_syll, cmap, title=label)
            cursors.append(ax_m.axvline(0, color="red", linewidth=1.6, zorder=10))
            ax_last = ax_m
        ax_last.set_xlabel("Frame")

        per_col.append((ax3d, pts_artist, line_artists, frame_text, coms, zoom, cursors))

    # Initial bounds.
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


def _draw_filmstrip_row(
    fig,
    outer_gridspec,
    keypoint_traj: np.ndarray,
    kp_names: list[str],
    n_frames: int,
    n_snapshots: int = 8,
) -> None:
    """Draw a row of 3D skeleton snapshots at evenly-spaced frames.

    Uses a nested GridSpec so the snapshots occupy a single horizontal
    strip above the band rows. Each snapshot shares the same axis bounds
    (computed across all snapshots) so motion is visible as translation
    across the strip.

    Args:
        fig: Figure to draw into.
        outer_gridspec: A single SubplotSpec allotted to the filmstrip row.
        keypoint_traj: ``[T, K, 3]`` keypoints in world coordinates.
        kp_names: Keypoint names (for resolving skeleton edges).
        n_frames: Total frames in the clip (for x-axis alignment).
        n_snapshots: How many snapshots to render across the strip.
    """
    T = min(keypoint_traj.shape[0], n_frames)
    frame_idx = np.linspace(0, T - 1, n_snapshots).astype(int)

    # Per-snapshot body-length so we can zoom each skeleton tight to its own
    # bounding box — the animal translates many body-lengths across a clip,
    # so a single global bound leaves every pose as a dot. Use a shared
    # per-snapshot zoom size (= median body extent across snapshots) so
    # scale is comparable frame-to-frame even though center moves.
    pts_all = keypoint_traj[frame_idx]  # [n_snap, K, 3]
    per_extent = pts_all.reshape(n_snapshots, -1, 3).ptp(axis=1)  # [n_snap, 3]
    zoom = float(np.median(per_extent.max(axis=1)))
    zoom = max(zoom, 1e-3) * 1.35  # 35% padding around each skeleton

    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    edges = [
        (name_to_idx[a], name_to_idx[b])
        for a, b in RODENT_SKELETON
        if a in name_to_idx and b in name_to_idx
    ]

    # Nested strip: 1 row × n_snapshots columns.
    strip_gs = GridSpecFromSubplotSpec(
        1, n_snapshots, subplot_spec=outer_gridspec, wspace=0.02,
    )
    # Color gradient so frame-order is obvious at a glance.
    cmap = plt.get_cmap("viridis")

    for j, f in enumerate(frame_idx):
        ax = fig.add_subplot(strip_gs[0, j], projection="3d")
        pts = keypoint_traj[f]
        color = cmap(j / max(n_snapshots - 1, 1))
        com = pts.mean(axis=0)  # [3]

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=10)
        for ia, ib in edges:
            ax.plot(
                [pts[ia, 0], pts[ib, 0]],
                [pts[ia, 1], pts[ib, 1]],
                [pts[ia, 2], pts[ib, 2]],
                color=color, linewidth=1.8, alpha=0.95,
            )

        # Center on this snapshot's COM with a uniform zoom.
        half = zoom / 2
        ax.set_xlim(com[0] - half, com[0] + half)
        ax.set_ylim(com[1] - half, com[1] + half)
        ax.set_zlim(com[2] - half, com[2] + half)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Annotate with frame number AND (x,y) COM position so you can still
        # read translation between snapshots.
        ax.set_title(
            f"t={f}\n(x={com[0]:.2f}, y={com[1]:.2f})",
            fontsize=7,
        )
        ax.view_init(elev=15, azim=-75)
        # Strip out the 3D panes for a cleaner look.
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("w")


def _draw_gt_row(ax, row: np.ndarray, n_frames: int) -> None:
    """Draw ground-truth behavior bands with named labels."""
    if row.size == 0:
        ax.set_xlim(0, n_frames)
        ax.set_yticks([])
        return
    # Segment the label row by runs of equal values.
    changes = np.where(np.diff(row) != 0)[0] + 1
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [row.size]])

    for s, e in zip(segment_starts, segment_ends):
        val = int(row[s])
        if val == UNLABELED:
            color = UNLABELED_COLOR
            label = ""
        else:
            label = BEHAVIORS[val]
            color = BEHAVIOR_COLORS[label]
        ax.axvspan(s, e, color=color, alpha=0.85, linewidth=0)
        if label and (e - s) >= 12:  # Skip text on tiny segments.
            ax.text(
                (s + e) / 2, 0.5, label,
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
                transform=ax.get_xaxis_transform(),
            )
    ax.set_xlim(0, n_frames)
    ax.set_yticks([])
    ax.set_ylabel("GT", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_model_row(
    ax,
    row: np.ndarray,
    n_frames: int,
    n_syll: int,
    cmap,
    title: str,
) -> None:
    """Draw syllable bands colored by syllable ID."""
    if row.size == 0:
        ax.set_xlim(0, n_frames)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=7, rotation=0, labelpad=28, ha="right", va="center")
        return
    changes = np.where(np.diff(row) != 0)[0] + 1
    segment_starts = np.concatenate([[0], changes])
    segment_ends = np.concatenate([changes, [row.size]])

    for s, e in zip(segment_starts, segment_ends):
        syll = int(row[s])
        ax.axvspan(s, e, color=cmap(syll / max(n_syll - 1, 1)), alpha=0.9, linewidth=0)

    ax.set_xlim(0, n_frames)
    ax.set_yticks([])
    ax.set_ylabel(title, fontsize=7, rotation=0, labelpad=28, ha="right", va="center")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_clip_figure(
    clip_idx: int,
    gt_row: np.ndarray,
    model_rows: list[tuple[str, np.ndarray, int]],
    keypoint_traj: np.ndarray,
    kp_names: list[str],
    output_path: Path,
    n_frames: int,
    n_snapshots: int = 8,
) -> None:
    """Render one per-clip figure: filmstrip + 1 GT + K model rows."""
    k = len(model_rows)

    # Outer layout: filmstrip (tall) + (k+1) thin band rows.
    fig = plt.figure(figsize=(16, 3.2 + 0.9 * (k + 1)))
    outer = GridSpec(
        2, 1, height_ratios=[3.0, 0.9 * (k + 1)],
        hspace=0.25, figure=fig,
    )
    _draw_filmstrip_row(fig, outer[0], keypoint_traj, kp_names, n_frames, n_snapshots)

    band_gs = GridSpecFromSubplotSpec(
        k + 1, 1, subplot_spec=outer[1], hspace=0.55,
    )
    ax_gt = fig.add_subplot(band_gs[0])
    _draw_gt_row(ax_gt, gt_row, n_frames)
    ax_gt.set_title(
        f"Clip {clip_idx} — Ground-truth behavior vs top-{k} KPMS models",
        fontsize=11,
    )

    cmap = plt.get_cmap("tab20")
    ax_last = None
    for i, (label, row, n_syll) in enumerate(model_rows, start=1):
        ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
        r = row[:n_frames]
        _draw_model_row(ax_m, r, n_frames, n_syll, cmap, title=label)
        ax_last = ax_m

    if ax_last is not None:
        ax_last.set_xlabel("Frame")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {output_path}")


def _render_combined_figure(
    clip_indices: list[int],
    gt_rows: list[np.ndarray],
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]],
    keypoint_trajs: list[np.ndarray],
    kp_names: list[str],
    output_path: Path,
    n_frames: int,
    n_snapshots: int = 8,
) -> None:
    """Render 3-column side-by-side combined figure (filmstrip + bands)."""
    if len(clip_indices) != 3:
        raise ValueError(f"Expected 3 clips for combined figure, got {len(clip_indices)}.")
    k = len(model_rows_per_clip[0])

    fig = plt.figure(figsize=(3 * 13, 3.2 + 0.9 * (k + 1)))
    # 2 outer rows × 3 columns: filmstrip row (tall) + band row (thin) per column.
    outer = GridSpec(
        2, 3, height_ratios=[3.0, 0.9 * (k + 1)],
        hspace=0.25, wspace=0.1, figure=fig,
    )

    cmap = plt.get_cmap("tab20")
    for col in range(3):
        # Filmstrip for column `col`.
        _draw_filmstrip_row(
            fig, outer[0, col], keypoint_trajs[col], kp_names, n_frames, n_snapshots,
        )

        # Band stack for column `col`.
        band_gs = GridSpecFromSubplotSpec(
            k + 1, 1, subplot_spec=outer[1, col], hspace=0.55,
        )
        ax_gt = fig.add_subplot(band_gs[0])
        _draw_gt_row(ax_gt, gt_rows[col], n_frames)
        ax_gt.set_title(f"Clip {clip_indices[col]}", fontsize=11)

        ax_last = None
        for i, (label, row, n_syll) in enumerate(model_rows_per_clip[col], start=1):
            ax_m = fig.add_subplot(band_gs[i], sharex=ax_gt)
            r = row[:n_frames]
            _draw_model_row(ax_m, r, n_frames, n_syll, cmap, title=label)
            ax_last = ax_m
        if ax_last is not None:
            ax_last.set_xlabel("Frame")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {output_path}")


def render(
    sweep_dir: Path,
    rank: str = "mse",
    top_k: int = 5,
    clips: list[int] | None = None,
    labels_path: Path | None = None,
    labeled_clip_range: tuple[int, int] = (642, 741),
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    n_frames: int = 250,
    n_snapshots: int = 8,
    output_format: str = "video",
    fps: int = 30,
) -> None:
    """Entry point that coordinates the whole rendering pipeline."""
    sweep_dir = Path(sweep_dir).resolve()
    results_path = sweep_dir / "sweep_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"sweep_results.json not found under {sweep_dir}")
    with results_path.open() as f:
        results = json.load(f)

    top = _select_topk(results["all_results"], rank=rank, top_k=top_k)
    if not top:
        raise RuntimeError("No valid fits in sweep_results.json — cannot render.")
    log.info(f"Top-{top_k} by {rank}:")
    for k, r in enumerate(top, 1):
        log.info(
            f"  M{k}: {r['setting']} seed={r['seed']} "
            f"MSE={r['reconstruction_mse']:.4f} "
            f"NMI={r.get('nmi_global', float('nan')):.3f}"
        )

    # Labels + balanced split + keypoint paths defaults
    repo_root = get_repo_root()
    if labels_path is None:
        labels_path = repo_root / "data" / "rodent" / "rodent_labels.csv"
    if balanced_split_path is None:
        balanced_split_path = repo_root / "data" / "rodent" / "rodent_balanced_splits.json"
    if reference_data_path is None:
        reference_data_path = repo_root / "data" / "rodent" / "rodent_reference_clips.h5"
    if stac_xml_path is None:
        stac_xml_path = repo_root / "c2a" / "assets" / "rodent.xml"

    frame_labels, labeled_clip_indices, _ = load_behavior_frames(
        labels_path, clip_range=labeled_clip_range, n_frames=n_frames,
    )
    all_clip_indices = _load_balanced_split(balanced_split_path)
    intersect_mask = np.isin(labeled_clip_indices, all_clip_indices)
    log.info(
        f"Labeled clips: {len(labeled_clip_indices)}; intersect with fit set: "
        f"{int(intersect_mask.sum())}"
    )

    # Load keypoints via FK for the filmstrip row. Shape: [N_fit, T, K, 3].
    # Matches the positional order of ``all_clip_indices``.
    log.info("Loading keypoints via FK (one-time; ~30s)...")
    keypoints, kp_names = _load_keypoints(
        str(reference_data_path), str(stac_xml_path), str(balanced_split_path),
    )
    log.info(f"Keypoints loaded: {keypoints.shape}")

    # Select the 3 clips to render.
    if clips is not None:
        if len(clips) != 3:
            raise ValueError(f"--clips must have exactly 3 values, got {len(clips)}.")
        for ci in clips:
            if ci not in labeled_clip_indices or ci not in all_clip_indices:
                raise ValueError(
                    f"Clip {ci} is not in both the labeled range and the fit "
                    f"set (labeled={list(labeled_clip_indices[:5])}..., "
                    f"fit={list(all_clip_indices[:5])}...)."
                )
        selected_clips = list(clips)
    else:
        selected_clips = _auto_select_clips(
            frame_labels, labeled_clip_indices, intersect_mask,
        )
    log.info(f"Rendering clips: {selected_clips}")

    # Position lookups.
    lbl_pos = {int(c): i for i, c in enumerate(labeled_clip_indices)}
    fit_pos = {int(c): i for i, c in enumerate(all_clip_indices)}

    # Load syllables for each top-K model.
    syllables_per_model: list[list[np.ndarray]] = []
    model_titles: list[str] = []
    model_n_syll: list[int] = []
    for k, r in enumerate(top, 1):
        project_dir = Path(r["project_dir"])
        if not project_dir.is_absolute():
            project_dir = get_repo_root() / project_dir
        sylls = _load_syllables_for_fit(project_dir, r["model_name"])
        if not sylls:
            raise RuntimeError(
                f"M{k}: no recordings found in results.h5 at "
                f"{project_dir / r['model_name']}."
            )
        if len(sylls) != len(all_clip_indices):
            log.warning(
                f"M{k}: results.h5 has {len(sylls)} recordings but balanced "
                f"split has {len(all_clip_indices)} clips. Using the shorter."
            )
        syllables_per_model.append(sylls)
        # Compact one-line label: trim the `_l10_arhmm` suffix (constant in
        # this sweep) to reduce clutter on the left margin.
        short = r["setting"].replace("_l10_arhmm", "")
        model_titles.append(
            f"M{k} {short}/s{r['seed']}  NMI={r.get('nmi_global', float('nan')):.3f}"
        )
        # Fallback max must span every clip, not just sylls[0] — any single
        # clip may only exercise a subset of the codebook.
        n_states_fallback = max(1, max(int(np.max(s)) for s in sylls) + 1)
        model_n_syll.append(int(r.get("n_states", n_states_fallback)))

    output_dir = sweep_dir / "figures"

    # Per-clip figures.
    gt_rows: list[np.ndarray] = []
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]] = []
    keypoint_trajs: list[np.ndarray] = []

    for ci in selected_clips:
        gt_row = frame_labels[lbl_pos[ci]]
        pos = fit_pos[ci]
        kp_traj = keypoints[pos]
        model_rows = [
            (model_titles[k], syllables_per_model[k][pos], model_n_syll[k])
            for k in range(len(top))
        ]
        if output_format in ("png", "both"):
            _render_clip_figure(
                clip_idx=ci,
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                output_path=output_dir / f"topk_bands_{rank}_clip{ci}.png",
                n_frames=n_frames,
                n_snapshots=n_snapshots,
            )
        if output_format in ("video", "both"):
            _render_clip_video(
                clip_idx=ci,
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                output_path=output_dir / f"topk_bands_{rank}_clip{ci}.mp4",
                n_frames=n_frames,
                fps=fps,
            )
        gt_rows.append(gt_row)
        model_rows_per_clip.append(model_rows)
        keypoint_trajs.append(kp_traj)

    # Combined 3-column figure/video.
    if output_format in ("png", "both"):
        _render_combined_figure(
            clip_indices=selected_clips,
            gt_rows=gt_rows,
            model_rows_per_clip=model_rows_per_clip,
            keypoint_trajs=keypoint_trajs,
            kp_names=kp_names,
            output_path=output_dir / f"topk_bands_{rank}_combined.png",
            n_frames=n_frames,
            n_snapshots=n_snapshots,
        )
    if output_format in ("video", "both"):
        _render_combined_video(
            clip_indices=selected_clips,
            gt_rows=gt_rows,
            model_rows_per_clip=model_rows_per_clip,
            keypoint_trajs=keypoint_trajs,
            kp_names=kp_names,
            output_path=output_dir / f"topk_bands_{rank}_combined.mp4",
            n_frames=n_frames,
            fps=fps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument(
        "--rank", choices=("mse", "nmi"), default="mse",
        help="How to rank the top-K models (default: mse).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--clips", type=int, nargs=3, default=None,
        help="Override auto-selected clips (3 ints, all in labeled ∩ fit).",
    )
    parser.add_argument(
        "--labels-path", type=Path, default=None,
        help="Path to rodent_labels.csv (default: <repo>/data/rodent/rodent_labels.csv).",
    )
    parser.add_argument(
        "--balanced-split", type=Path, default=None,
        help="Path to balanced splits JSON.",
    )
    parser.add_argument("--clip-range", type=int, nargs=2, default=[642, 741])
    parser.add_argument("--n-frames", type=int, default=250)
    parser.add_argument(
        "--reference-data", type=Path, default=None,
        help="Reference clips H5 (default: <repo>/data/rodent/rodent_reference_clips.h5).",
    )
    parser.add_argument(
        "--stac-xml", type=Path, default=None,
        help="stac-mjx rodent XML (default: <repo>/c2a/assets/rodent.xml).",
    )
    parser.add_argument(
        "--n-snapshots", type=int, default=8,
        help="Number of skeleton snapshots across the filmstrip row (PNG only).",
    )
    parser.add_argument(
        "--format", dest="output_format", choices=("video", "png", "both"),
        default="video",
        help=(
            "Output format. 'video' (default) = animated mp4 with skeleton + "
            "moving cursor. 'png' = static filmstrip + bands. 'both' = emit "
            "both for every clip."
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    args = parser.parse_args()

    render(
        sweep_dir=args.sweep_dir,
        rank=args.rank,
        top_k=args.top_k,
        clips=args.clips,
        labels_path=args.labels_path,
        labeled_clip_range=tuple(args.clip_range),
        balanced_split_path=args.balanced_split,
        reference_data_path=args.reference_data,
        stac_xml_path=args.stac_xml,
        n_frames=args.n_frames,
        n_snapshots=args.n_snapshots,
        output_format=args.output_format,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
