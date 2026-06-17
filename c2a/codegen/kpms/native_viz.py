"""Wrapper around ``keypoint_moseq.viz.generate_trajectory_plots``.

KPMS ships a native syllable-grid visualization: per-syllable trajectory
panels showing the average pose evolution across all bouts in 3D, with
projection planes and optional GIF / MP4 export. The layout is the
canonical "MoSeq grid" the lab is used to.

This module factors the native syllable-grid rendering out so the sweep
auto-render (``c2a/kpms/sweep_impl.py``) and the auto-merge orchestrator
(``c2a/kpms/merge_states.py``) can both produce the same artifact:

* :func:`render_native_for_fit`  — render one fit's trajectory plots.
* :func:`render_native_for_topk` — orchestrate top-K fits from a
  ``sweep_results.json`` (loads keypoints once, dispatches per fit).

A ``syllable_override`` argument lets the caller substitute the
``results[rec]["syllable"]`` arrays at render time, which is how the
auto-merge path renders the **merged** syllable grid without writing a
new ``results.h5`` to disk.

Import side-effects (kept local to this module):
  * Mocks ``keypoint_moseq.analysis`` to avoid pulling bokeh + numpy 2.x
    incompatibilities. KPMS only uses analysis for dashboards, never for
    trajectory_plots itself.
  * Patches ``kpms_viz.rasterize_figure`` for matplotlib >= 3.8 (the
    upstream uses removed ``canvas.tostring_rgb()``).
  * Patches ``kpms_viz.matplotlib_colormap_to_plotly`` for numpy 2.x.

These patches live behind a one-shot guard so multiple imports of this
module are idempotent.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# Lazy / patched KPMS import
# =============================================================================


_KPMS_LOADED = False
_kpms_io: Any = None
_kpms_viz: Any = None


def _ensure_kpms_loaded():
    """Load keypoint_moseq with the patches inline. Idempotent."""
    global _KPMS_LOADED, _kpms_io, _kpms_viz
    if _KPMS_LOADED:
        return _kpms_io, _kpms_viz

    # Mock keypoint_moseq.analysis BEFORE the package import: KPMS's
    # __init__ imports analysis -> bokeh -> numpy.bool8 (numpy 2.x removed
    # this), exploding any user of kpms_viz. Trajectory plots don't touch
    # analysis at all, so the mock is harmless for our use.
    if "keypoint_moseq.analysis" not in sys.modules:
        sys.modules["keypoint_moseq.analysis"] = type(sys)("mock_analysis")

    import matplotlib  # noqa: F401
    import matplotlib.backends.backend_agg
    import matplotlib.pyplot as plt
    from keypoint_moseq import io as kpms_io
    from keypoint_moseq import viz as kpms_viz

    # matplotlib >= 3.8 dropped FigureCanvasAgg.tostring_rgb. Provide a
    # buffer_rgba-based equivalent.
    if not hasattr(matplotlib.backends.backend_agg.FigureCanvasAgg, "tostring_rgb"):
        def _patched_rasterize(fig):
            canvas = fig.canvas
            canvas.draw()
            buf = canvas.buffer_rgba()
            rgba = np.frombuffer(buf, dtype="uint8").reshape(
                canvas.get_width_height()[::-1] + (4,)
            )
            return rgba[:, :, :3].copy()
        kpms_viz.rasterize_figure = _patched_rasterize

    # numpy 2.x: scalar repr changed. KPMS's plotly colormap converter
    # casts via repr-then-int; build the colormap explicitly instead.
    def _patched_cmap_to_plotly(cmap):
        cmap_obj = plt.colormaps[cmap]
        n = 255
        h = 1.0 / (n - 1)
        out = []
        for k in range(n):
            C = (np.array(cmap_obj(k * h)[:3]) * 255).astype(int)
            out.append([k * h, f"rgb({C[0]}, {C[1]}, {C[2]})"])
        return out
    kpms_viz.matplotlib_colormap_to_plotly = _patched_cmap_to_plotly

    _kpms_io = kpms_io
    _kpms_viz = kpms_viz
    _KPMS_LOADED = True
    return _kpms_io, _kpms_viz


# =============================================================================
# Rodent skeleton (matches c2a.codegen.kpms.sweep_impl.RODENT_SKELETON)
# =============================================================================


RODENT_SKELETON: list[tuple[str, str]] = [
    ("Snout", "SpineF"),
    ("SpineF", "SpineM"),
    ("SpineM", "SpineL"),
    ("SpineL", "TailBase"),
    ("Snout", "EarL"),
    ("Snout", "EarR"),
    ("SpineF", "ShoulderL"),
    ("ShoulderL", "ElbowL"),
    ("ElbowL", "WristL"),
    ("WristL", "HandL"),
    ("SpineF", "ShoulderR"),
    ("ShoulderR", "ElbowR"),
    ("ElbowR", "WristR"),
    ("WristR", "HandR"),
    ("SpineL", "HipL"),
    ("HipL", "KneeL"),
    ("KneeL", "AnkleL"),
    ("AnkleL", "FootL"),
    ("SpineL", "HipR"),
    ("HipR", "KneeR"),
    ("KneeR", "AnkleR"),
    ("AnkleR", "FootR"),
]


# =============================================================================
# Per-fit renderer
# =============================================================================


# =============================================================================
# Trajectory math (mirrors KPMS's pre/post-window + centroid + heading
# alignment) — used by our planes-grid renderer below
# =============================================================================


def _bout_starts(syllables: np.ndarray) -> np.ndarray:
    """Indices where the syllable changes (incl. index 0)."""
    if syllables.size == 0:
        return np.empty(0, dtype=int)
    diff = np.concatenate([[1], np.diff(syllables)])
    return np.where(diff != 0)[0]


def _rot_z(theta: float) -> np.ndarray:
    """Right-handed rotation about z by ``theta`` radians (3x3)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(3)
    R[0, 0] =  c; R[0, 1] = -s
    R[1, 0] =  s; R[1, 1] =  c
    return R


def _compute_syllable_trajectories(
    coordinates: dict[str, np.ndarray],
    results: dict,
    *,
    pre_frames: int,
    post_frames: int,
    min_frequency: float = 0.0,
    min_duration: int = 1,
    n_samples_per_syllable: int = 50,
    align_centroid: bool = True,
    align_heading: bool = True,
    rng_seed: int = 0,
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, float]]:
    """Per-syllable mean trajectory across sampled bouts.

    Mirrors KPMS's trajectory-extraction recipe:
      1. Find bout starts where ``syllable[t] != syllable[t-1]``.
      2. For each bout start, extract the keypoint window
         ``[start-pre_frames, start+post_frames)``.
      3. Translate by ``-centroid[start]`` (so origin = bout-start centroid).
      4. Rotate by ``-heading[start]`` about z (so heading = +x at bout start).
      5. Average across sampled instances per syllable.

    Args:
        coordinates: ``{rec_key: keypoints_TK3}`` (mm-scaled).
        results: KPMS results dict ``{rec_key: {"syllable", "centroid",
            "heading"}}``. ``centroid`` should already be on the same
            scale as ``coordinates`` (mm).
        pre_frames / post_frames: Window before/after bout start, in frames.
        min_frequency: Drop syllables whose total-frames fraction is below
            this. ``0.0`` keeps everything.
        min_duration: Drop bouts shorter than this (frames).
        n_samples_per_syllable: Cap per-syllable bout count for speed.
        align_centroid / align_heading: Toggle the alignment steps.
        rng_seed: Seed for the random sampler when bouts > cap.

    Returns:
        ``(trajectories, n_bouts_used, mean_duration_frames)`` where:
          * ``trajectories[s]`` has shape
            ``(pre_frames + post_frames, K, 3)``.
          * ``n_bouts_used[s]`` = number of bouts averaged for syllable s
            (post-min_duration filter, post-cap).
          * ``mean_duration_frames[s]`` = mean bout duration (in frames)
            across ALL eligible bouts for that syllable (pre-cap), so the
            statistic isn't biased by the per-syllable sampling cap.
    """
    rec_keys = sorted(coordinates.keys())
    n_window = pre_frames + post_frames
    rng = np.random.default_rng(rng_seed)

    # First pass: per-syllable frame counts (for min_frequency) and
    # eligible bouts (for sampling), plus all eligible bout durations
    # (for the mean-duration statistic — kept separate from the sampled
    # bouts so the cap doesn't bias the duration estimate).
    bouts_per_syll: dict[int, list[tuple[str, int]]] = {}
    durations_per_syll: dict[int, list[int]] = {}
    total_frames = 0
    syll_frame_counts: dict[int, int] = {}
    for rec in rec_keys:
        s_arr = np.asarray(results[rec]["syllable"]).astype(int)
        T = s_arr.size
        total_frames += T
        uniq, counts = np.unique(s_arr, return_counts=True)
        for u, c in zip(uniq, counts):
            syll_frame_counts[int(u)] = syll_frame_counts.get(int(u), 0) + int(c)
        starts = _bout_starts(s_arr)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else T
            duration = end - start
            if duration < min_duration:
                continue
            if start - pre_frames < 0 or start + post_frames > T:
                continue  # window out of clip
            sid = int(s_arr[start])
            bouts_per_syll.setdefault(sid, []).append((rec, int(start)))
            durations_per_syll.setdefault(sid, []).append(int(duration))

    # Apply min_frequency filter.
    eligible_sylls = [
        s for s in bouts_per_syll
        if syll_frame_counts.get(s, 0) / max(total_frames, 1) >= min_frequency
    ]

    trajectories: dict[int, np.ndarray] = {}
    n_bouts_used: dict[int, int] = {}
    mean_duration_frames: dict[int, float] = {}
    K = next(iter(coordinates.values())).shape[1]

    for s in sorted(eligible_sylls):
        bouts = bouts_per_syll[s]
        # Mean duration uses all eligible bouts for this syllable, not the
        # capped sample, so the statistic is stable as n_samples_per_syllable
        # changes.
        mean_duration_frames[s] = (
            float(np.mean(durations_per_syll[s])) if durations_per_syll[s] else 0.0
        )
        if len(bouts) > n_samples_per_syllable:
            idx = rng.choice(len(bouts), size=n_samples_per_syllable, replace=False)
            sampled = [bouts[i] for i in idx]
        else:
            sampled = bouts

        windows = np.empty((len(sampled), n_window, K, 3), dtype=np.float64)
        for i, (rec, start) in enumerate(sampled):
            kp = np.asarray(coordinates[rec])  # [T, K, 3]
            window = kp[start - pre_frames: start + post_frames].astype(np.float64)
            if align_centroid and "centroid" in results[rec]:
                c = np.asarray(results[rec]["centroid"][start])
                offset = np.zeros(3)
                offset[: c.shape[-1]] = c[: 3]
                window = window - offset
            if align_heading and "heading" in results[rec]:
                h = float(np.asarray(results[rec]["heading"])[start])
                R = _rot_z(-h)  # rotate so heading aligns with +x
                window = window @ R.T
            windows[i] = window

        trajectories[s] = windows.mean(axis=0)
        n_bouts_used[s] = int(len(sampled))

    return trajectories, n_bouts_used, mean_duration_frames


# =============================================================================
# Custom planes-as-rows × syllables-as-cols grid
# =============================================================================


_PROJECTION_DIM_INDEX: dict[str, tuple[int, int]] = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def _resolve_skeleton_indices(
    skeleton: list[tuple[str, str]], bodyparts: list[str],
) -> list[tuple[int, int]]:
    name_to_idx = {n: i for i, n in enumerate(bodyparts)}
    edges: list[tuple[int, int]] = []
    for a, b in skeleton:
        if a in name_to_idx and b in name_to_idx:
            edges.append((name_to_idx[a], name_to_idx[b]))
    return edges


def _compute_cell_view(
    trajectories: dict[int, np.ndarray],
    page_sylls: list[int],
    projection_planes: list[str],
    pad_frac: float = 0.05,
    min_extent_mm: float = 20.0,
) -> tuple[
    dict[str, tuple[float, float]],
    dict[tuple[int, str], tuple[float, float]],
]:
    """Per-row SHARED axis spans + per-cell centers.

    The renderer uses these to enforce that every cell in row ``XY``
    has the same xlim/ylim span (set to the largest extent across
    syllables in that row), centered on each cell's own data; ditto
    for row ``XZ``. The horizontal axis (X) span is also forced to
    match across rows (XY and XZ both project on X). Combined with
    a uniform-width GridSpec, this guarantees:

      * The rodent body is rendered at the same physical mm-per-inch
        scale in every cell across the whole figure.
      * All columns line up between XY and XZ rows (no rear-cell
        being narrower than a walking-cell because of per-cell data
        extents).
      * Each cell is centered on its own syllable's data, so a
        small-movement syllable still fills the middle of its panel.

    Args:
        trajectories: ``{syllable_id: [T, K, 3]}``.
        page_sylls: Syllables on this page (column order).
        projection_planes: Row order.
        pad_frac: Multiplier (1 + pad_frac) applied to each axis so the
            rodent doesn't kiss the panel edges.
        min_extent_mm: Floor on each axis range so a syllable that's
            barely moving still has a sane window. Body is ~80 mm long
            so anything below this floors at near-body scale.

    Returns:
        ``(row_ranges, cell_centers)`` where:
          * ``row_ranges[plane] = (range_x, range_y)`` — the (mm) span
            used by every cell in this row. ``range_x`` is identical
            across rows (X axis is shared between XY and XZ).
          * ``cell_centers[(col, plane)] = (cx, cy)`` — per-cell
            data centroid for window centering.
    """
    cell_centers: dict[tuple[int, str], tuple[float, float]] = {}
    row_x_max = 0.0
    row_y_max: dict[str, float] = {plane: 0.0 for plane in projection_planes}

    for plane in projection_planes:
        i0, i1 = _PROJECTION_DIM_INDEX[plane]
        for col, s in enumerate(page_sylls):
            traj = trajectories[s]
            a = traj[..., i0]; b = traj[..., i1]
            cell_centers[(col, plane)] = (
                float(0.5 * (a.min() + a.max())),
                float(0.5 * (b.min() + b.max())),
            )
            row_x_max = max(row_x_max, float(a.max() - a.min()))
            row_y_max[plane] = max(row_y_max[plane], float(b.max() - b.min()))

    pad = 1.0 + pad_frac
    shared_x = max(row_x_max * pad, min_extent_mm)
    row_ranges = {
        plane: (shared_x, max(row_y_max[plane] * pad, min_extent_mm))
        for plane in projection_planes
    }
    return row_ranges, cell_centers


def _render_planes_grid(
    trajectories: dict[int, np.ndarray],
    n_bouts_used: dict[int, int],
    mean_duration_frames: dict[int, float],
    skeleton_idx: list[tuple[int, int]],
    output_path: Path,
    *,
    projection_planes: list[str] = ("xy", "xz"),
    n_snapshots: int = 5,
    keypoint_colormap: str = "autumn",
    title: str = "",
    max_syllables_per_page: int = 5,
    fps: int = 30,
) -> None:
    """Render rows = projection planes, cols = syllables.

    Each cell shows ``n_snapshots`` 2D-projected skeleton overlays from
    one syllable's mean trajectory, colored by time within the bout.
    Cells share a uniform spatial scale (mm/pixel) but are centered on
    each syllable's own data — small-movement syllables fill the panel
    at the same scale as large-movement ones.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    syll_ids = sorted(trajectories.keys())
    if not syll_ids:
        log.warning(f"  No syllables survived filtering for {output_path}.")
        return

    n_planes = len(projection_planes)
    n_pages = (len(syll_ids) + max_syllables_per_page - 1) // max_syllables_per_page
    cmap = plt.get_cmap(keypoint_colormap)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for page in range(n_pages):
        lo = page * max_syllables_per_page
        hi = min(lo + max_syllables_per_page, len(syll_ids))
        page_sylls = syll_ids[lo:hi]
        n_cols = len(page_sylls)

        row_ranges, cell_centers = _compute_cell_view(
            trajectories, page_sylls, list(projection_planes),
        )

        # Layout: shared X-axis span across rows (so XY and XZ columns
        # align), uniform GridSpec cell widths, per-row data-aspect for
        # cell heights. mm-per-inch is fixed by targeting cell width =
        # cell_w_in, which makes the rodent's physical size identical
        # across every cell in the figure.
        shared_x_mm = row_ranges[projection_planes[0]][0]
        row_h_mm = [row_ranges[p][1] for p in projection_planes]
        cell_w_in = 2.5
        mm_per_inch = shared_x_mm / cell_w_in
        label_col_in = 0.5
        fig_w = cell_w_in * n_cols + label_col_in + 0.4
        fig_h = sum(row_h_mm) / mm_per_inch + 1.2

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            n_planes, n_cols + 1,
            width_ratios=[label_col_in] + [cell_w_in] * n_cols,
            height_ratios=row_h_mm,
            wspace=0.08, hspace=0.18, figure=fig,
            left=0.03, right=0.99, top=0.97, bottom=0.04,
        )

        for row, plane in enumerate(projection_planes):
            i0, i1 = _PROJECTION_DIM_INDEX[plane]
            range_x, range_y = row_ranges[plane]
            ax_lbl = fig.add_subplot(gs[row, 0])
            ax_lbl.text(
                0.5, 0.5, plane.upper(),
                ha="center", va="center",
                fontsize=22, fontweight="bold",
            )
            ax_lbl.axis("off")
            for col, s in enumerate(page_sylls):
                ax = fig.add_subplot(gs[row, col + 1])
                traj = trajectories[s]
                T = traj.shape[0]
                snap_idx = np.linspace(0, T - 1, n_snapshots).astype(int)
                for j, t in enumerate(snap_idx):
                    color = cmap(j / max(n_snapshots - 1, 1))
                    pts = traj[t]
                    for ia, ib in skeleton_idx:
                        ax.plot(
                            [pts[ia, i0], pts[ib, i0]],
                            [pts[ia, i1], pts[ib, i1]],
                            color=color, linewidth=2.0, alpha=0.9,
                        )
                    ax.scatter(
                        pts[:, i0], pts[:, i1],
                        c=[color], s=14, alpha=0.9,
                    )
                cx, cy = cell_centers[(col, plane)]
                ax.set_xlim(cx - range_x / 2.0, cx + range_x / 2.0)
                ax.set_ylim(cy - range_y / 2.0, cy + range_y / 2.0)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    n_used = n_bouts_used.get(s, 0)
                    dur_f = mean_duration_frames.get(s, 0.0)
                    dur_s = dur_f / float(fps) if fps > 0 else 0.0
                    ax.set_title(
                        f"Syll {s}\n"
                        f"{n_used} bouts · avg {dur_s:.2f}s ({dur_f:.0f}f)",
                        fontsize=13,
                    )

        out_path = output_path
        if n_pages > 1:
            out_path = output_path.with_name(
                f"{output_path.stem}_p{page + 1}{output_path.suffix}"
            )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(
            f"  PNG: {out_path}  "
            f"({n_planes} planes × {n_cols} syllables, native per-cell extents)"
        )


def _render_planes_grid_animated(
    trajectories: dict[int, np.ndarray],
    n_bouts_used: dict[int, int],
    mean_duration_frames: dict[int, float],
    skeleton_idx: list[tuple[int, int]],
    output_path: Path,
    *,
    projection_planes: list[str] = ("xy", "xz"),
    keypoint_colormap: str = "autumn",
    title: str = "",
    max_syllables_per_page: int = 5,
    fps: int = 30,
) -> None:
    """Animated GIF of the planes × syllables grid (lockstep across columns).

    Same cell layout as :func:`_render_planes_grid` (rows = projection
    planes, cols = syllables) but instead of overlaying snapshots, each
    cell animates through the bout window. All cells advance in lockstep
    so you can compare syllable timings at a glance.

    Color follows ``keypoint_colormap`` over the bout time fraction —
    so the rodent fades from start-of-bout color to end-of-bout color
    as the loop runs.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.gridspec import GridSpec

    syll_ids = sorted(trajectories.keys())
    if not syll_ids:
        log.warning(f"  No syllables for animated grid {output_path}.")
        return

    n_planes = len(projection_planes)
    n_pages = (len(syll_ids) + max_syllables_per_page - 1) // max_syllables_per_page
    cmap = plt.get_cmap(keypoint_colormap)
    # Single static color for the whole animation — the animation
    # already conveys time, a colour cycle on top just produces flicker.
    # cmap(0.4) is a warm orange in the default 'autumn' palette.
    static_color = cmap(0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for page in range(n_pages):
        lo = page * max_syllables_per_page
        hi = min(lo + max_syllables_per_page, len(syll_ids))
        page_sylls = syll_ids[lo:hi]
        n_cols = len(page_sylls)
        T = trajectories[page_sylls[0]].shape[0]

        # Per-row shared spans + per-cell centers (same recipe as the PNG).
        row_ranges, cell_centers = _compute_cell_view(
            trajectories, page_sylls, list(projection_planes),
        )
        shared_x_mm = row_ranges[projection_planes[0]][0]
        row_h_mm = [row_ranges[p][1] for p in projection_planes]
        cell_w_in = 2.5
        mm_per_inch = shared_x_mm / cell_w_in
        label_col_in = 0.5
        fig_w = cell_w_in * n_cols + label_col_in + 0.4
        fig_h = sum(row_h_mm) / mm_per_inch + 1.2

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            n_planes, n_cols + 1,
            width_ratios=[label_col_in] + [cell_w_in] * n_cols,
            height_ratios=row_h_mm,
            wspace=0.08, hspace=0.18, figure=fig,
            left=0.03, right=0.99, top=0.97, bottom=0.04,
        )

        # Pre-create artists for each (plane, syllable) cell.
        cells: list[dict] = []
        for row, plane in enumerate(projection_planes):
            i0, i1 = _PROJECTION_DIM_INDEX[plane]
            range_x, range_y = row_ranges[plane]
            ax_lbl = fig.add_subplot(gs[row, 0])
            ax_lbl.text(
                0.5, 0.5, plane.upper(),
                ha="center", va="center",
                fontsize=22, fontweight="bold",
            )
            ax_lbl.axis("off")
            for col, s in enumerate(page_sylls):
                ax = fig.add_subplot(gs[row, col + 1])
                lines = [
                    ax.plot([], [], "-", linewidth=2.0, alpha=0.9,
                            color=static_color)[0]
                    for _ in skeleton_idx
                ]
                (pts_artist,) = ax.plot([], [], "o", ms=5, alpha=0.9,
                                        color=static_color)
                cx, cy = cell_centers[(col, plane)]
                ax.set_xlim(cx - range_x / 2.0, cx + range_x / 2.0)
                ax.set_ylim(cy - range_y / 2.0, cy + range_y / 2.0)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    n_used = n_bouts_used.get(s, 0)
                    dur_f = mean_duration_frames.get(s, 0.0)
                    dur_s = dur_f / float(fps) if fps > 0 else 0.0
                    ax.set_title(
                        f"Syll {s}\n"
                        f"{n_used} bouts · avg {dur_s:.2f}s",
                        fontsize=13,
                    )
                cells.append({
                    "lines": lines, "pts_artist": pts_artist,
                    "i0": i0, "i1": i1,
                    "traj": trajectories[s],
                })

        def _update(t: int):
            artists = []
            for c in cells:
                pts = c["traj"][t]
                for ln, (ia, ib) in zip(c["lines"], skeleton_idx):
                    ln.set_data(
                        [pts[ia, c["i0"]], pts[ib, c["i0"]]],
                        [pts[ia, c["i1"]], pts[ib, c["i1"]]],
                    )
                    artists.append(ln)
                c["pts_artist"].set_data(pts[:, c["i0"]], pts[:, c["i1"]])
                artists.append(c["pts_artist"])
            return artists

        ani = FuncAnimation(
            fig, _update, frames=T, interval=1000.0 / fps, blit=False,
        )
        out_path = output_path
        if n_pages > 1:
            out_path = output_path.with_name(
                f"{output_path.stem}_p{page + 1}{output_path.suffix}"
            )
        # PillowWriter writes animated GIFs without ffmpeg dependencies.
        ani.save(str(out_path), writer=PillowWriter(fps=fps))
        plt.close(fig)
        log.info(
            f"  GIF: {out_path}  ({T} frames @ {fps} fps, "
            f"{n_planes} planes × {n_cols} syllables)"
        )


# =============================================================================
# Per-fit renderer
# =============================================================================


def render_native_for_fit(
    project_dir: str | Path,
    model_name: str,
    coordinates: dict[str, np.ndarray],
    output_dir: Path,
    *,
    syllable_override: list[np.ndarray] | None = None,
    centroid_scale: float = 1000.0,
    pre: float = 0.167,
    post: float = 0.5,
    min_frequency: float = 0.0,
    min_duration: int = 3,
    fps: int = 30,
    projection_planes: tuple[str, ...] = ("xy", "xz"),
    keypoint_colormap: str = "autumn",
    n_snapshots: int = 5,
    n_samples_per_syllable: int = 50,
    align_centroid: bool = True,
    align_heading: bool = True,
    skeleton: list[tuple[str, str]] | None = None,
    title: str | None = None,
    save_png: bool = True,
    save_gifs: bool = True,
    # Legacy kwargs accepted for backward-compat with sweep config but
    # not honoured by this renderer (which produces planes_grid.png /
    # planes_grid.gif, not per-syllable HTMLs/MP4s).
    interactive: bool = False,    # noqa: ARG001
    save_mp4s: bool = False,      # noqa: ARG001
    save_individually: bool = False,  # noqa: ARG001
    density_sample: bool = True,  # noqa: ARG001
) -> None:
    """Render a planes-as-rows × syllables-as-cols grid for one fit.

    Loads ``results.h5`` via KPMS (for ``syllable``, ``centroid``, and
    ``heading`` arrays + the bodyparts config), computes the per-syllable
    mean trajectory using the same pre/post-window + centroid + heading
    alignment KPMS uses, then renders a custom grid:

        +-----+---------+---------+---------+--- ...
        |     | Syll 0  | Syll 1  | Syll 2  |
        | XY  |  pose   |  pose   |  pose   |
        +-----+---------+---------+---------+--- ...
        | XZ  |  pose   |  pose   |  pose   |
        +-----+---------+---------+---------+--- ...

    All syllables that survive ``min_frequency`` and ``min_duration``
    appear (default ``min_frequency=0.0`` keeps everything; raise it if
    you have a long tail of single-frame noise syllables).

    Args:
        project_dir / model_name: KPMS project + fit identifiers.
        coordinates: ``{rec: keypoints_TK3}`` mm-scaled.
        output_dir: Where the planes_grid.png lands.
        syllable_override: Per-recording syllable arrays to substitute
            (used by the auto-merge path; indexing matches sorted
            ``coordinates`` keys).
        centroid_scale: Multiplier for ``results[rec]["centroid"]``
            (default 1000 = m → mm).
        pre / post: Pre/post window (seconds).
        min_frequency / min_duration: Filters; default ``min_frequency=0.0``
            so all syllables that appear at all are rendered.
        fps: Used to convert pre/post seconds → frames.
        projection_planes: Order of rows. Default ``("xy", "xz")``.
        keypoint_colormap: matplotlib colormap for time-coded snapshots.
        n_snapshots: Number of pose snapshots overlaid per panel.
        n_samples_per_syllable: Cap on per-syllable bouts averaged.
        align_centroid / align_heading: Toggle the alignment steps.
        skeleton: Edge list (default :data:`RODENT_SKELETON`).
        title: Optional figure suptitle.
        interactive / save_gifs / save_mp4s / save_individually /
            density_sample: legacy kwargs accepted but unused (this
            renderer produces one PNG per fit).
    """
    kpms_io, _ = _ensure_kpms_loaded()

    project_dir = str(project_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  Loading KPMS results: {project_dir}/{model_name}/results.h5")
    results = kpms_io.load_results(
        project_dir=project_dir, model_name=model_name,
    )

    # Centroid scaling so it lines up with mm coordinates.
    for rec_key in results:
        if "centroid" in results[rec_key]:
            results[rec_key]["centroid"] = (
                np.asarray(results[rec_key]["centroid"]) * centroid_scale
            )

    # Optional syllable substitution for merged-state renders.
    if syllable_override is not None:
        rec_keys = sorted(results.keys())
        if len(syllable_override) != len(rec_keys):
            raise ValueError(
                f"syllable_override has {len(syllable_override)} entries "
                f"but results.h5 has {len(rec_keys)} recordings."
            )
        for i, k in enumerate(rec_keys):
            cur = np.asarray(results[k]["syllable"])
            new = np.asarray(syllable_override[i])
            if new.shape != cur.shape:
                raise ValueError(
                    f"syllable_override[{i}] shape {new.shape} != "
                    f"results[{k}].syllable shape {cur.shape}."
                )
            results[k]["syllable"] = new.astype(cur.dtype)

    kpms_config = kpms_io.load_config(project_dir)
    bodyparts = kpms_config["bodyparts"]

    # Convert pre/post seconds to frames using fps.
    pre_frames = int(round(float(pre) * float(fps)))
    post_frames = int(round(float(post) * float(fps)))
    if pre_frames + post_frames < 2:
        raise ValueError(
            f"pre+post window is {pre_frames + post_frames} frames "
            f"(pre={pre}s, post={post}s, fps={fps}); too short to render."
        )

    coord_keys = sorted(coordinates.keys())
    result_keys = sorted(results.keys())
    if coord_keys != result_keys:
        raise RuntimeError(
            f"coordinates and results keys disagree: coords[:3]={coord_keys[:3]}, "
            f"results[:3]={result_keys[:3]}."
        )

    log.info(
        f"  Computing per-syllable mean trajectories (pre={pre_frames}f, "
        f"post={post_frames}f, min_frequency={min_frequency}, "
        f"min_duration={min_duration}, n_samples_per_syllable={n_samples_per_syllable})"
    )
    trajectories, n_bouts, mean_duration_frames = _compute_syllable_trajectories(
        coordinates=coordinates,
        results=results,
        pre_frames=pre_frames,
        post_frames=post_frames,
        min_frequency=float(min_frequency),
        min_duration=int(min_duration),
        n_samples_per_syllable=int(n_samples_per_syllable),
        align_centroid=align_centroid,
        align_heading=align_heading,
    )
    log.info(f"  {len(trajectories)} syllable(s) survived filter.")
    if mean_duration_frames:
        avg_dur = float(np.mean(list(mean_duration_frames.values())))
        avg_dur_s = avg_dur / float(fps)
        log.info(
            f"  Average bout duration across syllables: "
            f"{avg_dur:.1f} frames ({avg_dur_s:.2f}s @ {fps}fps)"
        )

    skel = list(skeleton) if skeleton is not None else RODENT_SKELETON
    skel_idx = _resolve_skeleton_indices(skel, bodyparts)
    fit_title = title or f"{Path(project_dir).name}/{model_name}"

    if save_png:
        _render_planes_grid(
            trajectories=trajectories,
            n_bouts_used=n_bouts,
            mean_duration_frames=mean_duration_frames,
            skeleton_idx=skel_idx,
            output_path=output_dir / "planes_grid.png",
            projection_planes=list(projection_planes),
            n_snapshots=int(n_snapshots),
            keypoint_colormap=keypoint_colormap,
            title=fit_title,
            fps=int(fps),
        )
    if save_gifs:
        _render_planes_grid_animated(
            trajectories=trajectories,
            n_bouts_used=n_bouts,
            mean_duration_frames=mean_duration_frames,
            skeleton_idx=skel_idx,
            output_path=output_dir / "planes_grid.gif",
            projection_planes=list(projection_planes),
            keypoint_colormap=keypoint_colormap,
            title=fit_title,
            fps=int(fps),
        )
    log.info(f"  Done -> {output_dir}")


# =============================================================================
# Top-K orchestrator
# =============================================================================


def _load_keypoints_mm(
    reference_data_path: Path,
    stac_xml_path: Path,
    balanced_split_path: Path,
) -> tuple[np.ndarray, list[str]]:
    """Load FK keypoints filtered to the balanced split, scaled to mm.

    Wraps ``c2a.codegen.kpms.sweep_impl._load_keypoints`` (which returns m-scale)
    and applies the ×1000 factor KPMS's ``get_limits`` requires.
    """
    from c2a.codegen.kpms.sweep_impl import _load_keypoints

    kp_4d, kp_names = _load_keypoints(
        str(reference_data_path), str(stac_xml_path), str(balanced_split_path),
    )
    return kp_4d * 1000.0, kp_names


def render_native_for_topk(
    sweep_dir: Path,
    *,
    rank: str = "mse",
    top_k: int = 5,
    output_subdir: str = "trajectory_plots",
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    syllable_override_per_fit: dict[int, list[np.ndarray]] | None = None,
    output_dir_override: Path | None = None,
    **traj_kwargs,
) -> dict:
    """Render KPMS-native trajectory plots for the top-K fits in a sweep.

    Args:
        sweep_dir: Directory containing ``sweep_results.json``.
        rank / top_k: Top-K selection (mirrors render_topk_bands).
        output_subdir: Subdirectory under ``{sweep_dir}/figures/`` to
            write per-fit grids (``M1``, ``M2``, ...). Ignored when
            ``output_dir_override`` is set.
        syllable_override_per_fit: Optional ``{fit_rank_1based:
            list_of_syllable_arrays}``. When given for a fit, those
            syllables substitute the original results[rec]["syllable"]
            during render. Used by the auto-merge path to show merged
            syllables.
        output_dir_override: Custom output root. When set, per-fit dirs
            are ``{output_dir_override}/M{k}`` regardless of ``sweep_dir``.
        traj_kwargs: Forwarded to :func:`render_native_for_fit` (pre,
            post, fps, projection_planes, etc.).

    Returns:
        Manifest dict ``{"M1": str, "M2": str, ...}`` mapping fit rank
        to its output directory.
    """
    from c2a.codegen.kpms.keypoint_loader import prepare_keypoints_for_kpms
    from c2a.codegen.kpms.render_topk_bands import _select_topk
    from c2a.paths import get_repo_root

    sweep_dir = Path(sweep_dir).resolve()
    results_path = sweep_dir / "sweep_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"sweep_results.json not under {sweep_dir}; cannot pick top-K."
        )
    with results_path.open() as f:
        sweep_results = json.load(f)
    top = _select_topk(sweep_results["all_results"], rank=rank, top_k=top_k)
    if not top:
        raise RuntimeError(f"No valid fits in {results_path}.")
    log.info(
        f"Native KPMS trajectory plots for top-{len(top)} fits (rank={rank})"
    )

    repo_root = get_repo_root()
    if balanced_split_path is None:
        balanced_split_path = (
            repo_root / "data" / "rodent" / "rodent_balanced_splits.json"
        )
    if reference_data_path is None:
        reference_data_path = (
            repo_root / "data" / "rodent" / "rodent_reference_clips.h5"
        )
    if stac_xml_path is None:
        stac_xml_path = repo_root / "c2a" / "assets" / "rodent.xml"

    log.info("Loading FK keypoints (one-time; ~30s)...")
    keypoints_mm, kp_names = _load_keypoints_mm(
        reference_data_path, stac_xml_path, balanced_split_path,
    )
    coordinates, _ = prepare_keypoints_for_kpms(keypoints_mm)
    log.info(f"  {len(coordinates)} recordings (mm-scaled).")

    if output_dir_override is not None:
        out_root = Path(output_dir_override)
    else:
        out_root = sweep_dir / "figures" / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}
    overrides = syllable_override_per_fit or {}
    for k, r in enumerate(top, start=1):
        project_dir = Path(r["project_dir"])
        if not project_dir.is_absolute():
            project_dir = repo_root / project_dir
        if not (project_dir / r["model_name"] / "results.h5").exists():
            log.warning(
                f"M{k}: {project_dir}/{r['model_name']}/results.h5 missing; "
                f"skipping."
            )
            continue
        m_dir = out_root / f"M{k}"
        log.info(
            f"--- M{k} (setting={r.get('setting')}, seed={r.get('seed')}) "
            f"-> {m_dir} ---"
        )
        render_native_for_fit(
            project_dir=project_dir,
            model_name=r["model_name"],
            coordinates=coordinates,
            output_dir=m_dir,
            syllable_override=overrides.get(k),
            **traj_kwargs,
        )
        manifest[f"M{k}"] = str(m_dir)

    log.info(f"Done. Outputs: {out_root}")
    return manifest
