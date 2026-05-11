"""Per-syllable example-bout grid for the top-K KPMS fits in a sweep.

For each of the top-K fits, walks every clip's syllable sequence, finds
the longest contiguous bouts of each syllable, and renders a paginated
grid of mini skeleton filmstrips (PNG) and animated mini-skeletons (MP4).

Layout (per page):

    +------------+-------+-------+-------+-------+-------+
    |            | Bout1 | Bout2 | Bout3 | Bout4 | Bout5 |
    | Syllable 0 |  ===  |  ===  |  ===  |  ===  |  ===  |
    | Syllable 1 |  ===  |  ===  |  ===  |  ===  |  ===  |
    | Syllable 2 |  ===  |  ===  |  ===  |  ===  |  ===  |
    | Syllable 3 |  ===  |  ===  |  ===  |  ===  |  ===  |
    | Syllable 4 |  ===  |  ===  |  ===  |  ===  |  ===  |
    +------------+-------+-------+-------+-------+-------+

Each ``===`` cell is either a 4-snapshot 3D-skeleton filmstrip (PNG)
or an animated 3D skeleton in lockstep with the page (MP4).

When ``num_states > 5`` (default ``--max-rows-per-page``), additional
pages are written: ``...syllables_5-9.png``, ``..._10-14.png``, etc.

Bouts are picked per the ``--bout-strategy`` flag — default is
**longest** (top-K longest contiguous runs of the syllable across all
clips).

Usage::

    python -m c2a.kpms.render_syllable_grids \\
        --sweep-dir outputs/kpms_sweep --top-k 5 --format both
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from c2a.kpms.render_topk_bands import (
    _load_balanced_split,
    _load_syllables_for_fit,
    _select_topk,
)
from c2a.kpms.sweep_impl import RODENT_SKELETON, _load_keypoints
from c2a.paths import get_repo_root
from c2a.render.topk_bands_render import _resolve_skeleton_edges

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Bouts
# =============================================================================


@dataclass
class Bout:
    """A contiguous run of one syllable in one clip."""

    clip_id: int             # original clip id (in the balanced split)
    fit_pos: int             # positional index into all_clip_indices / keypoints[fit_pos]
    start_frame: int
    end_frame: int           # exclusive
    syllable: int

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame

    def keypoint_slice(self, keypoints: np.ndarray) -> np.ndarray:
        """Return ``[duration, K, 3]`` of keypoints for this bout."""
        return keypoints[self.fit_pos, self.start_frame:self.end_frame]


def find_bouts_in_array(
    syllable_array: np.ndarray, syllable_id: int, min_duration: int = 3,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for each contiguous run of
    ``syllable_id`` of length ``>= min_duration``."""
    if syllable_array.size == 0:
        return []
    mask = (syllable_array == syllable_id).astype(np.int8)
    diff = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_duration]


def collect_bouts(
    syllables_per_clip: list[np.ndarray],
    all_clip_indices: np.ndarray,
    syllable_id: int,
    min_duration: int = 3,
) -> list[Bout]:
    """Across all clips in the fit set, collect every bout of one syllable."""
    out: list[Bout] = []
    for fit_pos, syll_arr in enumerate(syllables_per_clip):
        for s, e in find_bouts_in_array(syll_arr, syllable_id, min_duration):
            out.append(Bout(
                clip_id=int(all_clip_indices[fit_pos]),
                fit_pos=fit_pos,
                start_frame=s, end_frame=e,
                syllable=int(syllable_id),
            ))
    return out


def select_bouts(
    bouts: list[Bout], n: int, strategy: str = "longest",
) -> list[Bout]:
    """Pick ``n`` representative bouts by the given strategy."""
    if not bouts:
        return []
    if strategy == "longest":
        return sorted(bouts, key=lambda b: -b.duration)[:n]
    elif strategy == "median":
        # Sort by absolute distance from median duration; one per clip.
        durs = np.array([b.duration for b in bouts])
        med = float(np.median(durs))
        bouts_sorted = sorted(bouts, key=lambda b: abs(b.duration - med))
        seen_clips: set[int] = set()
        out: list[Bout] = []
        for b in bouts_sorted:
            if b.clip_id in seen_clips:
                continue
            out.append(b); seen_clips.add(b.clip_id)
            if len(out) >= n:
                break
        return out
    elif strategy == "random":
        rng = np.random.default_rng(0)
        idx = rng.choice(len(bouts), size=min(n, len(bouts)), replace=False)
        return [bouts[i] for i in idx]
    else:
        raise ValueError(f"Unknown bout strategy: {strategy!r}")


# =============================================================================
# Clarity styling
# =============================================================================
#
# "compact" reproduces the original style: small cells, uniform steelblue
# points + gray skeleton lines. "clear" enlarges cells, color-codes body
# parts (head=orange, spine=black, left limbs=blue, right limbs=red),
# and emphasizes the snout marker so head orientation is unambiguous.


HEAD_KP_NAMES: frozenset[str] = frozenset({"Snout", "EarL", "EarR"})
SPINE_KP_PREFIXES: tuple[str, ...] = ("Spine", "TailBase")


def _kp_color_clear(name: str) -> str:
    """Body-part color for one keypoint name, in the 'clear' style."""
    if name in HEAD_KP_NAMES:
        return "darkorange"
    if any(name.startswith(p) for p in SPINE_KP_PREFIXES):
        return "black"
    if name.endswith("L"):
        return "steelblue"
    if name.endswith("R"):
        return "crimson"
    return "gray"


def _edge_color_clear(name_a: str, name_b: str) -> str:
    """Edge color for the skeleton segment between two keypoints. If
    both endpoints are the same body group, use that group's color;
    spine wins for mixed segments (so the trunk is always black)."""
    ca, cb = _kp_color_clear(name_a), _kp_color_clear(name_b)
    if ca == cb:
        return ca
    if "black" in (ca, cb):
        return "black"
    return "gray"


def _kp_size_clear(name: str, base: float) -> float:
    """Snout gets ~2× the marker size so head direction is unambiguous."""
    return base * 2.2 if name == "Snout" else base


def _resolve_styling(
    clarity: str, kp_names: list[str], edges: list[tuple[int, int]],
) -> dict:
    """Pre-resolve per-keypoint / per-edge colors and sizes.

    Returned dict carries a few simple lookups that the per-cell
    renderers consume — keeping the per-frame draw loop allocation-free.
    """
    if clarity == "compact":
        n = len(kp_names)
        return {
            "clarity": "compact",
            "kp_colors": ["steelblue"] * n,
            "kp_sizes_static": [6.0] * n,
            "kp_sizes_anim": [3.0] * n,
            "edge_colors_static": ["gray"] * len(edges),
            "edge_colors_anim": ["gray"] * len(edges),
            "linewidth_static": 1.4,
            "linewidth_anim": 1.2,
            "cell_w_in": 2.4, "cell_h_in": 2.0,
            "label_fontsize": 9,
            "title_fontsize": 6,
        }
    if clarity == "clear":
        kp_colors = [_kp_color_clear(n) for n in kp_names]
        kp_sizes_static = [_kp_size_clear(n, 12.0) for n in kp_names]
        kp_sizes_anim = [_kp_size_clear(n, 6.0) for n in kp_names]
        edge_colors = [
            _edge_color_clear(kp_names[a], kp_names[b]) for a, b in edges
        ]
        return {
            "clarity": "clear",
            "kp_colors": kp_colors,
            "kp_sizes_static": kp_sizes_static,
            "kp_sizes_anim": kp_sizes_anim,
            "edge_colors_static": edge_colors,
            "edge_colors_anim": edge_colors,
            "linewidth_static": 2.5,
            "linewidth_anim": 2.2,
            "cell_w_in": 3.6, "cell_h_in": 3.0,
            "label_fontsize": 11,
            "title_fontsize": 8,
        }
    raise ValueError(f"Unknown clarity={clarity!r}; expected 'compact' or 'clear'.")


# =============================================================================
# Per-cell rendering
# =============================================================================


def _draw_bout_filmstrip(
    fig,
    cell_spec,
    keypoint_traj: np.ndarray,
    edges: list[tuple[int, int]],
    style: dict,
    n_snapshots: int = 4,
    title: str = "",
) -> None:
    """Static filmstrip into one cell (PNG path).

    Splits the cell into a thin title strip + n_snapshots horizontal 3D
    panels, each centered on its own COM with a shared zoom.
    """
    T = keypoint_traj.shape[0]
    if T == 0:
        ax = fig.add_subplot(cell_spec)
        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12, color="gray")
        ax.axis("off")
        return

    # Pick snapshot frames evenly across the bout.
    frame_idx = np.linspace(0, T - 1, n_snapshots).astype(int)
    pts_all = keypoint_traj[frame_idx]
    per_extent = np.ptp(pts_all.reshape(n_snapshots, -1, 3), axis=1)
    zoom = max(float(np.median(per_extent.max(axis=1))), 1e-3) * 1.35

    # Cell layout: title strip (h=0.18) on top, n_snapshots filmstrip below.
    inner = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=cell_spec,
        height_ratios=[0.18, 1.0], hspace=0.05,
    )
    if title:
        ax_t = fig.add_subplot(inner[0])
        ax_t.text(
            0.5, 0.5, title, ha="center", va="center",
            fontsize=style["title_fontsize"],
        )
        ax_t.axis("off")

    strip = GridSpecFromSubplotSpec(
        1, n_snapshots, subplot_spec=inner[1], wspace=0.02,
    )
    # In compact mode: per-snapshot color from viridis (fade through bout).
    # In clear mode: keep per-keypoint colors stable across snapshots so the
    # head/limbs are recognizable, and only the alpha fades.
    cmap = plt.get_cmap("viridis")
    is_clear = style["clarity"] == "clear"

    for j, f in enumerate(frame_idx):
        ax = fig.add_subplot(strip[0, j], projection="3d")
        pts = keypoint_traj[f]
        com = pts.mean(axis=0)
        frac = j / max(n_snapshots - 1, 1)
        if is_clear:
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=style["kp_colors"], s=style["kp_sizes_static"],
                alpha=0.6 + 0.4 * frac,
            )
            for (ia, ib), ec in zip(edges, style["edge_colors_static"]):
                ax.plot(
                    [pts[ia, 0], pts[ib, 0]],
                    [pts[ia, 1], pts[ib, 1]],
                    [pts[ia, 2], pts[ib, 2]],
                    color=ec, linewidth=style["linewidth_static"],
                    alpha=0.5 + 0.5 * frac,
                )
        else:
            color = cmap(frac)
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=[color], s=style["kp_sizes_static"][0],
            )
            for ia, ib in edges:
                ax.plot(
                    [pts[ia, 0], pts[ib, 0]],
                    [pts[ia, 1], pts[ib, 1]],
                    [pts[ia, 2], pts[ib, 2]],
                    color=color, linewidth=style["linewidth_static"], alpha=0.95,
                )
        half = zoom / 2
        ax.set_xlim(com[0] - half, com[0] + half)
        ax.set_ylim(com[1] - half, com[1] + half)
        ax.set_zlim(com[2] - half, com[2] + half)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=15, azim=-75)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("w")


def _setup_cell_animated(
    fig,
    cell_spec,
    keypoint_traj: np.ndarray,
    edges: list[tuple[int, int]],
    style: dict,
    title: str,
) -> dict | None:
    """Set up one animated 3D skeleton in one cell (MP4 path).

    Returns a dict with handles needed to update on each frame, or None
    if the bout is empty.
    """
    T = keypoint_traj.shape[0]
    if T == 0:
        ax = fig.add_subplot(cell_spec)
        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12, color="gray")
        ax.axis("off")
        return None

    inner = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=cell_spec,
        height_ratios=[0.16, 1.0], hspace=0.05,
    )
    if title:
        ax_t = fig.add_subplot(inner[0])
        ax_t.text(
            0.5, 0.5, title, ha="center", va="center",
            fontsize=style["title_fontsize"],
        )
        ax_t.axis("off")

    ax = fig.add_subplot(inner[1], projection="3d")

    # Per-bout zoom (median frame extent, *2.2 for headroom).
    per_frame_ext = np.ptp(keypoint_traj.reshape(T, -1, 3), axis=1)
    zoom = max(float(np.median(per_frame_ext.max(axis=1))) * 2.2, 1e-2)
    coms = keypoint_traj.mean(axis=1)

    pts0 = keypoint_traj[0]
    if style["clarity"] == "clear":
        # Use scatter so we can pass per-keypoint colors and sizes.
        pts_artist = ax.scatter(
            pts0[:, 0], pts0[:, 1], pts0[:, 2],
            c=style["kp_colors"], s=style["kp_sizes_anim"],
        )
    else:
        # Original compact behaviour: plot('o', ms=3) — note `ms` is in
        # points (linear), unrelated to scatter's `s` (points²). Hard-code
        # to preserve byte-equivalence with the pre-clarity renderer.
        (pts_artist,) = ax.plot(
            pts0[:, 0], pts0[:, 1], pts0[:, 2],
            "o", color="steelblue", ms=3,
        )
    line_artists = [
        ax.plot(
            [pts0[ia, 0], pts0[ib, 0]],
            [pts0[ia, 1], pts0[ib, 1]],
            [pts0[ia, 2], pts0[ib, 2]],
            "-", color=ec, linewidth=style["linewidth_anim"],
        )[0]
        for (ia, ib), ec in zip(edges, style["edge_colors_anim"])
    ]
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=15, azim=-75)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Initial bounds.
    c0 = coms[0]
    half = zoom / 2
    ax.set_xlim(c0[0] - half, c0[0] + half)
    ax.set_ylim(c0[1] - half, c0[1] + half)
    ax.set_zlim(c0[2] - half, c0[2] + half)

    return {
        "ax": ax,
        "pts_artist": pts_artist,
        "line_artists": line_artists,
        "keypoint_traj": keypoint_traj,
        "coms": coms,
        "zoom": zoom,
        "T": T,
        "is_scatter": style["clarity"] == "clear",
    }


def _update_cell_animated(cell: dict, edges: list[tuple[int, int]], t: int) -> list:
    """Advance one animated cell to time ``t``. ``t`` saturates at the
    last frame for shorter bouts so they hold the final pose."""
    t_clipped = min(t, cell["T"] - 1)
    pts = cell["keypoint_traj"][t_clipped]
    if cell["is_scatter"]:
        # PathCollection: update offsets3d (the supported public API).
        cell["pts_artist"]._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
    else:
        cell["pts_artist"].set_data(pts[:, 0], pts[:, 1])
        cell["pts_artist"].set_3d_properties(pts[:, 2])
    for ln, (ia, ib) in zip(cell["line_artists"], edges):
        ln.set_data([pts[ia, 0], pts[ib, 0]], [pts[ia, 1], pts[ib, 1]])
        ln.set_3d_properties([pts[ia, 2], pts[ib, 2]])
    c = cell["coms"][t_clipped]
    half = cell["zoom"] / 2
    cell["ax"].set_xlim(c[0] - half, c[0] + half)
    cell["ax"].set_ylim(c[1] - half, c[1] + half)
    cell["ax"].set_zlim(c[2] - half, c[2] + half)
    return [cell["pts_artist"], *cell["line_artists"]]


# =============================================================================
# Page rendering (PNG + MP4)
# =============================================================================


def _make_page_figure(
    n_rows: int, n_examples: int, fit_label: str, style: dict,
):
    """Build the outer figure + GridSpec used by both PNG and MP4 paths."""
    cw, ch = style["cell_w_in"], style["cell_h_in"]
    fig = plt.figure(figsize=(cw * n_examples + 1.6, ch * n_rows + 1.0))
    outer = GridSpec(
        n_rows, n_examples + 1,
        width_ratios=[0.20] + [1.0] * n_examples,
        hspace=0.45, wspace=0.05, figure=fig,
        left=0.02, right=0.99, top=0.92, bottom=0.04,
    )
    fig.suptitle(fit_label, fontsize=11)
    return fig, outer


def _draw_row_label(
    fig, cell_spec, syllable_id: int, n_bouts: int, style: dict,
) -> None:
    ax = fig.add_subplot(cell_spec)
    ax.text(
        0.5, 0.5, f"Syll {syllable_id}\n({n_bouts} bouts)",
        ha="center", va="center",
        fontsize=style["label_fontsize"], fontweight="bold",
    )
    ax.axis("off")


def render_page_png(
    fit_label: str,
    syllables_on_page: list[int],
    bouts_per_syll: dict[int, list[Bout]],
    keypoints: np.ndarray,
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    output_path: Path,
    n_examples: int = 5,
    n_snapshots: int = 4,
    clarity: str = "compact",
) -> None:
    """Render one PNG page: rows = syllables_on_page, cols = examples."""
    edges = _resolve_skeleton_edges(kp_names, skeleton)
    style = _resolve_styling(clarity, kp_names, edges)
    n_rows = len(syllables_on_page)
    fig, outer = _make_page_figure(n_rows, n_examples, fit_label, style)

    for row_i, sid in enumerate(syllables_on_page):
        bouts = bouts_per_syll.get(sid, [])
        _draw_row_label(fig, outer[row_i, 0], sid, len(bouts), style)
        for col_i in range(n_examples):
            cell_spec = outer[row_i, col_i + 1]
            if col_i >= len(bouts):
                ax_e = fig.add_subplot(cell_spec)
                ax_e.text(0.5, 0.5, "—", ha="center", va="center",
                          fontsize=12, color="gray")
                ax_e.axis("off")
                continue
            bout = bouts[col_i]
            kp_traj = bout.keypoint_slice(keypoints)
            title = (
                f"clip {bout.clip_id} | "
                f"t={bout.start_frame}-{bout.end_frame} | "
                f"dur={bout.duration}f"
            )
            _draw_bout_filmstrip(
                fig, cell_spec, kp_traj, edges, style,
                n_snapshots=n_snapshots, title=title,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140 if clarity == "clear" else 120,
                bbox_inches="tight")
    plt.close(fig)
    log.info(f"  PNG: {output_path}")


def render_page_mp4(
    fit_label: str,
    syllables_on_page: list[int],
    bouts_per_syll: dict[int, list[Bout]],
    keypoints: np.ndarray,
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    output_path: Path,
    n_examples: int = 5,
    fps: int = 30,
    clarity: str = "compact",
) -> None:
    """Render one MP4 page: rows = syllables_on_page, cols = examples,
    animated in lockstep."""
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    edges = _resolve_skeleton_edges(kp_names, skeleton)
    style = _resolve_styling(clarity, kp_names, edges)
    n_rows = len(syllables_on_page)
    fig, outer = _make_page_figure(n_rows, n_examples, fit_label, style)

    cells: dict[tuple[int, int], dict] = {}
    max_T = 1
    for row_i, sid in enumerate(syllables_on_page):
        bouts = bouts_per_syll.get(sid, [])
        _draw_row_label(fig, outer[row_i, 0], sid, len(bouts), style)
        for col_i in range(n_examples):
            cell_spec = outer[row_i, col_i + 1]
            if col_i >= len(bouts):
                ax_e = fig.add_subplot(cell_spec)
                ax_e.text(0.5, 0.5, "—", ha="center", va="center",
                          fontsize=12, color="gray")
                ax_e.axis("off")
                continue
            bout = bouts[col_i]
            kp_traj = bout.keypoint_slice(keypoints)
            title = (
                f"clip {bout.clip_id} | "
                f"t={bout.start_frame}-{bout.end_frame} | "
                f"dur={bout.duration}f"
            )
            cell = _setup_cell_animated(
                fig, cell_spec, kp_traj, edges, style, title,
            )
            if cell is not None:
                cells[(row_i, col_i)] = cell
                max_T = max(max_T, cell["T"])

    if not cells:
        log.warning(f"  MP4: nothing to animate; skipping {output_path}")
        plt.close(fig)
        return

    def update(t: int):
        artists: list = []
        for cell in cells.values():
            artists.extend(_update_cell_animated(cell, edges, t))
        return artists

    ani = FuncAnimation(fig, update, frames=max_T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    log.info(f"  MP4: {output_path} ({max_T} frames @ {fps} fps)")


# =============================================================================
# Top-level orchestrator
# =============================================================================


def render_for_fit(
    fit_rank: int,
    fit_meta: dict,
    syllables_per_clip: list[np.ndarray],
    n_states: int,
    keypoints: np.ndarray,
    kp_names: list[str],
    skeleton: list[tuple[str, str]],
    all_clip_indices: np.ndarray,
    output_dir: Path,
    n_examples: int = 5,
    max_rows_per_page: int = 5,
    bout_strategy: str = "longest",
    min_bout_duration: int = 3,
    output_format: str = "png",
    fps: int = 30,
    clarity: str = "compact",
) -> None:
    """Render syllable-grids for one fit. Paginates if n_states > max_rows.

    Args:
        clarity: ``"compact"`` (default, original style — small cells,
            uniform color) or ``"clear"`` (larger cells with body-part
            coloring; recommended for inspection of merged labels).
    """
    setting = fit_meta.get("setting", "?")
    seed = fit_meta.get("seed", "?")
    fit_label_base = (
        f"M{fit_rank} ({setting}, seed={seed})  —  "
        f"top-{n_examples} {bout_strategy} bouts per syllable"
    )

    # Collect all bouts for every syllable in this fit.
    bouts_per_syll: dict[int, list[Bout]] = {}
    for sid in range(n_states):
        all_bouts = collect_bouts(
            syllables_per_clip, all_clip_indices, sid,
            min_duration=min_bout_duration,
        )
        bouts_per_syll[sid] = select_bouts(all_bouts, n=n_examples, strategy=bout_strategy)

    n_pages = (n_states + max_rows_per_page - 1) // max_rows_per_page
    log.info(
        f"M{fit_rank}: {n_states} syllables → {n_pages} page(s) of "
        f"{max_rows_per_page} rows × {n_examples} examples"
    )

    for page in range(n_pages):
        lo = page * max_rows_per_page
        hi = min(lo + max_rows_per_page, n_states)
        syllables_on_page = list(range(lo, hi))
        page_label = f"{fit_label_base}  —  page {page + 1}/{n_pages}, syllables {lo}–{hi - 1}"
        stem = f"syllable_grid_M{fit_rank}_p{page + 1}_syll{lo}-{hi - 1}"

        if output_format in ("png", "both"):
            render_page_png(
                fit_label=page_label,
                syllables_on_page=syllables_on_page,
                bouts_per_syll=bouts_per_syll,
                keypoints=keypoints, kp_names=kp_names, skeleton=skeleton,
                output_path=output_dir / f"{stem}.png",
                n_examples=n_examples, clarity=clarity,
            )
        if output_format in ("video", "both"):
            render_page_mp4(
                fit_label=page_label,
                syllables_on_page=syllables_on_page,
                bouts_per_syll=bouts_per_syll,
                keypoints=keypoints, kp_names=kp_names, skeleton=skeleton,
                output_path=output_dir / f"{stem}.mp4",
                n_examples=n_examples, fps=fps, clarity=clarity,
            )


def render(
    sweep_dir: Path,
    rank: str = "mse",
    top_k: int = 5,
    n_examples: int = 5,
    max_rows_per_page: int = 5,
    bout_strategy: str = "longest",
    min_bout_duration: int = 3,
    output_format: str = "both",
    fps: int = 30,
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    clarity: str = "compact",
) -> None:
    """End-to-end orchestrator. Mirrors arg flow of render_topk_bands.render."""
    sweep_dir = Path(sweep_dir).resolve()
    results_path = sweep_dir / "sweep_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"sweep_results.json not at {results_path}")
    with results_path.open() as f:
        results = json.load(f)

    top = _select_topk(results["all_results"], rank=rank, top_k=top_k)
    if not top:
        raise RuntimeError("No valid fits in sweep_results.json — cannot render.")
    log.info(f"Rendering syllable grids for top-{len(top)} fits (by {rank}):")

    repo_root = get_repo_root()
    if balanced_split_path is None:
        balanced_split_path = repo_root / "data" / "rodent" / "rodent_balanced_splits.json"
    if reference_data_path is None:
        reference_data_path = repo_root / "data" / "rodent" / "rodent_reference_clips.h5"
    if stac_xml_path is None:
        stac_xml_path = repo_root / "c2a" / "assets" / "rodent.xml"

    all_clip_indices = _load_balanced_split(balanced_split_path)

    log.info("Loading keypoints via FK (one-time; ~30s)...")
    keypoints, kp_names = _load_keypoints(
        str(reference_data_path), str(stac_xml_path), str(balanced_split_path),
    )
    log.info(f"Keypoints: {keypoints.shape}")

    output_dir = sweep_dir / "figures" / "syllable_grids"
    output_dir.mkdir(parents=True, exist_ok=True)

    for k, r in enumerate(top, start=1):
        project_dir = Path(r["project_dir"])
        if not project_dir.is_absolute():
            project_dir = repo_root / project_dir
        try:
            sylls = _load_syllables_for_fit(project_dir, r["model_name"])
        except FileNotFoundError as e:
            log.warning(f"M{k}: skipping — {e}")
            continue
        if not sylls:
            log.warning(f"M{k}: no syllable arrays")
            continue

        # n_states: prefer explicit, fall back to data-derived max + 1.
        if "n_states" in r:
            n_states = int(r["n_states"])
        else:
            n_states = int(max(int(s.max()) for s in sylls)) + 1

        render_for_fit(
            fit_rank=k, fit_meta=r,
            syllables_per_clip=sylls, n_states=n_states,
            keypoints=keypoints, kp_names=kp_names, skeleton=RODENT_SKELETON,
            all_clip_indices=all_clip_indices,
            output_dir=output_dir,
            n_examples=n_examples,
            max_rows_per_page=max_rows_per_page,
            bout_strategy=bout_strategy,
            min_bout_duration=min_bout_duration,
            output_format=output_format,
            fps=fps,
            clarity=clarity,
        )

    log.info(f"Done. Outputs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--rank", choices=("mse", "nmi"), default="mse")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--max-rows-per-page", type=int, default=5)
    parser.add_argument(
        "--bout-strategy", choices=("longest", "median", "random"),
        default="longest",
    )
    parser.add_argument("--min-bout-duration", type=int, default=3)
    parser.add_argument(
        "--format", dest="output_format",
        choices=("png", "video", "both"), default="both",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--balanced-split", type=Path, default=None)
    parser.add_argument("--reference-data", type=Path, default=None)
    parser.add_argument("--stac-xml", type=Path, default=None)
    parser.add_argument(
        "--clarity", choices=("compact", "clear"), default="compact",
        help="Layout: 'compact' = original style, 'clear' = larger cells "
             "with body-part coloring (recommended for inspection).",
    )
    args = parser.parse_args()

    render(
        sweep_dir=args.sweep_dir,
        rank=args.rank,
        top_k=args.top_k,
        n_examples=args.n_examples,
        max_rows_per_page=args.max_rows_per_page,
        bout_strategy=args.bout_strategy,
        min_bout_duration=args.min_bout_duration,
        output_format=args.output_format,
        fps=args.fps,
        balanced_split_path=args.balanced_split,
        reference_data_path=args.reference_data,
        stac_xml_path=args.stac_xml,
        clarity=args.clarity,
    )


if __name__ == "__main__":
    main()
