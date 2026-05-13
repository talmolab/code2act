"""Top-N product-code "movement" grid renderer shared across variants.

Produces a single MP4 per variant with a ``grid_rows × grid_cols`` grid
(default 4×5 = 20 cells). Each cell holds **the top-K most-used code**
at rank ``(slot + 1)``, animated as ``M`` overlaid rodent skeletons —
one per long contiguous bout of that code. Bouts are COM-anchored to
their first frame so all ``M`` rodents start at the cell's local
origin and motion reads as displacement from there. Three distinct
colors (tab10 blue / orange / green by default) distinguish the M
samples per cell.

A coherent code makes the M overlaid rodents move together; an
incoherent code makes them diverge. That's the intended diagnostic.

The reader is **variant-agnostic** — it works on any
``best_codes.npz`` written by the kinematics-VAE family (VAE / VQVAE
/ PS-VQVAE), since all three share
:data:`c2a.codegen.shared.npz_schema.CodesPayload`. For PS-VQVAE, the
displayed code is the *folded* product index — matching the band
renderer's default and the rest of the pipeline.

Standalone CLI not provided — call from a per-variant render script
(see ``c2a/codegen/kinematics_vae/<variant>/render_topk_bands.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from c2a.codegen.kpms.sweep_impl import RODENT_SKELETON, _load_keypoints
from c2a.paths import get_repo_root
from c2a.render.topk_bands_render import _resolve_skeleton_edges

# Same trick as c2a.render.topk_bands_render: matplotlib's animation
# system shells out to ``ffmpeg``, which isn't on PATH in this env;
# point it at the bundled binary inside imageio_ffmpeg.
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

log = logging.getLogger(__name__)


# Three distinguishable colors for the 3 overlaid samples per cell.
DEFAULT_SAMPLE_COLORS: tuple[str, ...] = (
    "#1f77b4",   # tab10 blue
    "#ff7f0e",   # tab10 orange
    "#2ca02c",   # tab10 green
)


def _topn_bouts_for_code(
    all_codes: np.ndarray, code: int, n: int = 3,
) -> list[tuple[int, int, int]]:
    """Up to ``n`` longest contiguous bouts of ``code`` across rows of
    ``all_codes``, preferring different source clips for diversity.

    Greedy strategy: take the longest bout from any clip; then the
    longest bout from a NEW clip; etc. If fewer than ``n`` distinct
    clips contain the code, fall back to the next-longest bouts even
    from already-used clips.

    Returns a list of ``(clip_pos, start, end_exclusive)``
    ordered longest-first. Empty if the code never appears.
    """
    bouts: list[tuple[int, int, int, int]] = []   # (length, clip_pos, s, e)
    for ci in range(all_codes.shape[0]):
        seq = all_codes[ci]
        mask = (seq == code).astype(np.int8)
        if not mask.any():
            continue
        diffs = np.diff(np.concatenate(([0], mask, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            bouts.append((int(e - s), int(ci), int(s), int(e)))

    bouts.sort(reverse=True)
    picked: list[tuple[int, int, int]] = []
    used_clips: set[int] = set()

    for _L, ci, s, e in bouts:
        if ci not in used_clips:
            picked.append((ci, s, e))
            used_clips.add(ci)
            if len(picked) == n:
                return picked

    if len(picked) < n:
        already = {(ci, s) for ci, s, _ in picked}
        for _L, ci, s, e in bouts:
            if (ci, s) in already:
                continue
            picked.append((ci, s, e))
            if len(picked) == n:
                break
    return picked


def _load_bal_clip_indices(codes_npz_extras: dict) -> np.ndarray:
    train_idx = np.asarray(codes_npz_extras["train_indices"], dtype=np.int32)
    test_idx = np.asarray(codes_npz_extras["test_indices"], dtype=np.int32)
    return np.array(
        sorted(set(train_idx.tolist()) | set(test_idx.tolist())),
        dtype=np.int32,
    )


def render_top_codes_movement(
    *,
    codes_npz_path: Path,
    output_dir: Path,
    variant_label: str,
    n_top: int = 20,
    n_samples_per_code: int = 3,
    grid_rows: int = 4,
    grid_cols: int = 5,
    fps: int = 15,
    sample_colors: Sequence[str] = DEFAULT_SAMPLE_COLORS,
    keypoints: Optional[np.ndarray] = None,
    kp_names: Optional[list[str]] = None,
    edges: Optional[list[tuple[int, int]]] = None,
    reference_data_path: Optional[Path] = None,
    stac_xml_path: Optional[Path] = None,
    balanced_split_path: Optional[Path] = None,
    output_filename: Optional[str] = None,
) -> Path:
    """Render the ``grid_rows × grid_cols`` top-code movement video.

    Either pre-loaded ``keypoints`` + ``edges`` are passed in (the
    common case when the caller has already done forward kinematics
    for the band renderer), OR ``reference_data_path`` /
    ``stac_xml_path`` / ``balanced_split_path`` are given and FK is
    run internally.

    Args:
        codes_npz_path: Path to the variant's ``best_codes.npz``.
        output_dir: Where to write the MP4.
        variant_label: short tag for figure suptitle (e.g. ``"VAE"``).
        n_top: number of top codes to render (must equal
            ``grid_rows * grid_cols``).
        n_samples_per_code: number of overlaid bouts per cell. Must be
            ``<= len(sample_colors)``.
        grid_rows, grid_cols: figure layout.
        fps: video framerate. 15 fps matches the H5 mocap rate.
        sample_colors: hex colors for the per-cell overlaid samples.
        keypoints / kp_names / edges: if pre-loaded (skips FK).
        reference_data_path / stac_xml_path / balanced_split_path:
            paths for FK if ``keypoints`` is not provided.
        output_filename: override the default ``top{N}_codes_movement.mp4``.

    Returns the output MP4 path.
    """
    if grid_rows * grid_cols != n_top:
        raise ValueError(
            f"grid_rows * grid_cols ({grid_rows * grid_cols}) must equal "
            f"n_top ({n_top})."
        )
    if n_samples_per_code > len(sample_colors):
        raise ValueError(
            f"n_samples_per_code={n_samples_per_code} exceeds "
            f"len(sample_colors)={len(sample_colors)}. Pass more colors."
        )

    if keypoints is None:
        repo = get_repo_root()
        if reference_data_path is None:
            reference_data_path = repo / "data" / "rodent" / "rodent_reference_clips.h5"
        if stac_xml_path is None:
            stac_xml_path = repo / "c2a" / "assets" / "rodent.xml"
        if balanced_split_path is None:
            balanced_split_path = repo / "data" / "rodent" / "rodent_balanced_splits.json"
        log.info("render_top_codes_movement: loading keypoints via FK (~30s)...")
        keypoints, kp_names = _load_keypoints(
            str(reference_data_path), str(stac_xml_path),
            str(balanced_split_path),
        )
    if edges is None:
        if kp_names is None:
            raise ValueError("Pass kp_names if edges not pre-computed.")
        edges = _resolve_skeleton_edges(kp_names, RODENT_SKELETON)

    codes_npz_path = Path(codes_npz_path)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if not codes_npz_path.exists():
        raise FileNotFoundError(f"best_codes.npz not at {codes_npz_path}")

    meta_path = codes_npz_path.parent / "tokenize_meta.json"
    meta: dict = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {}
    )

    with np.load(codes_npz_path, allow_pickle=True) as data:
        extras = {k: np.asarray(data[k]) for k in data.files}
    all_codes = extras["all_codes"]                       # (N_bal, T)
    bal_idx = _load_bal_clip_indices(extras)
    K = int(extras["num_states"])

    if K < n_top:
        log.warning(
            f"{variant_label}: K={K} < n_top={n_top}; cropping to top {K}.",
        )

    counts = np.bincount(all_codes.ravel().astype(np.int64), minlength=K)
    top = np.argsort(-counts)[:n_top]
    log.info(
        f"{variant_label}: K={K}; top-{len(top)} codes (counts): "
        f"{[(int(c), int(counts[c])) for c in top]}"
    )

    cells: list[dict] = []
    for code in top:
        bouts = _topn_bouts_for_code(all_codes, int(code), n=n_samples_per_code)
        if not bouts:
            log.warning(f"  code {code} never appears; skipping cell.")
            continue
        cells.append({"code": int(code), "bouts": bouts})
    if not cells:
        raise RuntimeError(
            f"{variant_label}: no codes had any contiguous bouts."
        )

    # Per-cell: list of (L, K, 3) COM-anchored trajectories + zoom + len.
    cell_traj: list[list[np.ndarray]] = []
    cell_zoom: list[float] = []
    cell_max_len: list[int] = []
    for cell in cells:
        anchored = []
        radii = []
        for ci, s, e in cell["bouts"]:
            traj = keypoints[ci, s:e]                          # (L, K, 3)
            anchor = traj[0].mean(axis=0)
            traj_a = traj - anchor
            anchored.append(traj_a)
            radii.append(float(
                np.linalg.norm(traj_a.reshape(-1, 3), axis=1).max()
            ))
        cell_traj.append(anchored)
        cell_zoom.append(max(radii) * 1.25 + 0.05)
        cell_max_len.append(max(t.shape[0] for t in anchored))

    T_max = max(cell_max_len)
    log.info(
        f"  per-cell longest bouts: {cell_max_len}; video len = {T_max}"
    )

    fig = plt.figure(figsize=(22, 16))
    fig.subplots_adjust(
        left=0.02, right=0.99, top=0.92, bottom=0.02,
        wspace=0.05, hspace=0.50,
    )

    per_cell_artists: list[list[tuple]] = []
    for slot, cell in enumerate(cells):
        ax = fig.add_subplot(grid_rows, grid_cols, slot + 1, projection="3d")
        ax.view_init(elev=15, azim=-75)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("w")

        half = cell_zoom[slot]
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)

        code = cell["code"]
        bout_lens = [b[2] - b[1] for b in cell["bouts"]]
        clip_ids = [int(bal_idx[b[0]]) for b in cell["bouts"]]
        title = (
            f"#{slot + 1}  code {code}  (count {int(counts[code])})\n"
            f"bouts {'/'.join(str(L) for L in bout_lens)}f"
            f"  •  clips {'/'.join(str(c) for c in clip_ids)}"
        )
        ax.set_title(title, fontsize=8.5, pad=4)

        cell_arts = []
        for k, traj in enumerate(cell_traj[slot]):
            color = sample_colors[k % len(sample_colors)]
            pts = traj[0]
            sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, s=6)
            line_handles = []
            for ia, ib in edges:
                (ln,) = ax.plot(
                    [pts[ia, 0], pts[ib, 0]],
                    [pts[ia, 1], pts[ib, 1]],
                    [pts[ia, 2], pts[ib, 2]],
                    color=color, linewidth=1.3, alpha=0.85,
                )
                line_handles.append(ln)
            cell_arts.append((sc, line_handles))
        per_cell_artists.append(cell_arts)

    suptitle_parts = [
        f"{variant_label}",
        f"top-{len(cells)} codes • {n_samples_per_code} overlaid bouts/cell",
    ]
    if meta.get("group_codebook_sizes"):
        suptitle_parts.append(
            f"K_groups={meta['group_codebook_sizes']} • product K={K}"
        )
    else:
        suptitle_parts.append(f"K={K}")
    sup_artist = fig.suptitle(
        "  •  ".join(suptitle_parts) + f"  •  frame 0/{T_max - 1}",
        fontsize=14, y=0.965,
    )

    def update(t: int):
        flat = []
        for slot in range(len(cells)):
            for k, traj in enumerate(cell_traj[slot]):
                L = traj.shape[0]
                f = min(t, L - 1)
                pts = traj[f]
                sc, lines = per_cell_artists[slot][k]
                sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
                for (ia, ib), ln in zip(edges, lines):
                    ln.set_data_3d(
                        [pts[ia, 0], pts[ib, 0]],
                        [pts[ia, 1], pts[ib, 1]],
                        [pts[ia, 2], pts[ib, 2]],
                    )
                flat.append(sc)
                flat.extend(lines)
        sup_artist.set_text(
            "  •  ".join(suptitle_parts) + f"  •  frame {t}/{T_max - 1}"
        )
        flat.append(sup_artist)
        return flat

    ani = FuncAnimation(
        fig, update, frames=T_max, interval=1000 / fps, blit=False,
    )
    writer = FFMpegWriter(fps=fps, bitrate=6400)

    if output_filename is None:
        output_filename = f"top{n_top}_codes_movement.mp4"
    out_path = output_dir / output_filename
    ani.save(out_path, writer=writer)
    plt.close(fig)
    log.info(f"  Wrote {out_path}")
    return out_path
