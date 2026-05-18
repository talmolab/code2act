"""Helpers for the code-grid experiments.

Reused by:
- ``run_code_grid_rollouts`` (rollouts + raw/merged grid videos)
- ``run_code_grid_dynamics`` (DSA + rSLDS on merged-cluster hidden trajectories)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio
import matplotlib.cm as mpl_cm
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from c2a.rendering_utils import build_ghost_model, _disable_lights_recursive

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code colors
# ---------------------------------------------------------------------------


def make_code_colors(num_codes: int) -> np.ndarray:
    """Return ``(num_codes, 4)`` RGBA palette distinct across codes.

    Uses HSV evenly distributed in hue for visual distinguishability. The
    same color is reused everywhere a given code appears so the visual
    identity of a code is consistent across raw and merged grids.
    """
    hues = (np.arange(num_codes) / max(num_codes, 1)) % 1.0
    # Mix in fixed saturation + value for vibrant but readable colors.
    import colorsys

    rgba = np.zeros((num_codes, 4), dtype=np.float32)
    for i, h in enumerate(hues):
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        rgba[i] = [r, g, b, 1.0]
    return rgba


# ---------------------------------------------------------------------------
# Code usage ranking
# ---------------------------------------------------------------------------


def rank_codes_by_usage(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    num_codes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ranked_code_ids, usage_counts)``.

    ``ranked_code_ids[0]`` is the most-used code; ``usage_counts[c]`` is the
    raw count for code id ``c`` (NOT for the ranked position).
    """
    all_codes = np.concatenate(
        [train_codes.flatten(), test_codes.flatten()], axis=0,
    )
    all_codes = all_codes[(all_codes >= 0) & (all_codes < num_codes)]
    counts = np.bincount(all_codes.astype(np.int64), minlength=num_codes)
    ranked = np.argsort(-counts, kind="stable")
    return ranked, counts


# ---------------------------------------------------------------------------
# MMD between empirical trajectories
# ---------------------------------------------------------------------------


def _pool_qpos(
    qpos_per_code: dict[int, np.ndarray],
    strip_root_pos: bool,
) -> dict[int, np.ndarray]:
    """Flatten each (K, T, nq) array into (K*T, feat). Optional root strip.

    Root-pos strip drops the first 3 qpos dims (world translation). World
    rotation (quat dims 3..7) is kept because it's identity-invariant w.r.t.
    body posture, not w.r.t. orientation — but for behavior comparison we
    keep it as a feature.
    """
    out: dict[int, np.ndarray] = {}
    for c, arr in qpos_per_code.items():
        # arr: (K, T, nq)
        K, T, nq = arr.shape
        flat = arr.reshape(K * T, nq)
        if strip_root_pos and nq > 3:
            flat = flat[:, 3:]
        out[c] = flat.astype(np.float32)
    return out


def _median_pairwise(samples: np.ndarray, max_pairs: int = 5000) -> float:
    """Median pairwise Euclidean distance over a random subset of pairs."""
    n = samples.shape[0]
    if n < 2:
        return 1.0
    rng = np.random.RandomState(0)
    i = rng.randint(0, n, size=max_pairs)
    j = rng.randint(0, n, size=max_pairs)
    mask = i != j
    d = np.linalg.norm(samples[i[mask]] - samples[j[mask]], axis=1)
    if d.size == 0:
        return 1.0
    return float(np.median(d))


def _gauss_kernel_matrix(
    X: np.ndarray, Y: np.ndarray, sigma: float,
) -> np.ndarray:
    """K[i, j] = exp(-||X_i - Y_j||² / (2 σ²))."""
    # Compute squared distance matrix without forming the full diff tensor.
    XX = (X * X).sum(axis=1, keepdims=True)         # (n, 1)
    YY = (Y * Y).sum(axis=1, keepdims=True).T       # (1, m)
    sq = XX + YY - 2.0 * X @ Y.T
    np.maximum(sq, 0.0, out=sq)
    return np.exp(-sq / (2.0 * sigma * sigma))


def mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD² with Gaussian kernel.

    For small n, falls back to biased estimator (n^-2 normalization) — the
    unbiased estimator divides by n*(n-1) and is undefined at n=1.
    """
    n, m = X.shape[0], Y.shape[0]
    Kxx = _gauss_kernel_matrix(X, X, sigma)
    Kyy = _gauss_kernel_matrix(Y, Y, sigma)
    Kxy = _gauss_kernel_matrix(X, Y, sigma)
    if n > 1 and m > 1:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        return float(
            Kxx.sum() / (n * (n - 1))
            + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean()
        )
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def compute_mmd_distance_matrix(
    qpos_per_code: dict[int, np.ndarray],
    strip_root_pos: bool = True,
    kernel_sigma: str | float = "median",
    subsample_per_code: int | None = 1000,
) -> tuple[np.ndarray, list[int], float]:
    """Pairwise MMD distance matrix across codes.

    Args:
        qpos_per_code: ``{code_id: (K, T, nq) float array}``.
        strip_root_pos: Drop first 3 qpos dims (world translation).
        kernel_sigma: ``"median"`` (median heuristic over pooled samples)
            or a positive float.
        subsample_per_code: If not None, take a random subsample of this
            size per code to bound MMD compute. ``None`` = use all
            ``K * T`` per code.

    Returns:
        ``(D, code_ids, sigma_used)``: D is ``(N, N)`` with
        ``D[i, j] = sqrt(max(0, mmd²(code_i, code_j)))``; ``code_ids[i]``
        is the original code id for row/col i (stable across runs).
    """
    pooled = _pool_qpos(qpos_per_code, strip_root_pos)
    code_ids = sorted(pooled.keys())
    arrays = [pooled[c] for c in code_ids]

    if subsample_per_code is not None:
        rng = np.random.RandomState(0)
        arrays = [
            a if a.shape[0] <= subsample_per_code
            else a[rng.choice(a.shape[0], subsample_per_code, replace=False)]
            for a in arrays
        ]

    # Sigma from pooled-across-codes samples for cross-code comparability.
    pooled_all = np.concatenate(arrays, axis=0)
    if isinstance(kernel_sigma, str) and kernel_sigma == "median":
        sigma = _median_pairwise(pooled_all)
    else:
        sigma = float(kernel_sigma)
    if sigma <= 0:
        sigma = 1.0

    N = len(arrays)
    D2 = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            m2 = mmd_squared(arrays[i], arrays[j], sigma)
            D2[i, j] = m2
            D2[j, i] = m2
    D = np.sqrt(np.maximum(D2, 0.0))
    return D, code_ids, sigma


# ---------------------------------------------------------------------------
# Hierarchical clustering of codes
# ---------------------------------------------------------------------------


def _elbow_cutoff(merge_heights: np.ndarray) -> float:
    """Pick a cutoff height at the elbow of sorted merge heights.

    Sort the merge heights, take the second-derivative proxy (curvature
    of the consecutive-difference sequence), and pick the height at the
    sharpest knee. Falls back to median if the curve is too flat.
    """
    heights = np.sort(merge_heights)
    if heights.size < 3:
        return float(np.median(heights))
    # Knee = point of max curvature in the sorted-heights curve. Compute
    # the line from first to last point and find the point with max
    # perpendicular distance to it.
    p1 = np.array([0.0, heights[0]])
    p2 = np.array([float(len(heights) - 1), heights[-1]])
    line = p2 - p1
    line_len = np.linalg.norm(line)
    if line_len < 1e-12:
        return float(np.median(heights))
    points = np.stack(
        [np.arange(len(heights)), heights], axis=1,
    )
    diffs = points - p1
    # Perpendicular distance from each point to the (p1, p2) line.
    cross = np.abs(diffs[:, 0] * line[1] - diffs[:, 1] * line[0])
    perp = cross / line_len
    knee_idx = int(np.argmax(perp))
    return float(heights[knee_idx])


def cluster_codes(
    dist_matrix: np.ndarray,
    code_ids: list[int],
    cutoff: str = "elbow",
    fixed_k: int = 10,
    frac_median: float = 0.5,
) -> tuple[list[list[int]], dict[str, Any]]:
    """UPGMA hierarchical clustering of codes by distance.

    Returns:
        ``(clusters, info)``: ``clusters`` is a list of cluster code-id
        lists in original code id (not row-index) form, sorted by size
        descending; ``info`` carries Z, cutoff_used, and per-code cluster
        label.
    """
    N = dist_matrix.shape[0]
    if N < 2:
        return [[code_ids[0]]] if N == 1 else [], {
            "Z": np.zeros((0, 4)), "cutoff_used": 0.0,
            "labels": np.zeros(N, dtype=np.int32),
        }

    # condensed distance form for linkage
    cond = squareform(dist_matrix, checks=False)
    Z = hierarchy.linkage(cond, method="average")  # UPGMA

    if cutoff == "elbow":
        cutoff_used = _elbow_cutoff(Z[:, 2])
        labels = hierarchy.fcluster(Z, t=cutoff_used, criterion="distance")
    elif cutoff == "frac_median":
        med = float(np.median(dist_matrix[np.triu_indices(N, k=1)]))
        cutoff_used = frac_median * med
        labels = hierarchy.fcluster(Z, t=cutoff_used, criterion="distance")
    elif cutoff == "fixed_k":
        cutoff_used = float("nan")
        labels = hierarchy.fcluster(Z, t=fixed_k, criterion="maxclust")
    else:
        raise ValueError(f"Unknown cutoff strategy: {cutoff!r}")

    clusters: dict[int, list[int]] = {}
    for row, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(int(code_ids[row]))
    cluster_list = sorted(clusters.values(), key=lambda x: -len(x))
    return cluster_list, {
        "Z": Z,
        "cutoff_used": cutoff_used,
        "labels": np.asarray(labels, dtype=np.int32),
        "code_ids": list(code_ids),
        "strategy": cutoff,
    }


# ---------------------------------------------------------------------------
# Panel rendering (ghost model, K bodies per panel)
# ---------------------------------------------------------------------------


def recolor_base_body(mj_model: Any, rgba: list[float] | np.ndarray) -> None:
    """Recolor all geoms of the base (non-ghost) walker body in-place.

    ``build_ghost_model`` adds N ghost bodies on top of the base walker
    (``env._spec``). The base walker keeps the rgba from its XML, which
    makes one walker visually distinct from the K-1 ghosts. This helper
    overwrites the base body's geom rgba to match the ghosts.

    Ghost bodies are identified by the ``-ghost{i}`` suffix that
    ``build_ghost_model`` adds via ``frame.attach_body(..., suffix=...)``.
    Any geom whose owning body name does NOT contain ``-ghost`` is
    considered base.
    """
    rgba = np.asarray(rgba, dtype=np.float32)
    if rgba.shape != (4,):
        raise ValueError(f"rgba must be (4,), got {rgba.shape}")
    for gi in range(mj_model.ngeom):
        body_id = int(mj_model.geom_bodyid[gi])
        body_name = mj_model.body(body_id).name
        if "-ghost" not in body_name:
            mj_model.geom_rgba[gi] = rgba


def build_uniform_walkers(
    env: Any,
    walker_colors: list[list[float]],
    camera_distance: float = 0.8,
    camera_elevation: float = -30.0,
    camera_azimuth: float = 135.0,
) -> tuple[Any, int]:
    """Build a ghost model whose total walker count equals ``len(walker_colors)``.

    ``walker_colors[0]`` is applied to the base body (via post-compile
    geom recolor); ``walker_colors[1:]`` are baked into the ghost specs
    by ``build_ghost_model``. This avoids:

    * the "one walker has the original XML color" issue (base body not
      recolored), and
    * the "one walker stays at origin" issue when caller asks for K
      ghosts plus the K trajectories include the base.
    """
    n = len(walker_colors)
    if n < 1:
        raise ValueError("walker_colors must have >= 1 entry")
    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=n - 1,
        ghost_colors=[list(c) for c in walker_colors[1:]] or [[0, 0, 0, 0]],
        camera_distance=camera_distance,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
    )
    recolor_base_body(ghost_model, walker_colors[0])
    return ghost_model, base_nq


def render_panel_frames(
    ghost_model: Any,
    base_nq: int,
    trajectories_qpos: list[np.ndarray],
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Render T frames showing K bodies overlaid via the ghost model.

    Uses the default MuJoCo arena (skybox + floor) — the same look as
    training-eval rollouts. Only the walker bodies are recolored (by
    ``build_uniform_walkers``); background and floor stay default.

    Returns ``frames`` of shape ``(T, height, width, 3)`` uint8.
    """
    K = len(trajectories_qpos)
    T = min(len(q) for q in trajectories_qpos)
    if K == 0 or T == 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)

    ghost_model.vis.global_.offwidth = width
    ghost_model.vis.global_.offheight = height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=height, width=width)
    try:
        frames = np.zeros((T, height, width, 3), dtype=np.uint8)
        for t in range(T):
            data.qpos[:base_nq] = trajectories_qpos[0][t]
            for gi in range(1, K):
                q_start = base_nq + (gi - 1) * base_nq
                q_end = q_start + base_nq
                data.qpos[q_start:q_end] = trajectories_qpos[gi][t]
            mujoco.mj_forward(ghost_model, data)
            renderer.update_scene(data, camera=camera_name)
            frames[t] = renderer.render()
    finally:
        renderer.close()
    return frames


def add_banner(
    frame: np.ndarray,
    text: str,
    banner_h: int = 28,
    bg: tuple[int, int, int] = (245, 245, 245),
    fg: tuple[int, int, int] = (20, 20, 20),
    font_size: int = 14,
) -> np.ndarray:
    """Prepend an opaque banner above an RGB frame.

    Returns a new frame of shape ``(H + banner_h, W, 3)``.
    """
    H, W = frame.shape[:2]
    out = np.zeros((H + banner_h, W, 3), dtype=np.uint8)
    out[:banner_h] = np.array(bg, dtype=np.uint8)[None, None]
    out[banner_h:] = frame
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size,
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, max(0, (banner_h - font_size) // 2 - 1)), text, fill=fg, font=font)
    return np.asarray(img)


def stack_panels_into_grid(
    panel_frames: list[np.ndarray],
    rows: int,
    cols: int,
    grid_title: str,
    grid_banner_h: int = 44,
    grid_banner_bg: tuple[int, int, int] = (245, 245, 245),
    grid_banner_fg: tuple[int, int, int] = (20, 20, 20),
    pad_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Compose up to ``rows*cols`` panel videos into one grid video.

    The grid is truncated to the minimum number of rows needed for the
    panels supplied: if ``len(panel_frames) <= cols`` you get a single
    row, ``cols+1..2*cols`` two rows, etc. ``rows`` is the maximum
    allowed; extra panels beyond ``rows*cols`` are dropped. All input
    panels must have the same (T, H, W, 3).

    Returns ``(T, eff_rows*H + grid_banner_h, cols*W, 3)`` uint8.
    """
    if not panel_frames:
        raise ValueError("panel_frames is empty.")
    T, H, W, _ = panel_frames[0].shape
    for p in panel_frames:
        if p.shape != (T, H, W, 3):
            raise ValueError(
                f"All panels must share shape; got {p.shape} vs (T,H,W,3)="
                f"({T},{H},{W},3)."
            )

    max_panels = rows * cols
    if len(panel_frames) > max_panels:
        panel_frames = panel_frames[:max_panels]

    n_panels = len(panel_frames)
    eff_rows = (n_panels + cols - 1) // cols  # ceil
    # Pad the final partial row only.
    need = eff_rows * cols - n_panels
    if need > 0:
        blank = np.full((T, H, W, 3), pad_color, dtype=np.uint8)
        panel_frames = panel_frames + [blank] * need

    grid_W = cols * W
    grid_H = eff_rows * H + grid_banner_h
    out = np.zeros((T, grid_H, grid_W, 3), dtype=np.uint8)
    out[:, :grid_banner_h] = np.array(grid_banner_bg, dtype=np.uint8)[None, None, None]
    for r in range(eff_rows):
        for c in range(cols):
            i = r * cols + c
            y0 = grid_banner_h + r * H
            x0 = c * W
            out[:, y0:y0 + H, x0:x0 + W] = panel_frames[i]

    # Draw the grid title banner once on the first frame, then copy.
    img = Image.fromarray(out[0])
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            int(grid_banner_h * 0.55),
        )
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, max(0, (grid_banner_h - int(grid_banner_h * 0.55)) // 2 - 1)),
              grid_title, fill=grid_banner_fg, font=font)
    out[:, :grid_banner_h] = np.asarray(img)[:grid_banner_h]
    return out


def write_mp4(frames: np.ndarray, path: Path, fps: int = 50) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames as MP4. Path's parent is created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()


# ---------------------------------------------------------------------------
# Cluster composition serialization
# ---------------------------------------------------------------------------


def save_clusters_json(
    path: Path,
    pose_name: str,
    clusters: list[list[int]],
    info: dict[str, Any],
    code_usage: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cutoff = info.get("cutoff_used")
    if cutoff is None or (isinstance(cutoff, float) and np.isnan(cutoff)):
        cutoff_serialized: float | None = None
    else:
        cutoff_serialized = float(cutoff)
    payload = {
        "pose": pose_name,
        "strategy": info.get("strategy"),
        "cutoff_used": cutoff_serialized,
        "num_clusters": len(clusters),
        "clusters": [
            {
                "cluster_id": i,
                "size": len(cl),
                "codes": [int(c) for c in cl],
                "usage_total": int(sum(int(code_usage[c]) for c in cl)),
            }
            for i, cl in enumerate(clusters)
        ],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
