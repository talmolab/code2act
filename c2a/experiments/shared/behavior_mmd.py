"""MMD + KID between rolled-out cluster qpos and labeled-GT behavior qpos.

Mirrors the TopoVNL feature-analysis flow (``topo_mimic.analysis.feature_analysis.mmd_kid``)
adapted for the c2a code-grid context:

* Each (code, body) rollout's qpos becomes one *cluster window*.
* Each labeled (behavior, direction) segment from rodent_reference_clips.h5
  becomes one *GT window*.
* All windows are uniformly resampled to ``K`` frames, root-stripped, and
  flattened into ``(N, K*D)`` vectors before MMD/KID.

Distance estimators (unbiased):

* **KID** uses the polynomial kernel ``k(x,y) = ((x·y)/d + 1)^3``.
* **MMD-RBF** uses a Gaussian kernel with σ from the median heuristic on
  the pooled distribution X ∪ Y (matching TopoVNL).

Both can go slightly negative on matching distributions (small-sample
variance artifact of the unbiased estimator).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from c2a.codegen.shared.label_alignment import (
    BEHAVIORS,
    DIRECTIONS,
    UNLABELED,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernels + estimators (1:1 with topo_mimic.feature_analysis.mmd_kid)
# ---------------------------------------------------------------------------


def _flatten_windows(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    raise ValueError(f"expected (N, K, D) or (N, D), got shape {x.shape}")


def polynomial_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """KID polynomial kernel: ``((x·y)/d + 1)^3``."""
    d = float(X.shape[1])
    G = X @ Y.T
    return (G / d + 1.0) ** 3


def median_heuristic_sigma(
    pool: np.ndarray,
    max_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """``σ = sqrt(median pairwise squared distance / 2)`` on a subsample."""
    n = pool.shape[0]
    if n > max_samples:
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        pool = pool[idx]
    sq = np.sum(pool ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * pool @ pool.T
    D2 = np.maximum(D2, 0.0)
    iu = np.triu_indices_from(D2, k=1)
    med_sq = float(np.median(D2[iu])) if iu[0].size else 1.0
    if med_sq <= 0.0:
        return 1.0
    return float(np.sqrt(med_sq / 2.0))


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    sq_x = np.sum(X ** 2, axis=1)
    sq_y = np.sum(Y ** 2, axis=1)
    D2 = sq_x[:, None] + sq_y[None, :] - 2.0 * X @ Y.T
    D2 = np.maximum(D2, 0.0)
    return np.exp(-D2 / (2.0 * sigma * sigma))


def mmd2_unbiased(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.sum() * 2.0 / (n * m)
    return float(sum_xx + sum_yy - sum_xy)


def compute_mmd_kid(
    cluster_windows: np.ndarray,
    real_windows: np.ndarray,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """KID + RBF-MMD² between cluster windows and real-GT windows."""
    X = _flatten_windows(cluster_windows)
    Y = _flatten_windows(real_windows)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return {
            "kid": float("nan"), "mmd_rbf": float("nan"),
            "n_cluster": int(n), "n_real": int(m), "sigma": float("nan"),
        }
    Kxx = polynomial_kernel(X, X)
    Kyy = polynomial_kernel(Y, Y)
    Kxy = polynomial_kernel(X, Y)
    kid = mmd2_unbiased(Kxx, Kyy, Kxy)

    pool = np.concatenate([X, Y], axis=0)
    sigma = median_heuristic_sigma(pool, rng=rng)
    Kxx_g = rbf_kernel(X, X, sigma)
    Kyy_g = rbf_kernel(Y, Y, sigma)
    Kxy_g = rbf_kernel(X, Y, sigma)
    mmd_rbf = mmd2_unbiased(Kxx_g, Kyy_g, Kxy_g)

    return {
        "kid": kid,
        "mmd_rbf": mmd_rbf,
        "n_cluster": int(n),
        "n_real": int(m),
        "sigma": float(sigma),
    }


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_to_K(seq: np.ndarray, K: int) -> np.ndarray:
    """Linearly resample ``(T, D)`` to ``(K, D)``.

    If ``T == K`` returns ``seq`` unchanged. ``T < 2`` returns a tile.
    """
    T = int(seq.shape[0])
    if T == K:
        return seq.astype(np.float32, copy=False)
    if T < 2:
        return np.broadcast_to(seq[0], (K, seq.shape[1])).astype(np.float32, copy=False)
    src = np.linspace(0.0, T - 1.0, num=T, dtype=np.float64)
    dst = np.linspace(0.0, T - 1.0, num=K, dtype=np.float64)
    out = np.empty((K, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(dst, src, seq[:, d])
    return out


# ---------------------------------------------------------------------------
# GT segment extraction
# ---------------------------------------------------------------------------


@dataclass
class GTSegment:
    """A labeled, contiguous run within a single clip."""
    clip_id: int
    start: int          # H5 frame, inclusive
    end: int            # H5 frame, exclusive
    behavior: int
    direction: int


def gt_label_string(behavior: int, direction: int) -> str:
    """Canonical GT label string. Full 4 × 3 = 12 cell space.

    All behaviors carry direction. ``"unlabeled"`` for missing behavior;
    bare ``"<Behavior>"`` only if direction is missing.
    """
    if behavior < 0 or behavior >= len(BEHAVIORS):
        return "unlabeled"
    b = BEHAVIORS[behavior]
    if direction < 0 or direction >= len(DIRECTIONS):
        return b
    return f"{b}_{DIRECTIONS[direction][0]}"  # Immobile_S, Walk_L, Turn_S, ...


def extract_gt_segments(
    behavior_arr: np.ndarray,
    direction_arr: np.ndarray,
    label_clip_indices: np.ndarray,
    min_length: int = 10,
) -> list[GTSegment]:
    """Iterate per-clip and emit maximal runs of constant (behavior, direction).

    Args:
        behavior_arr: ``(N_clips, T)`` int from :func:`load_label_frames`.
        direction_arr: same shape.
        label_clip_indices: ``(N_clips,)`` clip ids matching the rows.
        min_length: skip runs shorter than this many H5 frames.

    Returns:
        ``list[GTSegment]``. UNLABELED runs are excluded.
    """
    out: list[GTSegment] = []
    for row in range(behavior_arr.shape[0]):
        clip_id = int(label_clip_indices[row])
        b = behavior_arr[row]
        d = direction_arr[row]
        T = b.shape[0]
        if T == 0:
            continue
        # Encode (b, d) tuple as single int. UNLABELED rows: any -1 → flag.
        key = b.astype(np.int64) * 100 + d.astype(np.int64)
        boundaries = np.where(np.diff(key) != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [T]))
        for s, e in zip(starts, ends):
            if e - s < min_length:
                continue
            bb = int(b[s])
            dd = int(d[s])
            if bb == UNLABELED or dd == UNLABELED:
                continue
            out.append(
                GTSegment(clip_id=clip_id, start=int(s), end=int(e),
                          behavior=bb, direction=dd)
            )
    return out


def group_gt_segments(
    segments: Sequence[GTSegment],
) -> dict[str, list[GTSegment]]:
    """Group GT segments by canonical label string."""
    out: dict[str, list[GTSegment]] = {}
    for seg in segments:
        lab = gt_label_string(seg.behavior, seg.direction)
        out.setdefault(lab, []).append(seg)
    return out


def gt_windows_from_segments(
    segments: Sequence[GTSegment],
    qpos_h5_path: Path | str,
    n_frames_per_clip: int,
    K: int,
    strip_root_dims: int,
    max_segments: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Materialize ``(N, K, D)`` windows for one GT class.

    Each segment ``[start, end)`` from clip ``c`` is sliced from the flat H5
    qpos array as ``qpos[c * n_frames_per_clip + start : c * + end]``,
    root-stripped, and resampled to K frames.
    """
    import h5py
    if max_segments is not None and len(segments) > max_segments:
        idx = rng.choice(len(segments), size=max_segments, replace=False)
        segments = [segments[i] for i in idx]
    windows: list[np.ndarray] = []
    with h5py.File(str(qpos_h5_path), "r") as f:
        qpos = f["qpos"]
        for seg in segments:
            i0 = int(seg.clip_id) * int(n_frames_per_clip) + int(seg.start)
            i1 = int(seg.clip_id) * int(n_frames_per_clip) + int(seg.end)
            arr = np.asarray(qpos[i0:i1], dtype=np.float32)
            if strip_root_dims > 0 and arr.shape[1] > strip_root_dims:
                arr = arr[:, strip_root_dims:]
            windows.append(resample_to_K(arr, K))
    if not windows:
        return np.empty((0, K, 0), dtype=np.float32)
    return np.stack(windows, axis=0)


# ---------------------------------------------------------------------------
# Cluster windows from cached rollouts
# ---------------------------------------------------------------------------


def cluster_windows_from_rollouts(
    cluster_codes: Sequence[int],
    rollouts_data_dir: Path,
    pose: str,
    K: int,
    strip_root_dims: int,
    max_windows: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Pool ``(N, K, D)`` rollout qpos windows across a cluster's codes.

    One window per (code, body); ``N = K_per_panel * |cluster_codes|`` before
    optional subsample to ``max_windows``.
    """
    windows: list[np.ndarray] = []
    for code in cluster_codes:
        p = rollouts_data_dir / f"rollout_{pose}_code{int(code):03d}.npz"
        if not p.exists():
            log.warning(f"missing rollout file {p}; skipping code {code}")
            continue
        d = np.load(p)
        qpos = np.asarray(d["qpos"], dtype=np.float32)  # (K_bodies, T, nq)
        for k in range(qpos.shape[0]):
            arr = qpos[k]
            if strip_root_dims > 0 and arr.shape[1] > strip_root_dims:
                arr = arr[:, strip_root_dims:]
            windows.append(resample_to_K(arr, K))
    if not windows:
        return np.empty((0, K, 0), dtype=np.float32), 0
    arr = np.stack(windows, axis=0)
    n_pre = int(arr.shape[0])
    if max_windows is not None and arr.shape[0] > max_windows:
        idx = rng.choice(arr.shape[0], size=max_windows, replace=False)
        arr = arr[idx]
    return arr, n_pre
