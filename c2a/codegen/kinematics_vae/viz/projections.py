"""Thin wrappers around sklearn PCA / t-SNE / umap-learn UMAP.

Used by the per-variant viz scripts to project the high-dimensional
latent (``z`` for VAE, ``z_e`` for VQVAE, per-branch ``μ`` for PS-VQVAE)
into a 2D plane for ground-truth-label and code coloring.

All projections operate on ``(N, D)`` float32 arrays of latents and
return ``(N, 2)``. The functions log their fit on a `transform` style
where applicable so the same projection is reused across rows of the
plot (label-colored row and code-colored row see the same 2D points).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def subsample(
    x: np.ndarray, n_max: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_sub, idx)`` where ``x_sub = x[idx]`` and
    ``len(x_sub) <= n_max``. Useful to keep t-SNE/UMAP runtime bounded."""
    if x.shape[0] <= n_max:
        return x, np.arange(x.shape[0])
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=int(n_max), replace=False)
    idx.sort()
    return x[idx], idx


def pca_2d(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Standard PCA → 2D. Deterministic given ``seed``."""
    from sklearn.decomposition import PCA

    p = PCA(n_components=2, random_state=int(seed))
    return p.fit_transform(x).astype(np.float32)


def pca_2d_with_variance(
    x: np.ndarray, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``pca_2d`` variant that also returns the per-component explained
    variance ratio (length 2), for plotting on the axis label."""
    from sklearn.decomposition import PCA

    p = PCA(n_components=2, random_state=int(seed))
    proj = p.fit_transform(x).astype(np.float32)
    return proj, p.explained_variance_ratio_.astype(np.float64)


def tsne_2d(
    x: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """t-SNE → 2D. PCA-reduces to 50 dims first if input is wider,
    which is sklearn's default (and what UMAP-comparable papers use)."""
    from sklearn.manifold import TSNE

    if x.shape[1] > 50:
        x = pca_2d_to_d(x, d=50, seed=seed)
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        max_iter=int(n_iter),
        random_state=int(seed),
        init="pca",
    )
    return tsne.fit_transform(x).astype(np.float32)


def umap_2d(
    x: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """UMAP → 2D. Optional dependency: ``pip install umap-learn``.

    Falls back to ``pca_2d`` with a warning if ``umap-learn`` is not
    installed (so the viz pipeline still produces something even on a
    minimal install).
    """
    try:
        import umap  # type: ignore
    except ImportError:
        log.warning(
            "umap-learn is not installed; falling back to PCA-2D for "
            "the 'UMAP' panel. `pip install umap-learn` to enable real UMAP."
        )
        return pca_2d(x, seed=seed)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(seed),
    )
    return reducer.fit_transform(x).astype(np.float32)


def pca_2d_to_d(x: np.ndarray, d: int = 50, seed: int = 0) -> np.ndarray:
    """PCA → d-dimensional (used as a t-SNE preprocessor)."""
    from sklearn.decomposition import PCA

    if x.shape[1] <= d:
        return x
    p = PCA(n_components=int(d), random_state=int(seed))
    return p.fit_transform(x).astype(np.float32)
