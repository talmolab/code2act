"""Per-variant latent visualization (3 projections × N rows).

Common layout
-------------
Each variant emits one figure with 3 columns (PCA, t-SNE, UMAP) and
``N`` rows. Within a column, all rows use the SAME 2D embedding —
points stay in the same position; only the coloring changes between
rows.

Row 1 is always **ground-truth-label coloring** (4-class behavior from
``c2a.codegen.shared.label_alignment``: Immobile / Rear / Turn / Walk).

Row 2 is the **discovered codes** (the variant's discrete code per
frame).

Variant-specific extensions
---------------------------
- VAE: just rows 1+2.
- VQVAE: rows 1+2 + overlay codebook centroids as larger X markers in
  the same projection (Row 3).
- PS-VQVAE: per-branch rows (one set of three columns per body group
  + a joint K_i × K_j heatmap as a separate figure).

The functions return matplotlib Figure objects; callers decide where
to save.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from c2a.codegen.kinematics_vae.viz.projections import (
    pca_2d,
    subsample,
    tsne_2d,
    umap_2d,
)

log = logging.getLogger(__name__)


def _color_legend_handles(unique_labels: np.ndarray, cmap_name: str = "tab10"):
    cmap = plt.get_cmap(cmap_name, max(int(unique_labels.max()) + 1, 1))
    return [
        plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=cmap(int(l)), markeredgecolor="none",
            label=str(int(l)),
        )
        for l in unique_labels
    ]


def _scatter_panel(
    ax,
    xy: np.ndarray,
    colors: np.ndarray,
    *,
    cmap: str,
    title: str,
    s: float = 4.0,
    alpha: float = 0.6,
) -> None:
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=colors, cmap=cmap, s=s, alpha=alpha,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def plot_latent_vae(
    z: np.ndarray,
    behavior_labels: np.ndarray,
    codes: np.ndarray,
    *,
    n_max: int = 20_000,
    seed: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Vanilla VAE: 3 cols × 2 rows (proj × {GT label, k-means code}).

    Args:
        z: ``(N, D)`` per-frame latents (typically μ).
        behavior_labels: ``(N,)`` integer GT behavior labels.
        codes: ``(N,)`` discovered codes (k-means cluster ids).
        n_max: Subsample to this many points before t-SNE/UMAP.
    """
    z_sub, idx = subsample(z, n_max=n_max, seed=seed)
    log.info("plot_latent_vae: %d points -> %d sampled", z.shape[0], z_sub.shape[0])

    proj_pca = pca_2d(z_sub, seed=seed)
    proj_tsne = tsne_2d(z_sub, seed=seed)
    proj_umap = umap_2d(z_sub, seed=seed)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    cols = [
        ("PCA", proj_pca),
        ("t-SNE", proj_tsne),
        ("UMAP", proj_umap),
    ]
    for j, (col_name, xy) in enumerate(cols):
        _scatter_panel(
            axes[0, j], xy, behavior_labels[idx],
            cmap="tab10", title=f"{col_name} — GT behavior",
        )
        _scatter_panel(
            axes[1, j], xy, codes[idx],
            cmap="tab20", title=f"{col_name} — VAE code",
        )

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_latent_vqvae(
    z_e: np.ndarray,
    codebook: np.ndarray,
    behavior_labels: np.ndarray,
    codes: np.ndarray,
    *,
    n_max: int = 20_000,
    seed: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Vanilla VQVAE: 3 cols × 3 rows.

    Row 0: GT behavior label.
    Row 1: discrete code.
    Row 2: same projection with **codebook centroids** overlaid as
        large ``X`` markers — visualizes whether the codebook is
        clustered or spread across the latent manifold.
    """
    z_sub, idx = subsample(z_e, n_max=n_max, seed=seed)
    # Project z_e AND codebook in the same PCA/UMAP space by concatenating.
    K = codebook.shape[0]
    stacked = np.concatenate([z_sub, codebook], axis=0)
    proj_pca_st = pca_2d(stacked, seed=seed)
    proj_tsne_st = tsne_2d(stacked, seed=seed)
    proj_umap_st = umap_2d(stacked, seed=seed)
    proj_pca, proj_pca_cb = proj_pca_st[:-K], proj_pca_st[-K:]
    proj_tsne, proj_tsne_cb = proj_tsne_st[:-K], proj_tsne_st[-K:]
    proj_umap, proj_umap_cb = proj_umap_st[:-K], proj_umap_st[-K:]

    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    cols = [
        ("PCA", proj_pca, proj_pca_cb),
        ("t-SNE", proj_tsne, proj_tsne_cb),
        ("UMAP", proj_umap, proj_umap_cb),
    ]
    for j, (col_name, xy, xy_cb) in enumerate(cols):
        _scatter_panel(
            axes[0, j], xy, behavior_labels[idx],
            cmap="tab10", title=f"{col_name} — GT behavior",
        )
        _scatter_panel(
            axes[1, j], xy, codes[idx],
            cmap="tab20", title=f"{col_name} — VQVAE code",
        )
        # Row 2: code-colored points + codebook centroids overlaid.
        ax = axes[2, j]
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=codes[idx], cmap="tab20", s=3, alpha=0.4, edgecolors="none",
        )
        ax.scatter(
            xy_cb[:, 0], xy_cb[:, 1],
            c=np.arange(K), cmap="tab20",
            s=80, marker="X", edgecolors="black", linewidths=0.8,
        )
        ax.set_title(f"{col_name} — codes + codebook ✕", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_code_usage_hist(
    codes: np.ndarray,
    num_codes: int,
    *,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of code usage. Useful for diagnosing codebook collapse
    (all mass concentrated on a few codes) at-a-glance."""
    counts = np.bincount(codes.ravel().astype(np.int64), minlength=int(num_codes))
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    ax.bar(np.arange(num_codes), counts, color="steelblue")
    ax.set_xlabel("Code")
    ax.set_ylabel("Count")
    ax.set_title(title or "Code usage histogram", fontsize=10)
    return fig
