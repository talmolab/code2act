"""Per-variant latent visualization (3 projections × N rows).

Common layout
-------------
Each variant emits one figure with 3 columns (PCA, t-SNE, UMAP) and
``N`` rows. Within a column, all rows use the SAME 2D embedding —
points stay in the same position; only the coloring changes between
rows.

Row 1 is always **ground-truth-label coloring** using the 12-class
``behavior × direction`` combined scheme from
``c2a.codegen.shared.label_alignment``
(``BEHAVIORS × DIRECTIONS`` = {Immobile, Rear, Turn, Walk} ×
{Left, Right, Straight}; combined index = ``behavior * 3 + direction``).
Hue encodes behavior, lightness encodes direction.

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
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from c2a.codegen.kinematics_vae.viz.projections import (
    pca_2d,
    pca_2d_with_variance,
    subsample,
    tsne_2d,
    umap_2d,
)


def _pc_label(name: str, explained: float | None) -> str:
    """Format an axis label like ``"PC1 (42.7%)"`` when variance is known."""
    if explained is None:
        return name
    return f"{name} ({explained * 100:.1f}%)"

log = logging.getLogger(__name__)


# GT-label name mapping for the 12-class combined ``behavior × direction``
# scheme exported by ``c2a.codegen.shared.label_alignment``. Combined index
# is ``behavior * 3 + direction`` over
# ``BEHAVIORS = (Immobile, Rear, Turn, Walk)`` ×
# ``DIRECTIONS = (Left, Right, Straight)``. ``-1`` = UNLABELED.
#
# Color scheme: HUE encodes behavior (blue=Immobile, red=Rear, green=Turn,
# orange=Walk), LIGHTNESS encodes direction (dark=Left, medium=Right,
# light=Straight). So the eye reads behavior first, then direction.
_GT_NAMES: tuple[str, ...] = (
    "UNLABELED",
    "Immobile_Left", "Immobile_Right", "Immobile_Straight",
    "Rear_Left", "Rear_Right", "Rear_Straight",
    "Turn_Left", "Turn_Right", "Turn_Straight",
    "Walk_Left", "Walk_Right", "Walk_Straight",
)
_GT_VALUES: tuple[int, ...] = (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_GT_COLORS: tuple[str, ...] = (
    "#999999",   # UNLABELED         — neutral grey
    "#08306b",   # Immobile_Left     — dark blue
    "#2171b5",   # Immobile_Right    — medium blue
    "#9ecae1",   # Immobile_Straight — light blue
    "#67000d",   # Rear_Left         — dark red
    "#cb181d",   # Rear_Right        — medium red
    "#fcae91",   # Rear_Straight     — light red/salmon
    "#00441b",   # Turn_Left         — dark green
    "#238b45",   # Turn_Right        — medium green
    "#a1d99b",   # Turn_Straight     — light green
    "#7f2704",   # Walk_Left         — dark orange/brown
    "#d94801",   # Walk_Right        — medium orange
    "#fdae6b",   # Walk_Straight     — light orange
)


def _gt_cmap_and_norm() -> tuple[ListedColormap, BoundaryNorm]:
    """``ListedColormap`` + ``BoundaryNorm`` so class index → color is
    deterministic (matplotlib's default colormap normalization with a
    raw integer column changes when min/max change between subsamples)."""
    cmap = ListedColormap(_GT_COLORS, name="c2a_gt")
    # Boundaries between consecutive class values: -1.5, -0.5, 0.5, …, 11.5
    # → len(_GT_VALUES) bins.
    boundaries = [v - 0.5 for v in _GT_VALUES] + [_GT_VALUES[-1] + 0.5]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    return cmap, norm


def _gt_legend_handles(present: np.ndarray):
    """Legend handles for the GT classes present in ``present``."""
    present_set = set(int(v) for v in np.unique(present).tolist())
    return [
        plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=color, markeredgecolor="none",
            label=name, markersize=8,
        )
        for val, name, color in zip(_GT_VALUES, _GT_NAMES, _GT_COLORS)
        if val in present_set
    ]


def _scatter_gt_panel(
    ax,
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    s: float = 4.0,
    alpha: float = 0.7,
) -> None:
    """GT-label scatter with stable index→color mapping."""
    cmap, norm = _gt_cmap_and_norm()
    ax.scatter(
        xy[:, 0], xy[:, 1],
        c=labels, cmap=cmap, norm=norm,
        s=s, alpha=alpha, edgecolors="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


def _scatter_code_panel(
    ax,
    xy: np.ndarray,
    codes: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "tab20",
    s: float = 3.5,
    alpha: float = 0.55,
):
    """Scatter colored by discrete code id. Returns the scatter handle
    so the caller can attach a colorbar."""
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=codes, cmap=cmap, s=s, alpha=alpha,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    return sc


def plot_latent_vae(
    z: np.ndarray,
    gt_labels: np.ndarray,
    codes: np.ndarray,
    *,
    n_max: int = 20_000,
    seed: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Vanilla VAE: 3 cols × 2 rows (proj × {GT label, k-means code}).

    Args:
        z: ``(N, D)`` per-frame latents (typically μ).
        gt_labels: ``(N,)`` integer GT combined behavior×direction
            labels in ``{-1=UNLABELED, 0..11}`` where the value is
            ``behavior * 3 + direction`` over ``BEHAVIORS × DIRECTIONS``
            (see :mod:`c2a.codegen.shared.label_alignment`).
        codes: ``(N,)`` discovered codes (k-means cluster ids).
        n_max: Subsample to this many points before t-SNE/UMAP.

    Layout:
        ┌─────────┬───────┬──────┐
        │ PCA     │ t-SNE │ UMAP │  ← row 0: GT behavior×direction (legend right)
        ├─────────┼───────┼──────┤
        │ PCA     │ t-SNE │ UMAP │  ← row 1: VAE code (colorbar right)
        └─────────┴───────┴──────┘
    """
    z_sub, idx = subsample(z, n_max=n_max, seed=seed)
    log.info("plot_latent_vae: %d points -> %d sampled", z.shape[0], z_sub.shape[0])

    proj_pca, pca_var = pca_2d_with_variance(z_sub, seed=seed)
    proj_tsne = tsne_2d(z_sub, seed=seed)
    proj_umap = umap_2d(z_sub, seed=seed)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    cols = [
        ("PCA", proj_pca,
         (_pc_label("PC1", pca_var[0]), _pc_label("PC2", pca_var[1]))),
        ("t-SNE", proj_tsne, ("t-SNE 1", "t-SNE 2")),
        ("UMAP", proj_umap, ("UMAP 1", "UMAP 2")),
    ]
    sub_labels = gt_labels[idx]
    sub_codes = codes[idx]
    code_sc = None
    for j, (col_name, xy, (xlab, ylab)) in enumerate(cols):
        _scatter_gt_panel(
            axes[0, j], xy, sub_labels,
            title=f"{col_name} — GT behavior×direction",
            xlabel=xlab, ylabel=ylab,
        )
        code_sc = _scatter_code_panel(
            axes[1, j], xy, sub_codes,
            title=f"{col_name} — VAE code",
            xlabel=xlab, ylabel=ylab,
        )
    # Legend on row 0 (right-aligned, outside the last column).
    axes[0, -1].legend(
        handles=_gt_legend_handles(sub_labels),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, frameon=False,
    )
    # Colorbar on row 1 (continuous code index).
    cbar = fig.colorbar(
        code_sc, ax=axes[1, -1], fraction=0.04, pad=0.04,
    )
    cbar.set_label("VAE code (k-means cluster id)", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_latent_vqvae(
    z_e: np.ndarray,
    codebook: np.ndarray,
    gt_labels: np.ndarray,
    codes: np.ndarray,
    *,
    n_max: int = 20_000,
    seed: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Vanilla VQVAE: 3 cols × 3 rows.

    Row 0: GT behavior label  (legend on the right of row 0)
    Row 1: discrete code      (colorbar on the right of row 1)
    Row 2: codes + codebook centroids overlaid as ``X`` markers, so
           you can see whether the codebook spreads across the latent
           manifold or piles into a few clusters.
    """
    z_sub, idx = subsample(z_e, n_max=n_max, seed=seed)
    # Project z_e AND codebook in the same PCA/UMAP space by concatenating.
    K = codebook.shape[0]
    stacked = np.concatenate([z_sub, codebook], axis=0)
    proj_pca_st, pca_var = pca_2d_with_variance(stacked, seed=seed)
    proj_tsne_st = tsne_2d(stacked, seed=seed)
    proj_umap_st = umap_2d(stacked, seed=seed)
    proj_pca, proj_pca_cb = proj_pca_st[:-K], proj_pca_st[-K:]
    proj_tsne, proj_tsne_cb = proj_tsne_st[:-K], proj_tsne_st[-K:]
    proj_umap, proj_umap_cb = proj_umap_st[:-K], proj_umap_st[-K:]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)
    cols = [
        ("PCA", proj_pca, proj_pca_cb,
         (_pc_label("PC1", pca_var[0]), _pc_label("PC2", pca_var[1]))),
        ("t-SNE", proj_tsne, proj_tsne_cb, ("t-SNE 1", "t-SNE 2")),
        ("UMAP", proj_umap, proj_umap_cb, ("UMAP 1", "UMAP 2")),
    ]
    sub_labels = gt_labels[idx]
    sub_codes = codes[idx]
    code_sc = None
    for j, (col_name, xy, xy_cb, (xlab, ylab)) in enumerate(cols):
        _scatter_gt_panel(
            axes[0, j], xy, sub_labels,
            title=f"{col_name} — GT behavior×direction",
            xlabel=xlab, ylabel=ylab,
        )
        code_sc = _scatter_code_panel(
            axes[1, j], xy, sub_codes,
            title=f"{col_name} — VQVAE code",
            xlabel=xlab, ylabel=ylab,
        )
        # Row 2: code-colored points + codebook centroids overlaid.
        _scatter_code_panel(
            axes[2, j], xy, sub_codes,
            title=f"{col_name} — codes + codebook ✕",
            xlabel=xlab, ylabel=ylab,
            alpha=0.4,
        )
        axes[2, j].scatter(
            xy_cb[:, 0], xy_cb[:, 1],
            c=np.arange(K), cmap="tab20",
            s=80, marker="X", edgecolors="black", linewidths=0.8,
        )

    axes[0, -1].legend(
        handles=_gt_legend_handles(sub_labels),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, frameon=False,
    )
    cbar = fig.colorbar(
        code_sc, ax=axes[1, -1], fraction=0.04, pad=0.04,
    )
    cbar.set_label(f"VQVAE code (K={K})", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_latent_ps_vqvae(
    z_e: np.ndarray,
    per_group_codes: np.ndarray,
    group_dims: list[int],
    group_codebook_sizes: list[int],
    gt_labels: np.ndarray,
    *,
    n_max: int = 20_000,
    seed: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """PS-VQVAE: per-group 3-cols × 2-rows (GT vs per-group code).

    Each body group is projected independently (PCA / t-SNE / UMAP over
    that group's sub-latent slice) and colored by (a) GT behavior, (b)
    discovered per-group code. Legend on row 0 of group 0; per-group
    colorbar on each code row.

    Use :func:`plot_ps_vqvae_joint_heatmap` separately for the K_i × K_j
    factor-decorrelation diagnostic.
    """
    G = int(per_group_codes.shape[-1])
    assert G == len(group_dims) == len(group_codebook_sizes)
    z_sub, idx = subsample(z_e, n_max=n_max, seed=seed)
    log.info(
        "plot_latent_ps_vqvae: %d points -> %d sampled, G=%d groups",
        z_e.shape[0], z_sub.shape[0], G,
    )

    offsets = np.cumsum([0] + group_dims)
    sub_labels = gt_labels[idx]
    fig, axes = plt.subplots(
        2 * G, 3, figsize=(16, 5 * G), constrained_layout=True,
    )
    if G == 1:
        axes = axes[np.newaxis, :]

    last_code_sc = None
    for g in range(G):
        z_g = z_sub[:, offsets[g]:offsets[g + 1]]
        codes_g = per_group_codes[idx, g]
        proj_pca, pca_var = pca_2d_with_variance(z_g, seed=seed)
        proj_tsne = tsne_2d(z_g, seed=seed)
        proj_umap = umap_2d(z_g, seed=seed)
        cols = [
            ("PCA", proj_pca,
             (_pc_label("PC1", pca_var[0]), _pc_label("PC2", pca_var[1]))),
            ("t-SNE", proj_tsne, ("t-SNE 1", "t-SNE 2")),
            ("UMAP", proj_umap, ("UMAP 1", "UMAP 2")),
        ]
        for j, (col_name, xy, (xlab, ylab)) in enumerate(cols):
            _scatter_gt_panel(
                axes[2 * g, j], xy, sub_labels,
                title=f"group {g} (dim={group_dims[g]}) {col_name} — GT",
                xlabel=xlab, ylabel=ylab,
            )
            last_code_sc = _scatter_code_panel(
                axes[2 * g + 1, j], xy, codes_g,
                title=f"group {g} (K={group_codebook_sizes[g]}) {col_name} — code",
                xlabel=xlab, ylabel=ylab,
            )
        # Per-group code colorbar on the right of the code row.
        fig.colorbar(
            last_code_sc, ax=axes[2 * g + 1, -1],
            fraction=0.04, pad=0.04,
        ).set_label(
            f"group {g} code (K={group_codebook_sizes[g]})", fontsize=8,
        )

    # Single GT legend on the top-right of the figure.
    axes[0, -1].legend(
        handles=_gt_legend_handles(sub_labels),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, frameon=False,
    )

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_ps_vqvae_joint_heatmap(
    per_group_codes: np.ndarray,
    group_codebook_sizes: list[int],
    *,
    title: Optional[str] = None,
) -> plt.Figure:
    """K_i × K_j joint-occupancy heatmaps for every pair of groups.

    Per VQ-MAP paper, the product-space groups should factorize into
    near-independent factors. A diagonal-aligned occupancy heatmap
    means the factors are correlated (degenerate product space); a
    uniform-ish heatmap means they're independent — the desired state.
    """
    G = len(group_codebook_sizes)
    if per_group_codes.shape[-1] != G:
        raise ValueError(
            f"per_group_codes last axis {per_group_codes.shape[-1]} != G={G}."
        )
    pairs = [(i, j) for i in range(G) for j in range(i + 1, G)]
    n_pairs = len(pairs)
    if n_pairs == 0:
        # Only one group — joint heatmap doesn't apply.
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "G=1 — no pairs to show",
                ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(
        1, n_pairs, figsize=(4 * n_pairs, 3.5), constrained_layout=True,
    )
    if n_pairs == 1:
        axes = [axes]
    flat = per_group_codes.reshape(-1, G)
    for ax, (i, j) in zip(axes, pairs):
        Ki, Kj = int(group_codebook_sizes[i]), int(group_codebook_sizes[j])
        heat = np.zeros((Ki, Kj), dtype=np.int64)
        np.add.at(heat, (flat[:, i].astype(np.int64), flat[:, j].astype(np.int64)), 1)
        im = ax.imshow(heat, cmap="viridis", aspect="auto")
        ax.set_xlabel(f"group {j} (K={Kj})")
        ax.set_ylabel(f"group {i} (K={Ki})")
        ax.set_title(f"joint occupancy ({i}, {j})")
        fig.colorbar(im, ax=ax)
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def plot_pca_pairs(
    z: np.ndarray,
    gt_labels: np.ndarray,
    *,
    pairs: Sequence[tuple[int, int]] = ((0, 1), (0, 2), (1, 2)),
    n_max: int = 20_000,
    seed: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Scott-style diagnostic: scatter ``len(pairs)`` PCA pairs colored
    by GT behavior so you can visually verify the principal axes carry
    real behavioral structure (one 2D PCA can hide variance in higher PCs).

    Default pairs = ``[(PC1, PC2), (PC1, PC3), (PC2, PC3)]``.
    """
    z_sub, idx = subsample(z, n_max=n_max, seed=seed)
    n_components = max(max(i, j) for i, j in pairs) + 1
    from sklearn.decomposition import PCA
    p = PCA(n_components=n_components, random_state=int(seed))
    proj = p.fit_transform(z_sub)
    var_ratio = p.explained_variance_ratio_

    sub_labels = gt_labels[idx]
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(5 * len(pairs) + 2, 4.5),
        constrained_layout=True,
    )
    if len(pairs) == 1:
        axes = [axes]
    for ax, (i, j) in zip(axes, pairs):
        _scatter_gt_panel(
            ax, proj[:, [i, j]], sub_labels,
            title=f"PCA: PC{i + 1} vs PC{j + 1}",
            xlabel=_pc_label(f"PC{i + 1}", var_ratio[i]),
            ylabel=_pc_label(f"PC{j + 1}", var_ratio[j]),
        )
    axes[-1].legend(
        handles=_gt_legend_handles(sub_labels),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, frameon=False,
    )
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
