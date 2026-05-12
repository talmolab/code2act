"""Top-level viz entry point invoked from each variant's tokenize step.

Given encoded balanced-clip latents + per-frame codes + (for VQ variants)
the trained codebook, this:

  1. Loads the GT behavior labels from ``rodent_labels.csv`` and flattens
     them to ``(N_train_frames,)`` aligned to the train clips' frames.
  2. Subsamples to a manageable N for t-SNE / UMAP.
  3. Calls the variant-specific plotting helper.
  4. Saves PNG sidecars next to ``best_codes.npz``.

PNG outputs:
  ``viz_latent_<variant>.png``      — 2D PCA / t-SNE / UMAP × rows
  ``viz_code_usage_<variant>.png``  — codebook utilization histogram
  ``viz_joint_heatmap_ps_vqvae.png``— PS-VQVAE only: K_i × K_j co-occupancy
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from c2a.codegen.kinematics_vae.viz.plot_latent import (
    plot_code_usage_hist,
    plot_latent_ps_vqvae,
    plot_latent_vae,
    plot_latent_vqvae,
    plot_pca_pairs,
    plot_ps_vqvae_joint_heatmap,
)
from c2a.codegen.shared.label_alignment import load_label_frames
from c2a.paths import get_repo_root

log = logging.getLogger(__name__)


def _gt_train_frame_labels(
    labels_csv_path: str | Path,
    train_indices: np.ndarray,
    n_frames: int,
    csv_to_h5_subsample: int = 2,
) -> np.ndarray:
    """Return ``(n_train_clips * n_frames,)`` int behavior labels for the
    train clips, flattened in clip-major frame order.

    Returns:
        flattened labels in ``{-1, 0, 1, 2, 3}`` where -1 is UNLABELED.
    """
    max_clip = int(np.max(train_indices))
    arr = load_label_frames(
        labels_csv_path,
        clip_range=(0, max_clip),
        n_frames=n_frames,
        csv_to_h5_subsample=int(csv_to_h5_subsample),
    )
    behavior = arr.behavior[train_indices]   # (n_train, n_frames)
    return behavior.reshape(-1).astype(np.int32)


def _train_subset(
    *,
    z_or_e: np.ndarray,                # (n_clips, n_frames, latent_dim)
    codes: np.ndarray,                 # (n_clips, n_frames) OR (n_clips, n_frames, G)
    clip_indices: np.ndarray,          # (n_clips,) sorted-union order
    train_indices: np.ndarray,         # (n_train,) caller order
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the encoded sorted-union arrays down to train clips,
    flatten to ``(n_train * n_frames, latent_dim)`` + matching codes."""
    idx_to_pos = {int(c): pos for pos, c in enumerate(clip_indices.tolist())}
    train_positions = [idx_to_pos[int(c)] for c in train_indices.tolist()]
    z_train = z_or_e[train_positions]           # (n_train, n_frames, D)
    codes_train = codes[train_positions]        # (n_train, n_frames[, G])
    n_train, n_frames, D = z_train.shape
    z_flat = z_train.reshape(n_train * n_frames, D)
    if codes_train.ndim == 3:
        codes_flat = codes_train.reshape(n_train * n_frames, codes_train.shape[-1])
    else:
        codes_flat = codes_train.reshape(n_train * n_frames)
    return z_flat, codes_flat


def write_vae_viz(
    output_dir: str | Path,
    *,
    z: np.ndarray,
    codes: np.ndarray,
    clip_indices: np.ndarray,
    train_indices: np.ndarray,
    num_codes: int,
    labels_csv_path: str | Path,
    n_max: int = 20_000,
    seed: int = 0,
    csv_to_h5_subsample: int = 2,
) -> list[Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    z_flat, codes_flat = _train_subset(
        z_or_e=z, codes=codes, clip_indices=clip_indices,
        train_indices=train_indices,
    )
    gt = _gt_train_frame_labels(
        labels_csv_path, train_indices, z.shape[1],
        csv_to_h5_subsample=csv_to_h5_subsample,
    )
    log.info("VAE viz: %d frames -> subsample %d", z_flat.shape[0], n_max)

    saved: list[Path] = []
    fig = plot_latent_vae(
        z_flat, gt, codes_flat, n_max=n_max, seed=seed,
        title="kinematics-VAE latent (z=μ) — train frames",
    )
    p = output_dir / "viz_latent_kinematics_vae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    fig = plot_code_usage_hist(
        codes_flat, num_codes, title="kinematics-VAE — k-means code usage",
    )
    p = output_dir / "viz_code_usage_kinematics_vae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # Scott-style PC-pair diagnostic on the full 24-D latent.
    fig = plot_pca_pairs(
        z_flat, gt,
        title="kinematics-VAE — PCA pairs on latent (train frames)",
    )
    p = output_dir / "viz_pca_pairs_kinematics_vae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)
    log.info("Wrote %d viz PNGs to %s", len(saved), output_dir)
    return saved


def write_vqvae_viz(
    output_dir: str | Path,
    *,
    z_e: np.ndarray,
    codes: np.ndarray,
    codebook: np.ndarray,
    clip_indices: np.ndarray,
    train_indices: np.ndarray,
    num_codes: int,
    labels_csv_path: str | Path,
    n_max: int = 20_000,
    seed: int = 0,
    csv_to_h5_subsample: int = 2,
) -> list[Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    z_flat, codes_flat = _train_subset(
        z_or_e=z_e, codes=codes, clip_indices=clip_indices,
        train_indices=train_indices,
    )
    gt = _gt_train_frame_labels(
        labels_csv_path, train_indices, z_e.shape[1],
        csv_to_h5_subsample=csv_to_h5_subsample,
    )
    log.info("VQVAE viz: %d frames -> subsample %d", z_flat.shape[0], n_max)

    saved: list[Path] = []
    fig = plot_latent_vqvae(
        z_flat, codebook, gt, codes_flat, n_max=n_max, seed=seed,
        title="VQVAE latent (z_e) + codebook centroids — train frames",
    )
    p = output_dir / "viz_latent_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    fig = plot_code_usage_hist(
        codes_flat, num_codes, title="VQVAE — codebook usage",
    )
    p = output_dir / "viz_code_usage_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # Scott-style PC-pair diagnostic on the full 24-D z_e.
    fig = plot_pca_pairs(
        z_flat, gt, title="VQVAE — PCA pairs on z_e (train frames)",
    )
    p = output_dir / "viz_pca_pairs_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)
    log.info("Wrote %d viz PNGs to %s", len(saved), output_dir)
    return saved


def write_ps_vqvae_viz(
    output_dir: str | Path,
    *,
    z_e: np.ndarray,
    per_group_codes: np.ndarray,            # (n_clips, n_frames, G)
    product_codes: np.ndarray,              # (n_clips, n_frames) folded index
    clip_indices: np.ndarray,
    train_indices: np.ndarray,
    group_dims: Sequence[int],
    group_codebook_sizes: Sequence[int],
    num_codes_product: int,
    labels_csv_path: str | Path,
    n_max: int = 20_000,
    seed: int = 0,
    csv_to_h5_subsample: int = 2,
) -> list[Path]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    z_flat, codes_flat = _train_subset(
        z_or_e=z_e, codes=per_group_codes,
        clip_indices=clip_indices, train_indices=train_indices,
    )                                        # codes_flat: (N, G)
    _, prod_flat = _train_subset(
        z_or_e=z_e, codes=product_codes,
        clip_indices=clip_indices, train_indices=train_indices,
    )
    gt = _gt_train_frame_labels(
        labels_csv_path, train_indices, z_e.shape[1],
        csv_to_h5_subsample=csv_to_h5_subsample,
    )
    log.info("PS-VQVAE viz: %d frames -> subsample %d", z_flat.shape[0], n_max)

    saved: list[Path] = []
    fig = plot_latent_ps_vqvae(
        z_flat, codes_flat, list(group_dims), list(group_codebook_sizes), gt,
        n_max=n_max, seed=seed,
        title="PS-VQVAE per-group latent — train frames",
    )
    p = output_dir / "viz_latent_ps_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    fig = plot_ps_vqvae_joint_heatmap(
        codes_flat.reshape(-1, codes_flat.shape[-1]),
        list(group_codebook_sizes),
        title="PS-VQVAE — K_i × K_j joint occupancy",
    )
    p = output_dir / "viz_joint_heatmap_ps_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    fig = plot_code_usage_hist(
        prod_flat, num_codes_product,
        title=f"PS-VQVAE — product-index usage (K={num_codes_product})",
    )
    p = output_dir / "viz_code_usage_ps_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)

    # Scott-style PC-pair diagnostic on the full concatenated z_e.
    fig = plot_pca_pairs(
        z_flat, gt, title="PS-VQVAE — PCA pairs on z_e (train frames)",
    )
    p = output_dir / "viz_pca_pairs_ps_vqvae.png"
    fig.savefig(p, dpi=150); plt.close(fig); saved.append(p)
    log.info("Wrote %d viz PNGs to %s", len(saved), output_dir)
    return saved
