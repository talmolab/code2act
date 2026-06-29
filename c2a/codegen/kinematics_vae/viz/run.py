"""Top-level viz entry point invoked from the VAE variant's tokenize step.

Given encoded balanced-clip latents + per-frame codes, this:

  1. Loads the GT behavior labels from ``rodent_labels.csv`` and flattens
     them to ``(N_train_frames,)`` aligned to the train clips' frames.
  2. Subsamples to a manageable N for t-SNE / UMAP.
  3. Calls the variant-specific plotting helper.
  4. Saves PNG sidecars next to ``best_codes.npz``.

PNG outputs:
  ``viz_latent_<variant>.png``      — 2D PCA / t-SNE / UMAP × rows
  ``viz_code_usage_<variant>.png``  — codebook utilization histogram
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from c2a.codegen.kinematics_vae.viz.plot_latent import (
    plot_code_usage_hist,
    plot_latent_vae,
    plot_pca_pairs,
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
    """Return ``(n_train_clips * n_frames,)`` int combined ``behavior ×
    direction`` labels for the train clips, flattened in clip-major frame
    order.

    The combined index is ``behavior * 3 + direction`` over the canonical
    vocabularies ``BEHAVIORS × DIRECTIONS`` from
    :mod:`c2a.codegen.shared.label_alignment` ({Immobile, Rear, Turn,
    Walk} × {Left, Right, Straight}). Frames where either track is
    UNLABELED stay at -1.

    Returns:
        flattened labels in ``{-1, 0..11}`` where -1 is UNLABELED.
    """
    max_clip = int(np.max(train_indices))
    arr = load_label_frames(
        labels_csv_path,
        clip_range=(0, max_clip),
        n_frames=n_frames,
        csv_to_h5_subsample=int(csv_to_h5_subsample),
    )
    combined = arr.combined[train_indices]   # (n_train, n_frames)
    return combined.reshape(-1).astype(np.int32)


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
