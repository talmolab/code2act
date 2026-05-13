"""Discretize VAE latents into per-frame codes via K-Means (+ optional PCA).

Pipeline:
    1. Restore the trained VAE (encoder + Gaussian bottleneck) from
       a checkpoint directory written by
       :func:`c2a.codegen.kinematics_vae.backbone.train_loop.train`.
    2. Encode every balanced-split clip frame deterministically:
       ``z = μ``.
    3. (Optional) Fit a PCA on the flattened ``z`` train-set and reduce
       to ``pca_dim`` components.
    4. Fit ``KMeans(n_clusters=num_codes)`` on the train-clip latents
       (positionally indexed in the sorted balanced-split union).
    5. Predict codes for every clip frame.
    6. Subset / re-order against ``train_indices`` / ``test_indices``
       and write a CodesPayload-compatible ``.npz`` via
       :func:`c2a.codegen.shared.npz_schema.write_codes_npz` with
       ``model_type="kinematics_vae"``.

Outputs:
    ``<out_dir>/best_codes.npz``  — the codes-npz (downstream consumer).
    ``<out_dir>/tokenizer.pkl``   — pickled (PCA, KMeans) for re-use.
    ``<out_dir>/tokenize_meta.json`` — provenance: ckpt path, hyperparams,
                                       train-only cluster count, etc.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from c2a.codegen.kinematics_vae.backbone.codegen import (
    encode_balanced_clips,
    load_checkpoint,
)
from c2a.codegen.kinematics_vae.vae.bottleneck import GaussianBottleneck
from c2a.codegen.kinematics_vae.viz.run import write_vae_viz
from c2a.codegen.shared.npz_schema import write_codes_npz
from c2a.paths import get_repo_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def tokenize(
    ckpt_dir: str | Path,
    reference_data_path: str | Path,
    balanced_split_path: str | Path,
    output_path: str | Path,
    *,
    num_codes: int = 50,
    pca_dim: Optional[int] = None,
    seed: int = 0,
    n_init: int = 10,
    n_joints: int = 67,
    use_qvel: bool = False,
    window_len: int = 10,
    horizon: int = 5,
    save_tokenizer: bool = True,
    labels_csv_path: str | Path | None = None,
    write_viz: bool = True,
) -> Path:
    """Build per-frame codes for every balanced-split clip and write
    the unified codes npz.

    Args:
        ckpt_dir: Checkpoint dir from VAE pretraining (e.g.
            ``outputs/tokenizer/trained_kinematics_vae_ckpts/vae/step_0030000``).
        reference_data_path: Path to ``rodent_reference_clips.h5``.
        balanced_split_path: Path to ``rodent_balanced_splits.json``.
        output_path: Target ``.npz`` (typically ``<sweep_dir>/
            best_codes.npz``).
        num_codes: K-Means cluster count. Sized for downstream
            ``nn.Embed(num_codes, ...)``.
        pca_dim: If not ``None``, PCA-reduce to this dim before K-Means.
            Reduces noise + speeds up clustering on the 24-D VAE latent
            (typical: 8 or 12; matches KPMS practice of low-D state space).
        seed: Random seed for K-Means initialization.
        n_init: Number of K-Means restarts.

    Returns:
        Resolved Path of the written ``best_codes.npz``.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load split for downstream subsetting.
    with open(balanced_split_path) as f:
        splits = json.load(f)
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    all_indices = np.asarray(
        sorted(set(int(i) for i in train_indices) | set(int(i) for i in test_indices)),
        dtype=np.int64,
    )

    bottleneck = GaussianBottleneck()
    modules, params = load_checkpoint(
        ckpt_dir,
        bottleneck=bottleneck,
        n_joints=n_joints,
        use_qvel=use_qvel,
        window_len=window_len,
        horizon=horizon,
    )
    encoded = encode_balanced_clips(
        reference_data_path,
        balanced_split_path,
        modules,
        params["encoder"],
        params["bottleneck"],
        n_joints=n_joints,
        use_qvel=use_qvel,
        window_len=window_len,
        horizon=horizon,
        deterministic=True,
        bottleneck_kwargs={"sigma_max": 0.0},
        rng_seed=int(seed),
    )

    # z is already in sorted-union order: encoded.clip_indices = all_indices.
    if not np.array_equal(encoded.clip_indices, all_indices):
        raise RuntimeError(
            "encode_balanced_clips returned clip_indices that don't match "
            "the sorted balanced-split union — refusing to fit K-Means on "
            "misaligned latents."
        )
    z = encoded.mean  # (n_clips, n_frames, latent_dim) — μ when deterministic
    n_clips, n_frames, latent_dim = z.shape
    log.info("Encoded %d clips × %d frames × %d-D latent", n_clips, n_frames, latent_dim)

    # ── Fit transform on TRAIN frames only to avoid leakage ──────────────────
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices.tolist())}
    train_positions = [idx_to_pos[int(c)] for c in train_indices.tolist()]
    train_z = z[train_positions].reshape(-1, latent_dim)
    log.info("Fitting tokenizer on %d train frames", train_z.shape[0])

    pca: Optional[PCA] = None
    if pca_dim is not None and pca_dim > 0:
        pca = PCA(n_components=int(pca_dim), random_state=int(seed))
        train_features = pca.fit_transform(train_z)
        log.info(
            "PCA: %d -> %d dims (explained variance: %.4f)",
            latent_dim, pca_dim,
            float(np.sum(pca.explained_variance_ratio_)),
        )
    else:
        train_features = train_z

    kmeans = KMeans(
        n_clusters=int(num_codes),
        random_state=int(seed),
        n_init=int(n_init),
    )
    kmeans.fit(train_features)
    log.info(
        "K-Means: K=%d, inertia=%.6f, train uniques=%d",
        num_codes,
        float(kmeans.inertia_),
        int(np.unique(kmeans.labels_).size),
    )

    # ── Predict codes for ALL frames in sorted-union order ───────────────────
    z_flat = z.reshape(-1, latent_dim)
    if pca is not None:
        feat_flat = pca.transform(z_flat)
    else:
        feat_flat = z_flat
    codes_flat = kmeans.predict(feat_flat).astype(np.int32)
    all_codes = codes_flat.reshape(n_clips, n_frames)

    # ── Subset to caller-order train/test ────────────────────────────────────
    train_codes = np.stack(
        [all_codes[idx_to_pos[int(c)]] for c in train_indices.tolist()], axis=0,
    )
    test_codes = np.stack(
        [all_codes[idx_to_pos[int(c)]] for c in test_indices.tolist()], axis=0,
    )

    # ── Write the codes npz via the chokepoint writer ────────────────────────
    out = write_codes_npz(
        output_path,
        train_codes=train_codes,
        test_codes=test_codes,
        train_indices=train_indices,
        test_indices=test_indices,
        num_states=int(num_codes),
        model_type="kinematics_vae",
        all_codes=all_codes,
        all_indices=all_indices,
        mean_duration=_estimate_mean_duration(all_codes),
        extra={
            "ckpt_dir": str(Path(ckpt_dir).resolve()),
            "latent_dim": np.int32(int(latent_dim)),
            "pca_dim": np.int32(int(pca_dim) if pca_dim is not None else -1),
            "kmeans_seed": np.int32(int(seed)),
            "kmeans_inertia": np.float64(float(kmeans.inertia_)),
        },
    )

    if save_tokenizer:
        tokenizer_path = output_path.parent / "tokenizer.pkl"
        with tokenizer_path.open("wb") as f:
            pickle.dump({"pca": pca, "kmeans": kmeans}, f)
        log.info("Saved tokenizer to %s", tokenizer_path)

    meta_path = output_path.parent / "tokenize_meta.json"
    meta = {
        "ckpt_dir": str(Path(ckpt_dir).resolve()),
        "num_codes": int(num_codes),
        "pca_dim": int(pca_dim) if pca_dim is not None else None,
        "kmeans_seed": int(seed),
        "kmeans_n_init": int(n_init),
        "kmeans_inertia": float(kmeans.inertia_),
        "train_uniques": int(np.unique(train_codes).size),
        "test_uniques": int(np.unique(test_codes).size),
        "model_type": "kinematics_vae",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Saved tokenize metadata to %s", meta_path)

    if write_viz:
        labels_csv_path = labels_csv_path or (
            get_repo_root() / "data" / "rodent" / "rodent_labels.csv"
        )
        try:
            write_vae_viz(
                output_path.parent,
                z=z,                            # (n_clips, n_frames, latent_dim)
                codes=all_codes,                # (n_clips, n_frames)
                clip_indices=all_indices,
                train_indices=train_indices,
                num_codes=int(num_codes),
                labels_csv_path=labels_csv_path,
                seed=int(seed),
            )
        except Exception as e:
            log.warning("VAE viz step failed (non-fatal): %s", e)

    return out


def _estimate_mean_duration(all_codes: np.ndarray) -> float:
    """Mean run-length of identical codes across all clips."""
    durations: list[int] = []
    for clip in all_codes:
        if clip.size == 0:
            continue
        changes = np.where(np.diff(clip) != 0)[0] + 1
        segments = np.split(clip, changes)
        durations.extend(int(s.size) for s in segments)
    return float(np.mean(durations)) if durations else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize VAE latents via K-Means (+ optional PCA)."
    )
    parser.add_argument("--ckpt-dir", required=True, type=str)
    parser.add_argument("--reference-data", default=None, type=str)
    parser.add_argument("--balanced-split", default=None, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--num-codes", type=int, default=50)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    args = parser.parse_args()

    repo = get_repo_root()
    ref = args.reference_data or str(repo / "data/rodent/rodent_reference_clips.h5")
    split = args.balanced_split or str(repo / "data/rodent/rodent_balanced_splits.json")
    tokenize(
        ckpt_dir=args.ckpt_dir,
        reference_data_path=ref,
        balanced_split_path=split,
        output_path=args.output,
        num_codes=args.num_codes,
        pca_dim=args.pca_dim,
        seed=args.seed,
        n_init=args.n_init,
    )


if __name__ == "__main__":
    main()
