"""Materialize per-frame PS-VQVAE codes for every balanced-split clip.

The PS-VQVAE bottleneck emits per-group codes ``(N, G)`` from
``aux_metrics["codes"]``. Paper convention (VQ-MAP p.5 l.177) keeps
them as a tuple ``(k^{(1)}, ..., k^{(G)})``. For the c2a downstream
``MoSeqImitation`` decoder, which uses a single ``nn.Embed(num_codes,
...)`` lookup, we **fold** the tuple into a product index::

    product_idx = sum_{g} code_g * prod_{h > g} K_h

This is a stable mixed-radix encoding with the FIRST group as the most
significant digit — i.e. CB1 changes occupy contiguous blocks of
``prod_{h>0} K_h`` consecutive product codes. Downstream consumers
that want to recover the per-group codes can divmod the product idx
by the cumulative-product table emitted in the npz's ``extra`` field.

Outputs:
    ``<out_dir>/best_codes.npz`` — CodesPayload-compatible npz with
        ``model_type="ps_vqvae"`` and product-folded codes.
    ``<out_dir>/tokenize_meta.json`` — provenance: per-group
        codebook sizes, fold formula, ckpt path.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from c2a.codegen.kinematics_vae.backbone.codegen import (
    encode_balanced_clips,
    load_checkpoint,
)
from c2a.codegen.kinematics_vae.ps_vqvae.bottleneck import PSVQVAEBottleneck
from c2a.codegen.kinematics_vae.viz.run import write_ps_vqvae_viz
from c2a.codegen.shared.npz_schema import write_codes_npz
from c2a.codegen.kinematics_vae.vae.tokenize import _estimate_mean_duration
from c2a.paths import get_repo_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def fold_to_product_index(
    codes: np.ndarray, group_codebook_sizes: Sequence[int]
) -> np.ndarray:
    """Mixed-radix fold per-group codes into a single product index.

    Args:
        codes: ``(..., G)`` integer per-group codes; ``codes[..., g]
            in [0, K_g)``.
        group_codebook_sizes: ``(K_1, ..., K_G)``.

    Returns:
        ``(...,)`` int32 product index, ``in [0, prod(K_g))``. The
        first group is the most significant digit.
    """
    K = list(int(k) for k in group_codebook_sizes)
    if codes.shape[-1] != len(K):
        raise ValueError(
            f"codes has G={codes.shape[-1]} but group_codebook_sizes "
            f"has {len(K)} entries."
        )
    # Place values: pv[g] = prod(K[g+1:])
    pv = [1] * len(K)
    acc = 1
    for g in range(len(K) - 1, -1, -1):
        pv[g] = acc
        acc *= K[g]
    out = np.zeros(codes.shape[:-1], dtype=np.int64)
    for g, p in enumerate(pv):
        out = out + codes[..., g].astype(np.int64) * p
    total = int(np.prod(K))
    if out.max(initial=0) >= total or out.min(initial=0) < 0:
        raise RuntimeError(
            f"Product fold produced indices outside [0, {total}); "
            f"check per-group codes are in range."
        )
    return out.astype(np.int32)


def tokenize(
    ckpt_dir: str | Path,
    reference_data_path: str | Path,
    balanced_split_path: str | Path,
    output_path: str | Path,
    *,
    group_dims: Sequence[int],
    group_codebook_sizes: Sequence[int],
    n_joints: int = 67,
    use_qvel: bool = False,
    window_len: int = 10,
    horizon: int = 5,
    labels_csv_path: str | Path | None = None,
    write_viz: bool = True,
) -> Path:
    """Encode every clip frame and write the unified codes npz."""
    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(balanced_split_path) as f:
        splits = json.load(f)
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    all_indices = np.asarray(
        sorted(set(int(i) for i in train_indices)
               | set(int(i) for i in test_indices)),
        dtype=np.int64,
    )

    bottleneck = PSVQVAEBottleneck(
        group_dims=tuple(int(d) for d in group_dims),
        group_codebook_sizes=tuple(int(K) for K in group_codebook_sizes),
    )
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
        bn_buffers=params.get("bottleneck_buffers"),
        n_joints=n_joints,
        use_qvel=use_qvel,
        window_len=window_len,
        horizon=horizon,
        deterministic=True,
    )
    if "codes" not in encoded.aux:
        raise RuntimeError(
            "PSVQVAEBottleneck did not emit per-window 'codes' in aux_metrics."
        )
    per_group_codes = encoded.aux["codes"]   # (n_clips, n_frames, G)
    if per_group_codes.ndim != 3 or per_group_codes.shape[-1] != len(group_codebook_sizes):
        raise RuntimeError(
            f"PS-VQVAE per-group codes have shape {per_group_codes.shape}; "
            f"expected (n_clips, n_frames, G={len(group_codebook_sizes)})."
        )

    K_total = int(np.prod([int(K) for K in group_codebook_sizes]))
    all_codes = fold_to_product_index(per_group_codes, group_codebook_sizes)
    log.info(
        "PS-VQVAE codes: (n_clips=%d, n_frames=%d, G=%d) -> product K=%d; "
        "uniques used=%d/%d",
        per_group_codes.shape[0], per_group_codes.shape[1],
        per_group_codes.shape[2], K_total,
        int(np.unique(all_codes).size), K_total,
    )

    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices.tolist())}
    train_codes = np.stack(
        [all_codes[idx_to_pos[int(c)]] for c in train_indices.tolist()], axis=0,
    )
    test_codes = np.stack(
        [all_codes[idx_to_pos[int(c)]] for c in test_indices.tolist()], axis=0,
    )

    out = write_codes_npz(
        output_path,
        train_codes=train_codes,
        test_codes=test_codes,
        train_indices=train_indices,
        test_indices=test_indices,
        num_states=K_total,
        model_type="ps_vqvae",
        all_codes=all_codes,
        all_indices=all_indices,
        mean_duration=_estimate_mean_duration(all_codes),
        extra={
            "ckpt_dir": str(Path(ckpt_dir).resolve()),
            "ps_group_dims": np.asarray(group_dims, dtype=np.int32),
            "ps_group_codebook_sizes": np.asarray(
                group_codebook_sizes, dtype=np.int32,
            ),
            # Per-group codes preserved for downstream visualization
            # (paper convention reports CB1/CB2 separately).
            "ps_per_group_codes": per_group_codes.astype(np.int32),
        },
    )

    meta_path = output_path.parent / "tokenize_meta.json"
    meta = {
        "ckpt_dir": str(Path(ckpt_dir).resolve()),
        "group_dims": [int(d) for d in group_dims],
        "group_codebook_sizes": [int(K) for K in group_codebook_sizes],
        "product_num_codes": K_total,
        "train_uniques": int(np.unique(train_codes).size),
        "test_uniques": int(np.unique(test_codes).size),
        "model_type": "ps_vqvae",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Saved tokenize metadata to %s", meta_path)

    if write_viz:
        labels_csv_path = labels_csv_path or (
            get_repo_root() / "data" / "rodent" / "rodent_labels.csv"
        )
        try:
            write_ps_vqvae_viz(
                output_path.parent,
                z_e=encoded.mean,                    # (n_clips, n_frames, latent_dim)
                per_group_codes=per_group_codes,     # (n_clips, n_frames, G)
                product_codes=all_codes,             # (n_clips, n_frames) folded
                clip_indices=all_indices,
                train_indices=train_indices,
                group_dims=group_dims,
                group_codebook_sizes=group_codebook_sizes,
                num_codes_product=K_total,
                labels_csv_path=labels_csv_path,
            )
        except Exception as e:
            log.warning("PS-VQVAE viz step failed (non-fatal): %s", e)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Materialize per-frame PS-VQVAE codes."
    )
    parser.add_argument("--ckpt-dir", required=True, type=str)
    parser.add_argument("--reference-data", default=None, type=str)
    parser.add_argument("--balanced-split", default=None, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--group-dims", type=int, nargs="+", required=True,
        help="Per-group code_dim; sums must equal latent_dim.",
    )
    parser.add_argument(
        "--group-codebook-sizes", type=int, nargs="+", required=True,
        help="Per-group codebook size K_g.",
    )
    args = parser.parse_args()

    repo = get_repo_root()
    ref = args.reference_data or str(repo / "data/rodent/rodent_reference_clips.h5")
    split = args.balanced_split or str(repo / "data/rodent/rodent_balanced_splits.json")
    tokenize(
        ckpt_dir=args.ckpt_dir,
        reference_data_path=ref,
        balanced_split_path=split,
        output_path=args.output,
        group_dims=args.group_dims,
        group_codebook_sizes=args.group_codebook_sizes,
    )


if __name__ == "__main__":
    main()
