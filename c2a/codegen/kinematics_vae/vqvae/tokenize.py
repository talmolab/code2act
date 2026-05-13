"""Materialize per-frame VQ-VAE codes for every balanced-split clip.

The VQ-VAE bottleneck already emits a discrete code per encoded window
(``aux_metrics["codes"]``); this script just routes them through the
per-frame padding pipeline in :mod:`c2a.codegen.kinematics_vae.backbone.
codegen` and the unified codes-npz writer.

Unlike :mod:`c2a.codegen.kinematics_vae.vae.tokenize`, no K-Means / PCA
step is needed — the discrete representation is built into the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from c2a.codegen.kinematics_vae.backbone.codegen import (
    encode_balanced_clips,
    load_checkpoint,
)
from c2a.codegen.kinematics_vae.vqvae.bottleneck import VQVAEBottleneck
from c2a.codegen.kinematics_vae.viz.run import write_vqvae_viz
from c2a.codegen.shared.npz_schema import write_codes_npz
from c2a.codegen.kinematics_vae.vae.tokenize import _estimate_mean_duration
from c2a.paths import get_repo_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def tokenize(
    ckpt_dir: str | Path,
    reference_data_path: str | Path,
    balanced_split_path: str | Path,
    output_path: str | Path,
    *,
    codebook_size: int,
    code_dim: int,
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

    bottleneck = VQVAEBottleneck(codebook_size=codebook_size, code_dim=code_dim)
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
        bottleneck_kwargs=None,
    )

    if "codes" not in encoded.aux:
        raise RuntimeError(
            "VQVAEBottleneck did not emit per-window 'codes' in aux_metrics; "
            "cannot tokenize. Check the bottleneck implementation."
        )

    all_codes = encoded.aux["codes"].astype(np.int32)   # (n_clips, n_frames)
    log.info(
        "Decoded codes: shape=%s, train uniques=%d / K=%d",
        all_codes.shape,
        int(np.unique(all_codes).size),
        codebook_size,
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
        num_states=int(codebook_size),
        model_type="vqvae",
        all_codes=all_codes,
        all_indices=all_indices,
        mean_duration=_estimate_mean_duration(all_codes),
        extra={
            "ckpt_dir": str(Path(ckpt_dir).resolve()),
            "code_dim": np.int32(int(code_dim)),
        },
    )

    meta_path = output_path.parent / "tokenize_meta.json"
    meta = {
        "ckpt_dir": str(Path(ckpt_dir).resolve()),
        "codebook_size": int(codebook_size),
        "code_dim": int(code_dim),
        "train_uniques": int(np.unique(train_codes).size),
        "test_uniques": int(np.unique(test_codes).size),
        "model_type": "vqvae",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Saved tokenize metadata to %s", meta_path)

    if write_viz:
        labels_csv_path = labels_csv_path or (
            get_repo_root() / "data" / "rodent" / "rodent_labels.csv"
        )
        # Codebook lives under params["bottleneck"]["params"]["quantizer"]["codebook"]
        # — the {"params": ...} outer wrapper is the Flax variables shape.
        try:
            cb_subtree = params["bottleneck"]["params"]
            codebook = np.asarray(cb_subtree["quantizer"]["codebook"])
            write_vqvae_viz(
                output_path.parent,
                z_e=encoded.mean,           # (n_clips, n_frames, latent_dim)
                codes=all_codes,
                codebook=codebook,
                clip_indices=all_indices,
                train_indices=train_indices,
                num_codes=int(codebook_size),
                labels_csv_path=labels_csv_path,
            )
        except Exception as e:
            log.warning("VQVAE viz step failed (non-fatal): %s", e)

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Materialize per-frame VQ-VAE codes for every balanced clip."
    )
    parser.add_argument("--ckpt-dir", required=True, type=str)
    parser.add_argument("--reference-data", default=None, type=str)
    parser.add_argument("--balanced-split", default=None, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--codebook-size", type=int, required=True)
    parser.add_argument("--code-dim", type=int, default=24)
    args = parser.parse_args()

    repo = get_repo_root()
    ref = args.reference_data or str(repo / "data/rodent/rodent_reference_clips.h5")
    split = args.balanced_split or str(repo / "data/rodent/rodent_balanced_splits.json")
    tokenize(
        ckpt_dir=args.ckpt_dir,
        reference_data_path=ref,
        balanced_split_path=split,
        output_path=args.output,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
    )


if __name__ == "__main__":
    main()
