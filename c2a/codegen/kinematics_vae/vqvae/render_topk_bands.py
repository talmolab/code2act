"""Top-K band figures + mp4s for the vanilla-VQVAE variant.

Same layout as :mod:`c2a.codegen.kinematics_vae.vae.render_topk_bands`,
but the model row is the learned VQ code instead of a post-hoc k-means
assignment.

Usage::

    python -m c2a.codegen.kinematics_vae.vqvae.render_topk_bands \\
        --codes-npz trained_kinematics_vae_ckpts/vqvae/best_codes.npz \\
        --output-dir trained_kinematics_vae_ckpts/vqvae/ \\
        --format both
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from c2a.codegen.kinematics_vae.render_common import (
    ModelRow,
    render_topk_bands,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _vqvae_model_rows(clip_codes: np.ndarray, extras: dict) -> list[ModelRow]:
    """One band: the learned VQ codebook assignment per frame."""
    K = int(extras["num_states"])
    label = f"VQVAE  K={K}"
    return [(label, clip_codes.astype(np.int32), K)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--codes-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--clips", type=int, nargs=3, default=None,
        help="Override auto-selected clips (3 ints).",
    )
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--balanced-split", type=Path, default=None)
    parser.add_argument("--reference-data", type=Path, default=None)
    parser.add_argument("--stac-xml", type=Path, default=None)
    parser.add_argument("--clip-range", type=int, nargs=2, default=[0, 841])
    parser.add_argument("--csv-to-h5-subsample", type=int, default=2)
    parser.add_argument("--n-frames", type=int, default=250)
    parser.add_argument("--n-snapshots", type=int, default=8)
    parser.add_argument(
        "--format", dest="output_format",
        choices=("video", "png", "both"), default="both",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    render_topk_bands(
        codes_npz_path=args.codes_npz,
        model_rows_factory=_vqvae_model_rows,
        variant_label="vqvae",
        output_dir=args.output_dir,
        clips=args.clips,
        labels_path=args.labels_path,
        balanced_split_path=args.balanced_split,
        reference_data_path=args.reference_data,
        stac_xml_path=args.stac_xml,
        labeled_clip_range=tuple(args.clip_range),
        csv_to_h5_subsample=args.csv_to_h5_subsample,
        n_frames=args.n_frames,
        n_snapshots=args.n_snapshots,
        output_format=args.output_format,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
