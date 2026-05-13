"""Top-K band figures + mp4s for the kinematics-VAE (k-means) variant.

Renders the GT behavior band + the kinematics-VAE k-means code band for
3 auto-selected clips, both as filmstrip PNGs and as animated MP4s
(skeleton + bands + cursor). Mirrors :mod:`c2a.codegen.shimoseq.
render_topk_bands` exactly, except there's one model row per figure
(this variant has a single fit, not a top-K sweep).

Usage::

    python -m c2a.codegen.kinematics_vae.vae.render_topk_bands \\
        --codes-npz outputs/tokenizer/trained_kinematics_vae_ckpts/vae/best_codes.npz \\
        --output-dir outputs/tokenizer/trained_kinematics_vae_ckpts/vae/ \\
        --format both
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from c2a.codegen.kinematics_vae.render_common import (
    ModelRow,
    load_fk_keypoints,
    render_topk_bands,
)
from c2a.codegen.kinematics_vae.render_top_codes import (
    render_top_codes_movement,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _vae_model_rows(clip_codes: np.ndarray, extras: dict) -> list[ModelRow]:
    """One band: the post-hoc k-means code on the frozen VAE latent."""
    K = int(extras["num_states"])
    label = f"kinematics-VAE  k-means K={K}"
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
    parser.add_argument(
        "--n-top", type=int, default=20,
        help="Number of top codes to include in the movement video.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of overlaid bouts per cell in the movement video.",
    )
    parser.add_argument(
        "--skip-bands", action="store_true",
        help="Skip the band figures + clip MP4s (only emit the movement video).",
    )
    parser.add_argument(
        "--skip-top-codes-movement", action="store_true",
        help="Skip the top-K codes movement grid video.",
    )
    args = parser.parse_args()

    # FK once, share across both renderers.
    keypoints, kp_names, _edges = load_fk_keypoints(
        reference_data_path=args.reference_data,
        stac_xml_path=args.stac_xml,
        balanced_split_path=args.balanced_split,
    )

    if not args.skip_bands:
        render_topk_bands(
            codes_npz_path=args.codes_npz,
            model_rows_factory=_vae_model_rows,
            variant_label="kinematics_vae",
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
            keypoints=keypoints,
            kp_names=kp_names,
        )

    if not args.skip_top_codes_movement:
        render_top_codes_movement(
            codes_npz_path=args.codes_npz,
            output_dir=args.output_dir,
            variant_label="kinematics-VAE",
            n_top=args.n_top,
            n_samples_per_code=args.n_samples,
            keypoints=keypoints,
            kp_names=kp_names,
        )


if __name__ == "__main__":
    main()
