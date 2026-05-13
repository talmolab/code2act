"""Top-K band figures + mp4s for the product-space VQVAE (PS-VQVAE) variant.

PS-VQVAE has more to show than VAE/VQVAE because each frame has a
**tuple** of per-group codes plus a folded product code. The default
band layout is::

    GT           (4-class behavior)
    product K=∏   (the folded code, default)
    group 0 K_0   (root,  with --all-groups)
    group 1 K_1   (axial, with --all-groups)
    group 2 K_2   (limb,  with --all-groups)

CLI knobs:

* ``--all-groups`` — append a band row per body group
  (rows = GT + product + each per-group code).
* ``--use-group N`` — render the per-group code for group N **instead**
  of the product code (useful for inspecting a single factor).
* (default, neither flag) — single product-code band, parity with VAE/VQVAE.

Usage::

    python -m c2a.codegen.kinematics_vae.ps_vqvae.render_topk_bands \\
        --codes-npz trained_kinematics_vae_ckpts/ps_vqvae/best_codes.npz \\
        --output-dir trained_kinematics_vae_ckpts/ps_vqvae/ \\
        --all-groups --format both
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


_GROUP_LABELS = ("root", "axial", "limb")


def _ps_vqvae_rows_factory(
    *, all_groups: bool, use_group: int | None,
):
    """Build a ``model_rows_factory`` closure for the selected mode.

    The closure must accept ``(clip_codes, extras)`` and return
    ``list[ModelRow]``. ``clip_codes`` here is the *folded product code*
    sequence (shape ``(T,)``). Per-group codes are pulled from
    ``extras['ps_per_group_codes']`` and the running balanced-clip
    position is inferred by matching the folded sequence against
    ``extras['all_codes']``.
    """
    if all_groups and use_group is not None:
        raise ValueError("Pass at most one of --all-groups, --use-group.")

    def factory(clip_codes: np.ndarray, extras: dict) -> list[ModelRow]:
        sizes = [int(x) for x in extras["ps_group_codebook_sizes"]]
        per_group = np.asarray(extras["ps_per_group_codes"], dtype=np.int32)
        G = len(sizes)
        if per_group.shape[-1] != G:
            raise RuntimeError(
                f"ps_per_group_codes last axis {per_group.shape[-1]} != G={G}"
            )

        # Locate this clip's row in the balanced-clip ordering of
        # ps_per_group_codes by exact-match on the folded code sequence.
        all_codes = np.asarray(extras["all_codes"], dtype=np.int32)
        row_matches = np.all(all_codes == clip_codes[None], axis=1)
        n_match = int(row_matches.sum())
        if n_match != 1:
            raise RuntimeError(
                f"Expected exactly 1 row of all_codes matching this clip's "
                f"product sequence; got {n_match}."
            )
        row = int(np.argmax(row_matches))
        per_group_clip = per_group[row]  # (T, G)

        K_total = int(np.prod(sizes))
        rows: list[ModelRow] = []
        if use_group is not None:
            g = int(use_group)
            if not 0 <= g < G:
                raise ValueError(f"--use-group {g} out of [0, {G})")
            label = (
                f"PS-VQVAE g{g} ({_GROUP_LABELS[g] if g < len(_GROUP_LABELS) else 'g'})  "
                f"K={sizes[g]}"
            )
            rows.append((label, per_group_clip[:, g].astype(np.int32), sizes[g]))
            return rows

        # Default: product code as the headline row.
        rows.append((
            f"PS-VQVAE product  K={K_total}",
            clip_codes.astype(np.int32),
            K_total,
        ))
        if all_groups:
            for g in range(G):
                name = _GROUP_LABELS[g] if g < len(_GROUP_LABELS) else f"g{g}"
                rows.append((
                    f"PS-VQVAE g{g} ({name})  K={sizes[g]}",
                    per_group_clip[:, g].astype(np.int32),
                    sizes[g],
                ))
        return rows

    return factory


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
        "--all-groups", action="store_true",
        help="Render product code + a band per body group.",
    )
    parser.add_argument(
        "--use-group", type=int, default=None,
        help="Render only the code for body group N (instead of product).",
    )
    parser.add_argument(
        "--variant-label", default=None,
        help="Override the file-name prefix (default depends on flag).",
    )
    args = parser.parse_args()

    if args.variant_label is not None:
        variant_label = args.variant_label
    elif args.use_group is not None:
        variant_label = f"ps_vqvae_group{int(args.use_group)}"
    elif args.all_groups:
        variant_label = "ps_vqvae_all_groups"
    else:
        variant_label = "ps_vqvae_product"

    factory = _ps_vqvae_rows_factory(
        all_groups=args.all_groups, use_group=args.use_group,
    )

    render_topk_bands(
        codes_npz_path=args.codes_npz,
        model_rows_factory=factory,
        variant_label=variant_label,
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
