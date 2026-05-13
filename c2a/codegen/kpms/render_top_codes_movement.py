"""Top-N syllable "movement" video grid for a KPMS ``best_codes.npz``.

Mirrors the kinematics-VAE family's ``top20_codes_movement.mp4`` artifact
for KPMS syllables (raw or merged). Grid size auto-derives from ``K`` so
the figure fits whatever alphabet size the merged KPMS produced (typical
range: 5–30 codes).

The actual animation work is delegated to
:func:`c2a.codegen.kinematics_vae.render_top_codes.render_top_codes_movement`
— the renderer is variant-agnostic since every variant in this repo
writes the same :data:`c2a.codegen.shared.npz_schema.CodesPayload`
schema (``all_codes``, ``train_indices``, ``test_indices``,
``num_states``).

Usage::

    # Raw KPMS top syllables
    python -m c2a.codegen.kpms.render_top_codes_movement \\
        --codes-npz outputs/tokenizer/kpms_sweep/best_codes.npz \\
        --output-dir outputs/tokenizer/kpms_sweep/

    # KPMS auto-merged best rank
    python -m c2a.codegen.kpms.render_top_codes_movement \\
        --codes-npz outputs/tokenizer/kpms_sweep/auto_merged/M1/codes_merged.npz \\
        --output-dir outputs/tokenizer/kpms_sweep/auto_merged/M1/ \\
        --variant-label "KPMS-merged M1"
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

from c2a.codegen.kinematics_vae.render_common import load_fk_keypoints
from c2a.codegen.kinematics_vae.render_top_codes import (
    render_top_codes_movement,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _auto_grid(k_effective: int, *, cols: int = 5) -> tuple[int, int]:
    """Pick ``(grid_rows, grid_cols)`` to fit exactly ``k_effective`` cells
    on a fixed-cols grid (cols=5 by default to match the kinematics-VAE
    videos). Extra grid cells past ``k_effective`` stay empty.

    Examples (cols=5):
        k=5  -> (1, 5)
        k=10 -> (2, 5)
        k=13 -> (3, 5)  (last row has 2 blank cells)
        k=15 -> (3, 5)
        k=20 -> (4, 5)
    """
    if k_effective <= 0:
        raise ValueError(f"k_effective must be > 0, got {k_effective}")
    grid_rows = max(1, math.ceil(k_effective / cols))
    return grid_rows, cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--codes-npz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--variant-label", default=None,
        help="Suptitle prefix (default auto-detected from npz model_type).",
    )
    parser.add_argument(
        "--n-top", type=int, default=20,
        help="Cap on number of top codes shown. Auto-clamped to K from "
             "the npz if K < n_top.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3,
        help="Number of overlaid bouts per cell (one rodent each).",
    )
    parser.add_argument(
        "--grid-cols", type=int, default=5,
        help="Number of columns in the figure grid (default 5).",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--reference-data", type=Path, default=None)
    parser.add_argument("--stac-xml", type=Path, default=None)
    parser.add_argument("--balanced-split", type=Path, default=None)
    args = parser.parse_args()

    # Peek at npz to determine K and auto-pick grid.
    with np.load(args.codes_npz, allow_pickle=True) as data:
        K = int(np.asarray(data["num_states"]))
        model_type = str(np.asarray(data["model_type"]))
        is_merged = bool(np.asarray(data.get("is_merged", False)))
    n_top_effective = min(args.n_top, K)
    grid_rows, grid_cols = _auto_grid(n_top_effective, cols=args.grid_cols)
    log.info(
        f"KPMS movement video: K={K} model={model_type} merged={is_merged}; "
        f"n_top={n_top_effective}, grid={grid_rows}x{grid_cols}"
    )

    if args.variant_label is None:
        if is_merged:
            args.variant_label = "KPMS (merged syllables)"
        else:
            args.variant_label = "KPMS (raw syllables)"

    keypoints, kp_names, _edges = load_fk_keypoints(
        reference_data_path=args.reference_data,
        stac_xml_path=args.stac_xml,
        balanced_split_path=args.balanced_split,
    )

    render_top_codes_movement(
        codes_npz_path=args.codes_npz,
        output_dir=args.output_dir,
        variant_label=args.variant_label,
        n_top=n_top_effective,
        n_samples_per_code=args.n_samples,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        fps=args.fps,
        keypoints=keypoints,
        kp_names=kp_names,
    )


if __name__ == "__main__":
    main()
