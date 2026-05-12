"""Generate shMoSeq codes in the CodesPayload-compatible npz schema.

The shMoSeq sweep writes ``best_states.npz`` keyed by ``clip_<id>`` (one
1D state array per clip), plus ``sweep_results.json`` carrying the
selection metadata. The downstream MoSeq decoder, however, expects the
same flat ``train_codes``/``test_codes`` schema that KPMS emits. This
script bridges the two:

1. Load ``sweep_results.json`` → grab ``best_model``.
2. Load ``best_states.npz`` → per-clip state arrays.
3. Subset / re-order against the balanced train/test split.
4. Write the unified codes npz via
   :func:`c2a.codegen.shared.npz_schema.write_codes_npz`.

Usage::

    python -m c2a.codegen.shimoseq.generate_codes \\
        --sweep-dir outputs/shimoseq_sweep \\
        --balanced-split data/rodent/rodent_balanced_splits.json \\
        --output outputs/shimoseq_sweep/best_codes.npz

With the unified file written, the decoder trainer reads::

    kpms_config:
      code_source: shimoseq
      codes_path: outputs/shimoseq_sweep/best_codes.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from c2a.codegen.shared.npz_schema import write_codes_npz
from c2a.paths import get_repo_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def generate_codes(
    sweep_dir: str | Path,
    balanced_split_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Materialize the shMoSeq sweep's best fit as a unified codes npz.

    Args:
        sweep_dir: Directory containing ``sweep_results.json`` and
            ``best_states.npz`` (the output of
            ``c2a.codegen.shimoseq.sweep_impl.run_sweep``).
        balanced_split_path: Path to the balanced split JSON
            (``data/rodent/rodent_balanced_splits.json``).
        output_path: Target ``.npz`` path.

    Returns:
        Resolved Path of the written file.

    Raises:
        FileNotFoundError: ``sweep_results.json`` or ``best_states.npz`` missing.
        RuntimeError: clip-id mismatch between the sweep output and the
            balanced split (i.e. the sweep was run against a different split).
    """
    sweep_dir = Path(sweep_dir)
    results_path = sweep_dir / "sweep_results.json"
    states_path = sweep_dir / "best_states.npz"

    if not results_path.exists():
        raise FileNotFoundError(
            f"shimoseq sweep_results.json not found at {results_path}. Run "
            f"`python -m c2a.codegen.run_sweep --config "
            f"c2a/configs/shimoseq_sweep.yaml` first."
        )
    if not states_path.exists():
        raise FileNotFoundError(
            f"shimoseq best_states.npz not found at {states_path}. The "
            f"sweep finished but didn't write the per-clip state arrays "
            f"— check sweep logs."
        )

    with results_path.open() as f:
        results = json.load(f)
    best = results["best_model"]
    n_states = int(best["n_states"])
    trans_kappa = float(best["trans_kappa"])
    mean_duration = float(best.get("mean_duration", 0.0))
    log.info(
        "shimoseq best fit: n_states=%d, trans_kappa=%.2e, "
        "mean_duration=%.2f, seed=%s",
        n_states, trans_kappa, mean_duration, best.get("seed"),
    )

    with open(balanced_split_path) as f:
        splits = json.load(f)
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)

    # shMoSeq's best_states.npz uses ``clip_<position>`` keys where
    # ``position`` is the index into ``sorted(set(train) | set(test))``
    # — the same positional convention KPMS's syllable_loader uses (see
    # ``c2a.codegen.shimoseq.syllable_loader.load_kpms_codes`` which
    # writes ``f"clip_{ci}"`` for ``ci in range(n_clips)``). So we must
    # first build the split→position map, then look up by position.
    all_indices = sorted(set(int(c) for c in train_indices.tolist())
                         | set(int(c) for c in test_indices.tolist()))
    idx_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}

    data = dict(np.load(str(states_path), allow_pickle=False))
    # best_states.npz mixes per-clip state arrays (keys ``clip_<pos>``)
    # with scalar metadata (n_states, trans_kappa, seed, …). Filter to
    # the clip arrays only.
    clip_arrays: dict[int, np.ndarray] = {}
    for k, v in data.items():
        if not k.startswith("clip_"):
            continue
        try:
            pos = int(k.removeprefix("clip_"))
        except ValueError:
            log.warning("Skipping non-int clip key %r in %s", k, states_path)
            continue
        clip_arrays[pos] = np.asarray(v, dtype=np.int32)

    if not clip_arrays:
        raise RuntimeError(
            f"best_states.npz at {states_path} has no clip_<pos> entries; "
            f"refusing to write empty codes file."
        )
    if len(clip_arrays) != len(all_indices):
        raise RuntimeError(
            f"best_states.npz at {states_path} has {len(clip_arrays)} clip "
            f"arrays but the balanced split union has {len(all_indices)} "
            f"clips. The sweep was run against a different balanced split "
            f"than {balanced_split_path}."
        )

    # All clip arrays must have the same length (= clip frames).
    lens = {a.shape[0] for a in clip_arrays.values()}
    if len(lens) != 1:
        raise RuntimeError(
            f"shimoseq clip state arrays have inconsistent lengths {lens} "
            f"in {states_path}; the sweep was run with mixed clip_length."
        )
    n_frames = lens.pop()

    train_positions = [idx_to_pos[int(c)] for c in train_indices.tolist()]
    test_positions = [idx_to_pos[int(c)] for c in test_indices.tolist()]
    missing_train = [p for p in train_positions if p not in clip_arrays]
    missing_test = [p for p in test_positions if p not in clip_arrays]
    if missing_train or missing_test:
        raise RuntimeError(
            f"shimoseq best_states.npz at {states_path} is missing "
            f"positions required by the balanced split: "
            f"train_missing={missing_train[:10]} ({len(missing_train)} total), "
            f"test_missing={missing_test[:10]} ({len(missing_test)} total)."
        )

    train_codes = np.stack(
        [clip_arrays[p] for p in train_positions], axis=0,
    ).astype(np.int32)
    test_codes = np.stack(
        [clip_arrays[p] for p in test_positions], axis=0,
    ).astype(np.int32)

    # ``num_states`` from sweep_results.json is authoritative — it reflects
    # the AR-HHMM's nominal cluster count, not just the active states. The
    # downstream embedding table must be sized for the full range.
    max_seen = max(int(train_codes.max()), int(test_codes.max()))
    if max_seen >= n_states:
        # Defensive: should not happen if the sweep is well-behaved.
        raise RuntimeError(
            f"shimoseq codes have max state {max_seen} but sweep_results "
            f"declares n_states={n_states}. Refusing to write an undersized "
            f"embedding table."
        )

    # all_codes follows the KPMS convention (see
    # ``c2a.codegen.kpms.generate_codes`` and the loader contract in
    # ``c2a.codegen.shared.npz_schema``): row ``i`` corresponds to the
    # clip at *sorted-union position ``i``* (NOT caller-train-then-test
    # order). Several downstream analyses (``c2a.experiments.run_gait_dynamics``,
    # ``c2a.experiments.run_inception_distance``, …) index
    # ``all_codes[clip_idx, ...]`` with a sorted-union position. Emitting
    # a different ordering would silently misalign every cross-tokenizer
    # plot that compares shMoSeq with KPMS or future kinematics-VAE
    # variants. shMoSeq's clip_arrays are already keyed by sorted-union
    # position via ``syllable_loader.load_kpms_codes``, so this is a
    # cheap restack.
    all_codes = np.stack(
        [clip_arrays[p] for p in range(len(all_indices))], axis=0,
    ).astype(np.int32)

    output_path = Path(output_path)
    out = write_codes_npz(
        output_path,
        train_codes=train_codes,
        test_codes=test_codes,
        train_indices=train_indices,
        test_indices=test_indices,
        num_states=n_states,
        model_type="shimoseq",
        all_codes=all_codes,
        all_indices=np.asarray(all_indices, dtype=np.int64),
        kappa=trans_kappa,
        mean_duration=mean_duration,
        extra={
            "seed": np.int32(int(best.get("seed", -1))),
            "median_duration": np.float64(float(best.get("median_duration", 0.0))),
            "marginal_loglik": np.float64(
                float(best.get("marginal_loglik", 0.0))
            ),
        },
    )
    log.info(
        "Wrote shimoseq codes to %s: train %s, test %s, num_codes=%d",
        out, train_codes.shape, test_codes.shape, n_states,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate shMoSeq codes in the unified codes-npz schema"
    )
    parser.add_argument(
        "--sweep-dir", type=str, required=True,
        help="Directory containing sweep_results.json + best_states.npz",
    )
    parser.add_argument(
        "--balanced-split", type=str, required=True,
        help="Path to balanced split JSON",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .npz path (default: <sweep-dir>/best_codes.npz)",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = get_repo_root() / sweep_dir
    output_path = args.output or (sweep_dir / "best_codes.npz")

    generate_codes(
        sweep_dir=sweep_dir,
        balanced_split_path=args.balanced_split,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
