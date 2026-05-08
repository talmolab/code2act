"""Manually merge KPMS syllable states by modifying the checkpoint.

This is a *real* merge — not just relabeling.  For each group of state
indices specified, we collapse the per-state arrays in
``model_snapshots/{iter}/params`` (``Ab``, ``Q``, ``pi``, ``betas``) and
the per-frame state assignment ``states/z``, then reindex the remaining
states to a compact ``[0, new_num_states)`` range.  The result is a
valid (smaller-K) AR-HMM that's a coarsening of the original.

Compare with the upstream :func:`keypoint_moseq.apply_syllable_mapping`,
which only relabels the ``results.h5`` syllable arrays — it does not
modify ``Ab/Q/pi/betas``.  Use this script when you also want the
checkpoint itself to reflect the merge (so resampling, generation, and
downstream re-fits use the merged model).

Usage::

    python -m c2a.kpms.merge_states \\
        --checkpoint outputs/kpms_sweep/.../checkpoint.h5 \\
        --merge "1,3" --merge "4,5,12" \\
        --output  outputs/kpms_sweep/.../checkpoint_merged.h5

You can repeat ``--merge`` for as many groups as you need.  States not
in any group keep their identities.  After all merges, indices are
compacted and (by convention) sorted by usage so the lowest index is
the most frequent merged state.

By default we only merge the **final** snapshot in the checkpoint.
Use ``--all-snapshots`` to merge every snapshot (useful for resumption
diagnostics; mostly cosmetic otherwise).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# Group parsing + validation
# =============================================================================


def parse_merge_args(merge_strs: list[str]) -> list[list[int]]:
    """Parse ``--merge "1,3"`` repeats into a list-of-lists.

    Validates that no state appears in more than one group.
    """
    groups: list[list[int]] = []
    seen: set[int] = set()
    for s in merge_strs:
        try:
            ids = [int(x.strip()) for x in s.split(",")]
        except ValueError as e:
            raise ValueError(f"--merge value {s!r} is not a comma-separated int list") from e
        if len(ids) < 2:
            raise ValueError(f"--merge group must have ≥ 2 states; got {ids}")
        if any(i in seen for i in ids):
            raise ValueError(f"State appears in multiple --merge groups: {ids}")
        seen.update(ids)
        groups.append(sorted(ids))
    return groups


# =============================================================================
# Per-snapshot merge (the real model surgery)
# =============================================================================


def _state_usage(z: np.ndarray, mask: np.ndarray, num_states: int) -> np.ndarray:
    """Per-state frame counts under the mask.

    ``z`` has shape ``(N, T-nlags)``; ``mask`` has ``(N, T)``.  We align
    them by trimming mask to the last ``z.shape[1]`` columns.
    """
    assert z.ndim == 2 and mask.ndim == 2
    nlags = mask.shape[1] - z.shape[1]
    m = mask[:, nlags:].astype(bool)
    z_valid = z[m]
    counts = np.bincount(z_valid, minlength=num_states)
    return counts.astype(np.int64)


def _merge_pi(
    pi: np.ndarray, groups: list[list[int]], group_to_new: dict[int, int],
    weights_within_group: dict[tuple[int, ...], np.ndarray],
    keep: list[int], old_to_new: dict[int, int],
) -> np.ndarray:
    """Build the merged transition matrix.

    Step 1: column-collapse — for every j, sum ``pi[i, j]`` over j in a
    group into a single column for that group.
    Step 2: row-collapse — for every i in a group, take a usage-weighted
    average of the rows.
    Step 3: reindex the remaining surviving rows/columns into a compact
    range.

    Resulting ``pi_new`` is row-stochastic by construction (re-normalize
    at the end as a numerical safety net).
    """
    K = pi.shape[0]
    Knew = len(keep)
    pi_new = np.zeros((Knew, Knew), dtype=pi.dtype)
    # Build a representative per old index: i ∈ group → group's new index;
    # i not in any group → old_to_new[i].
    old_to_repr_new: dict[int, int] = dict(old_to_new)
    for g_tuple, _w in weights_within_group.items():
        new_idx = group_to_new[g_tuple]
        for j in g_tuple:
            old_to_repr_new[j] = new_idx

    for i in range(K):
        i_new = old_to_repr_new[i]
        for j in range(K):
            j_new = old_to_repr_new[j]
            # Row aggregation (i): if i ∈ group g, weight by w_i within g.
            i_group = next((g for g in groups if i in g), None)
            if i_group is None:
                w = 1.0
            else:
                w = float(weights_within_group[tuple(i_group)][i_group.index(i)])
            pi_new[i_new, j_new] += w * pi[i, j]
    # Renormalize rows (safety).
    row_sums = pi_new.sum(axis=1, keepdims=True)
    pi_new = pi_new / np.maximum(row_sums, 1e-12)
    return pi_new


def merge_snapshot(
    snapshot: dict, groups: list[list[int]], usage: np.ndarray,
) -> tuple[dict, dict[int, int]]:
    """Merge one model snapshot in-place (returns a new dict).

    Returns ``(new_snapshot, old_to_new_map)`` where the map covers
    every old state index.  States in a merged group share the new index
    of their group; lone states get their own re-indexed slot.
    """
    K = int(snapshot["params"]["Ab"].shape[0])
    in_some_group = set(s for g in groups for s in g)
    lone = [s for s in range(K) if s not in in_some_group]

    # Decide new indices.  Convention: lone states retain order, then
    # groups appended after.  We then resort by usage so lowest = most
    # frequent.
    new_order: list[list[int] | int] = [s for s in lone] + [list(g) for g in groups]
    # Compute usage for each new entry (sum within group).
    new_usage = []
    for entry in new_order:
        if isinstance(entry, list):
            new_usage.append(int(usage[entry].sum()))
        else:
            new_usage.append(int(usage[entry]))
    # Sort descending by usage.
    sorted_idx = list(np.argsort(-np.asarray(new_usage)))
    new_order_sorted = [new_order[i] for i in sorted_idx]

    # Build maps.
    old_to_new: dict[int, int] = {}
    group_to_new: dict[tuple[int, ...], int] = {}
    keep: list[int] = []  # for column-collapse only — represents "what to look at"
    for new_idx, entry in enumerate(new_order_sorted):
        if isinstance(entry, list):
            for j in entry:
                old_to_new[j] = new_idx
            group_to_new[tuple(sorted(entry))] = new_idx
            keep.append(new_idx)
        else:
            old_to_new[entry] = new_idx
            keep.append(new_idx)

    # Within-group weights (usage-normalized).
    weights_within_group: dict[tuple[int, ...], np.ndarray] = {}
    for g in groups:
        u = usage[g].astype(np.float64)
        if u.sum() == 0:
            w = np.full(len(g), 1.0 / len(g))
        else:
            w = u / u.sum()
        weights_within_group[tuple(sorted(g))] = w

    # ------- Per-state arrays -------
    Ab = snapshot["params"]["Ab"]      # (K, lat, lat*nlags+1)
    Q  = snapshot["params"]["Q"]       # (K, lat, lat)
    betas = snapshot["params"]["betas"]  # (K,)
    pi = snapshot["params"]["pi"]      # (K, K)

    Knew = len(new_order_sorted)
    Ab_new = np.empty((Knew, *Ab.shape[1:]), dtype=Ab.dtype)
    Q_new  = np.empty((Knew, *Q.shape[1:]),  dtype=Q.dtype)
    betas_new = np.zeros(Knew, dtype=betas.dtype)
    for new_idx, entry in enumerate(new_order_sorted):
        if isinstance(entry, list):
            w = weights_within_group[tuple(sorted(entry))]
            Ab_new[new_idx]   = (w[:, None, None] * Ab[entry]).sum(axis=0)
            Q_new[new_idx]    = (w[:, None, None] * Q[entry]).sum(axis=0)
            betas_new[new_idx] = betas[entry].sum()
        else:
            Ab_new[new_idx]   = Ab[entry]
            Q_new[new_idx]    = Q[entry]
            betas_new[new_idx] = betas[entry]
    # Renormalize betas (kpms convention: sum to one).
    if betas_new.sum() > 0:
        betas_new = betas_new / betas_new.sum()

    pi_new = _merge_pi(
        pi, groups, group_to_new, weights_within_group,
        keep, old_to_new,
    )

    # ------- z (per-frame state assignment) -------
    z = snapshot["states"]["z"]
    z_new = np.vectorize(old_to_new.get)(z).astype(z.dtype)

    # ------- Build merged snapshot dict -------
    new_snapshot: dict = {
        "params": {
            **{k: v for k, v in snapshot["params"].items()
               if k not in ("Ab", "Q", "pi", "betas")},
            "Ab": Ab_new,
            "Q":  Q_new,
            "pi": pi_new,
            "betas": betas_new,
        },
        "states": {
            **{k: v for k, v in snapshot["states"].items() if k != "z"},
            "z": z_new,
        },
        "hypparams": _update_num_states_in_hypparams(snapshot["hypparams"], Knew),
    }
    if "noise_prior" in snapshot:
        new_snapshot["noise_prior"] = snapshot["noise_prior"]
    if "seed" in snapshot:
        new_snapshot["seed"] = snapshot["seed"]
    return new_snapshot, old_to_new


def _update_num_states_in_hypparams(hp: dict, new_K: int) -> dict:
    """Return a deep-ish copy of ``hp`` with both ``num_states`` fields
    updated."""
    out: dict = {}
    for k, v in hp.items():
        if isinstance(v, dict):
            sub = dict(v)
            if "num_states" in sub:
                sub["num_states"] = np.int64(new_K)
            out[k] = sub
        else:
            out[k] = v
    return out


# =============================================================================
# HDF5 I/O
# =============================================================================


def _read_snapshot(f: h5py.File, snap_key: str) -> dict:
    """Read one snapshot group into a nested dict of ndarrays."""

    def _walk(g):
        out: dict = {}
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                out[k] = _walk(v)
            else:
                out[k] = v[()]
        return out

    return _walk(f[snap_key])


def _write_snapshot(f: h5py.File, snap_key: str, snap: dict) -> None:
    """Write a nested-dict snapshot under ``model_snapshots/{snap_key}``.
    Creates groups/datasets, deleting prior content under ``snap_key``."""

    if snap_key in f:
        del f[snap_key]
    grp = f.create_group(snap_key)

    def _walk(d, g):
        for k, v in d.items():
            if isinstance(v, dict):
                sub = g.create_group(k)
                _walk(v, sub)
            else:
                arr = np.asarray(v)
                g.create_dataset(k, data=arr)

    _walk(snap, grp)


def merge_checkpoint(
    in_path: Path,
    out_path: Path,
    groups: list[list[int]],
    *,
    all_snapshots: bool = False,
) -> None:
    """Top-level: read ``in_path``, merge per ``groups``, write ``out_path``.

    The output is an HDF5 file with the same schema as the input but
    with reduced ``num_states``.  ``data/`` and ``metadata/`` are copied
    through verbatim.  By default only the final snapshot is merged
    (rest are dropped); use ``all_snapshots=True`` to merge every
    snapshot in the input.
    """
    if not in_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")
    if out_path.exists():
        raise FileExistsError(
            f"Output already exists: {out_path}.  Delete it or pick a "
            f"different --output path."
        )

    log.info(f"Reading {in_path}")
    with h5py.File(in_path, "r") as fin:
        snap_keys = sorted(fin["model_snapshots"].keys(), key=int)
        if not snap_keys:
            raise RuntimeError(f"No snapshots found in {in_path}")
        last_key = snap_keys[-1]
        log.info(f"Snapshots: {snap_keys}; merging {'all' if all_snapshots else last_key!r}")

        # Read mask for usage computation (shared across all snapshots).
        mask = fin["data/mask"][()]

        # Need usage of the LAST snapshot (used to weight all merges,
        # even across snapshots — usage is essentially the same once
        # training has converged).
        last_z = fin[f"model_snapshots/{last_key}/states/z"][()]
        K = int(fin[f"model_snapshots/{last_key}/params/Ab"].shape[0])
        usage = _state_usage(last_z, mask, num_states=K)
        log.info(f"Per-state usage (final snapshot): {usage.tolist()}")

        # Pre-flight: validate that all merge state ids are in [0, K)
        for g in groups:
            for s in g:
                if s < 0 or s >= K:
                    raise ValueError(f"State {s} out of range for K={K}: groups={groups}")

        # Collect snapshots to merge.
        snapshots_to_merge = snap_keys if all_snapshots else [last_key]
        merged_snapshots: dict[str, tuple[dict, dict[int, int]]] = {}
        for sk in snapshots_to_merge:
            snap = _read_snapshot(fin, f"model_snapshots/{sk}")
            new_snap, old_to_new = merge_snapshot(snap, groups, usage)
            merged_snapshots[sk] = (new_snap, old_to_new)

        # Read everything else (data, metadata) verbatim.
        data_dict = _read_snapshot(fin, "data") if "data" in fin else {}
        metadata_dict = _read_snapshot(fin, "metadata") if "metadata" in fin else {}

    # Write output.
    log.info(f"Writing {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as fout:
        snap_root = fout.create_group("model_snapshots")
        for sk, (new_snap, old_to_new) in merged_snapshots.items():
            _write_snapshot(fout, f"model_snapshots/{sk}", new_snap)
        if data_dict:
            _write_snapshot(fout, "data", data_dict)
        if metadata_dict:
            _write_snapshot(fout, "metadata", metadata_dict)

        # Record provenance: groups + maps for downstream tools.
        prov_grp = fout.create_group("merge_provenance")
        prov_grp.attrs["source_checkpoint"] = str(in_path)
        prov_grp.attrs["original_num_states"] = K
        prov_grp.attrs["new_num_states"] = (
            len(merged_snapshots[last_key][0]["params"]["betas"])
        )
        prov_grp.attrs["all_snapshots"] = bool(all_snapshots)
        prov_grp.create_dataset("usage_input", data=usage)
        prov_grp.create_dataset(
            "groups_json",
            data=np.array(json.dumps(groups), dtype=h5py.string_dtype()),
        )
        last_map = merged_snapshots[last_key][1]
        prov_grp.create_dataset(
            "old_to_new",
            data=np.array(
                [[k, v] for k, v in sorted(last_map.items())],
                dtype=np.int64,
            ),
        )

    new_K = merged_snapshots[last_key][0]["params"]["betas"].shape[0]
    log.info(
        f"Done.  K={K} → K_new={new_K}; "
        f"merged groups={groups}; "
        f"final snapshot iter={last_key}"
    )


# =============================================================================
# best_codes.npz re-encode
# =============================================================================


def reencode_codes_npz(
    codes_in: Path, codes_out: Path, old_to_new: dict[int, int],
) -> None:
    """Apply the same old→new state map to a ``best_codes.npz`` so the
    downstream policy training sees the merged labels."""
    if not codes_in.exists():
        raise FileNotFoundError(codes_in)
    log.info(f"Re-encoding codes: {codes_in} → {codes_out}")
    data = dict(np.load(codes_in, allow_pickle=False))
    remap = np.vectorize(old_to_new.get)
    for key in ("all_codes", "train_codes", "test_codes"):
        if key in data:
            data[key] = remap(data[key]).astype(np.int32)
    np.savez(codes_out, **data)
    log.info("  done.")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Input checkpoint.h5")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output checkpoint path.")
    parser.add_argument(
        "--merge", action="append", default=[], required=True,
        help='Group of state ids to merge, comma-separated, e.g. --merge "1,3". '
             "Repeat for multiple groups.",
    )
    parser.add_argument(
        "--all-snapshots", action="store_true",
        help="Merge every snapshot (default: last only).",
    )
    parser.add_argument(
        "--codes-in", type=Path, default=None,
        help="Optional: a best_codes.npz to also re-encode with the merge map.",
    )
    parser.add_argument(
        "--codes-out", type=Path, default=None,
        help="Where to write the re-encoded best_codes.npz.",
    )
    args = parser.parse_args()

    groups = parse_merge_args(args.merge)
    log.info(f"Merge groups: {groups}")

    merge_checkpoint(
        in_path=args.checkpoint,
        out_path=args.output,
        groups=groups,
        all_snapshots=args.all_snapshots,
    )

    if args.codes_in is not None:
        if args.codes_out is None:
            raise ValueError("--codes-in requires --codes-out")
        # Re-load the map from the new checkpoint for safety.
        with h5py.File(args.output, "r") as f:
            mp = f["merge_provenance/old_to_new"][()]
        old_to_new = {int(k): int(v) for k, v in mp}
        reencode_codes_npz(args.codes_in, args.codes_out, old_to_new)


if __name__ == "__main__":
    main()
