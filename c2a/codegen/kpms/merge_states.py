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

    python -m c2a.codegen.kpms.merge_states \\
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

    The caller MUST pass each group sorted ascending; ``parse_merge_args``
    enforces this for the manual CLI, and ``propose_merge_groups_from_distance``
    does too. We assert here as a defensive contract — the per-group-tuple
    keys downstream (``weights_within_group``, ``group_to_new``) all depend
    on this invariant for consistency.
    """
    for g in groups:
        if list(g) != sorted(g):
            raise ValueError(
                f"merge_snapshot requires each group sorted ascending; got {g}"
            )
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
    downstream policy training sees the merged labels.

    The output npz carries explicit provenance fields so downstream loaders
    (``c2a.codegen.code_sources.load_codes`` with ``code_source: kpms_merged``) can
    verify that they aren't being handed a raw file by mistake:

    * ``is_merged``  — boolean True (np.bool_).
    * ``original_num_states`` — K of the input file (before merging).
    * ``new_num_states`` — K of the output file (after merging) = number
      of distinct values in ``old_to_new``.
    * ``num_states`` — overwritten to ``new_num_states`` so any consumer
      that reads it gets the correct value (the input npz's value would
      otherwise leak through and over-state the codebook size).
    """
    if not codes_in.exists():
        raise FileNotFoundError(codes_in)
    log.info(f"Re-encoding codes: {codes_in} → {codes_out}")
    data = dict(np.load(codes_in, allow_pickle=False))

    # Refuse to re-merge an already-merged file: doing so would overwrite
    # the original_num_states provenance with whatever this file's own
    # (post-prior-merge) num_states was, losing the chain back to the raw K.
    prior_is_merged = bool(np.asarray(data["is_merged"]).item()) if (
        "is_merged" in data
    ) else False
    if prior_is_merged:
        raise RuntimeError(
            f"{codes_in} already carries is_merged=True. Re-encoding a "
            f"merged file would overwrite original_num_states and lose "
            f"the chain to the raw K. Re-encode from the unmerged source npz."
        )

    remap = np.vectorize(old_to_new.get)
    for key in ("all_codes", "train_codes", "test_codes"):
        if key in data:
            data[key] = remap(data[key]).astype(np.int32)

    original_K = int(data.get("num_states", max(old_to_new.keys()) + 1))
    new_K = len(set(old_to_new.values()))
    data["num_states"] = np.int32(new_K)
    data["original_num_states"] = np.int32(original_K)
    data["new_num_states"] = np.int32(new_K)
    data["is_merged"] = np.bool_(True)

    np.savez(codes_out, **data)
    log.info(f"  done. K={original_K} -> {new_K}, is_merged=True")


# =============================================================================
# Automatic syllable merging via AR-HMM emission similarity (criterion 2)
# =============================================================================
#
# Each KPMS syllable k is a per-state autoregressive Gaussian:
#
#     x_t | history ~ N( A_k @ z, Q_k ),   z = [x_{t-1}; ...; x_{t-p}; 1]
#
# Two states are emission-redundant iff their conditional Gaussians produce
# nearly-identical predictions across the empirical history distribution.
# We measure that with a closed-form Bhattacharyya distance, averaged over
# the empirical p(z) (mean m and covariance S derived from states/x in
# the checkpoint itself):
#
#     B(i,j) = (1/8) [ tr(Σ⁻¹ ΔA S ΔAᵀ) + (ΔA m)ᵀ Σ⁻¹ (ΔA m) ]
#            + (1/2) [ log|Σ| − ½ log|Q_i| − ½ log|Q_j| ]
#     where  ΔA = A_i − A_j,    Σ = (Q_i + Q_j)/2
#
# Bhattacharyya is symmetric and bounded for non-degenerate Gaussians; the
# coefficient BC = exp(−B) gives an interpretable "% overlap" reading.
# B < 0.3 ↔ BC > 0.74 (~74% one-step-prediction overlap) is the default
# cutoff — see propose_merge_groups_from_distance.

# Default Bhattacharyya threshold for proposing merges. BC = exp(-0.3) ≈ 0.74.
# Below this distance, two AR-HMM states predict near-identical next-step
# distributions under typical pose history. Calibrate by inspecting the
# B-histogram printed at run time and rerunning with a different value.
# Bumped 2026-05-11 from 0.3 → 2.0 to merge more aggressively. At 0.3
# the criterion was so strict that only obvious AR-duplicates got
# collapsed (B-distance distribution typically has min ≈ 0.1, median
# ≈ 15-25), leaving the duration-graded duplicate-pose family
# (entry/hold/exit phases of the same behavior) all unmerged. BC =
# exp(-2.0) ≈ 0.135 = "merge if the two AR emissions overlap by ≥ 14%".
DEFAULT_BHATTACHARYYA_THRESHOLD: float = 2.0


def _build_history_vectors(
    x: np.ndarray, mask: np.ndarray | None, nlags: int,
) -> np.ndarray:
    """Stack lag history vectors z_t = [x_{t-1}; ...; x_{t-p}; 1].

    Args:
        x: Latent trajectories ``(N, T_x, d)``.
        mask: Optional boolean mask ``(N, T_x)`` — ``True`` for valid
            frames. ``None`` means use all frames.
        nlags: AR order (= ``p``). Must satisfy ``T_x > nlags``.

    Returns:
        Flat history matrix ``(M, d*p + 1)`` where ``M`` is the number of
        valid timepoints across all clips after dropping the first
        ``nlags`` warm-up frames per clip.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (N, T, d); got shape {x.shape}")
    N, T, d = x.shape
    if T <= nlags:
        raise ValueError(f"T={T} <= nlags={nlags}; cannot build history")

    n_valid = T - nlags
    z = np.empty((N, n_valid, d * nlags + 1), dtype=np.float64)
    # Lag L=0 corresponds to x_{t-1} (one step back); L=p-1 to x_{t-p}.
    for L in range(nlags):
        z[:, :, L * d:(L + 1) * d] = x[:, nlags - L - 1: T - L - 1, :]
    z[:, :, -1] = 1.0

    if mask is not None:
        # mask is (N, T); align to z's time axis (drop first nlags frames).
        mask_z = mask[:, nlags:].astype(bool)
        return z[mask_z]
    return z.reshape(-1, d * nlags + 1)


def _bhattacharyya_pair(
    A_i: np.ndarray, Q_i: np.ndarray,
    A_j: np.ndarray, Q_j: np.ndarray,
    z_mean: np.ndarray, z_cov: np.ndarray,
    *, eps: float = 1e-6,
) -> float:
    """Closed-form Bhattacharyya distance between two AR-HMM emissions.

    See module-level commentary for the formula. Returns a non-negative
    scalar; ``0`` means identical emission distributions (under the
    given history statistics), larger values mean less overlap.
    """
    d = Q_i.shape[0]
    Sigma = 0.5 * (Q_i + Q_j) + eps * np.eye(d)

    dA = A_i - A_j  # (d, d*p+1)
    iSigma = np.linalg.inv(Sigma)

    diff_mean = dA @ z_mean  # (d,)
    cov_part = float(np.trace(iSigma @ dA @ z_cov @ dA.T))
    mean_part = float(diff_mean @ iSigma @ diff_mean)
    mean_term = (cov_part + mean_part) / 8.0

    # Use slogdet for numerical stability; jitter before logdet too.
    _, log_S  = np.linalg.slogdet(Sigma)
    _, log_Qi = np.linalg.slogdet(Q_i + eps * np.eye(d))
    _, log_Qj = np.linalg.slogdet(Q_j + eps * np.eye(d))
    cov_term = 0.5 * (log_S - 0.5 * (log_Qi + log_Qj))
    return mean_term + cov_term


def compute_pairwise_bhattacharyya(snapshot: dict) -> np.ndarray:
    """Pairwise Bhattacharyya distance matrix for one KPMS snapshot.

    Uses ``params/Ab`` and ``params/Q`` for the per-state emissions, and
    derives the empirical history distribution from ``states/x``.

    Returns:
        Symmetric ``(K, K)`` matrix; diagonal is zero. Larger values mean
        more distinguishable emissions.
    """
    Ab = np.asarray(snapshot["params"]["Ab"])         # (K, d, d*p+1)
    Q  = np.asarray(snapshot["params"]["Q"])          # (K, d, d)
    if "x" not in snapshot.get("states", {}):
        raise KeyError(
            "Snapshot states/x is required to derive empirical history "
            "statistics for the Bhattacharyya criterion."
        )
    x = np.asarray(snapshot["states"]["x"])           # (N, T_x, d)

    K, d, n_cols = Ab.shape
    if (n_cols - 1) % d != 0:
        raise RuntimeError(
            f"Ab has {n_cols} columns; expected d*nlags + 1 with d={d}."
        )
    nlags = (n_cols - 1) // d

    z = _build_history_vectors(x, mask=None, nlags=nlags)
    if z.shape[0] < z.shape[1]:
        log.warning(
            "Empirical history sample (M=%d) is smaller than its "
            "dimensionality (D=%d); covariance will be under-determined.",
            z.shape[0], z.shape[1],
        )
    z_mean = z.mean(axis=0)
    # ddof=0 to match population covariance (large M makes this irrelevant).
    z_cov = np.cov(z.T, ddof=0)

    B = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            B[i, j] = B[j, i] = _bhattacharyya_pair(
                Ab[i], Q[i], Ab[j], Q[j], z_mean, z_cov,
            )
    # AM-GM on regularized SPD matrices doesn't strictly hold (Σ and the
    # individual Q are jittered separately), so cov_term can drift slightly
    # negative. Sanity-bound it; below this we assume a real numerical
    # blow-up rather than rounding error.
    if B.min() < -1e-3:
        raise RuntimeError(
            f"Bhattacharyya matrix has unexpectedly negative entries "
            f"(min={B.min():.4g}). This usually indicates ill-conditioned "
            f"Q matrices. Inspect the snapshot's params/Q for near-singular "
            f"per-state covariance."
        )
    return np.maximum(B, 0.0)


def propose_merge_groups_from_distance(
    B: np.ndarray,
    threshold: float = DEFAULT_BHATTACHARYYA_THRESHOLD,
    *,
    linkage: str = "complete",
) -> list[list[int]]:
    """Propose merge groups by thresholding a pairwise distance matrix.

    Args:
        B: Symmetric ``(K, K)`` pairwise distance.
        threshold: Cutoff. Pairs/clusters with distance ``< threshold``
            become merge candidates.
        linkage: Aggregation strategy:
            * ``"complete"`` (default, recommended): hierarchical clustering
              with complete linkage, cut at ``threshold``. Every pair in
              a cluster must be below threshold — conservative.
            * ``"single"``: connected components on the threshold graph.
              Transitive chains can over-merge (A~B, B~C → A,B,C merged
              even if A and C are far apart). Use only when you trust
              the threshold tightly.

    Returns:
        Sorted list of merge groups; each group is a list of original
        state ids of size ``>= 2``. States that don't merge are omitted.
    """
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError(f"B must be square; got shape {B.shape}")
    K = B.shape[0]
    if K < 2:
        return []

    if linkage == "single":
        # Connected components on the symmetric below-threshold graph.
        # Implement union-find inline so we don't pull in scipy.csgraph
        # for one trivial computation.
        parent = list(range(K))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(K):
            for j in range(i + 1, K):
                if B[i, j] < threshold:
                    union(i, j)
        labels = np.array([find(k) for k in range(K)])

    elif linkage == "complete":
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage as h_linkage
        from scipy.spatial.distance import squareform

        # squareform expects exactly symmetric, zero diagonal; clip
        # negative numerical noise to keep it valid.
        B_pos = np.maximum(B, 0.0)
        np.fill_diagonal(B_pos, 0.0)
        condensed = squareform(B_pos, checks=False)
        Z = h_linkage(condensed, method="complete")
        labels = fcluster(Z, t=threshold, criterion="distance") - 1
    else:
        raise ValueError(
            f"Unknown linkage={linkage!r}; expected 'complete' or 'single'."
        )

    groups: list[list[int]] = []
    for lab in range(int(labels.max()) + 1):
        members = sorted(int(k) for k in range(K) if labels[k] == lab)
        if len(members) >= 2:
            groups.append(members)
    groups.sort()
    return groups


def auto_merge_one(
    checkpoint_in: Path,
    checkpoint_out: Path,
    *,
    threshold: float = DEFAULT_BHATTACHARYYA_THRESHOLD,
    linkage: str = "complete",
) -> dict:
    """Auto-propose-and-merge for ONE KPMS checkpoint.

    Computes pairwise B on the final snapshot, proposes groups under
    ``threshold`` / ``linkage``, and applies them via
    :func:`merge_checkpoint`. Also writes a JSON report alongside the
    output checkpoint.

    Returns:
        Dict with ``groups``, ``threshold``, ``linkage``, ``B_path``,
        ``original_K``, ``new_K``, and ``B_summary`` (min / median / max
        of the off-diagonal B values). Empty ``groups`` means no merges
        were proposed — checkpoint_out is still produced as a copy of
        the input via merge_checkpoint(groups=[]).
    """
    if not checkpoint_in.exists():
        raise FileNotFoundError(checkpoint_in)
    log.info(f"Auto-merge: scanning {checkpoint_in}")

    with h5py.File(checkpoint_in, "r") as fin:
        snap_keys = sorted(fin["model_snapshots"].keys(), key=int)
        if not snap_keys:
            raise RuntimeError(f"No snapshots in {checkpoint_in}")
        snap = _read_snapshot(fin, f"model_snapshots/{snap_keys[-1]}")

    B = compute_pairwise_bhattacharyya(snap)
    K = int(B.shape[0])

    iu = np.triu_indices(K, k=1)
    off_diag = B[iu]
    if off_diag.size > 0:
        b_min = float(off_diag.min())
        b_med = float(np.median(off_diag))
        b_max = float(off_diag.max())
        log.info(
            f"  B-matrix: K={K}, off-diag min/median/max = "
            f"{b_min:.3f} / {b_med:.3f} / {b_max:.3f}"
        )
        # Histogram for the user's eye.
        bins = [0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, np.inf]
        hist, _ = np.histogram(off_diag, bins=bins)
        log.info(
            "  B-histogram (off-diag pairs): "
            + ", ".join(
                f"<{bins[i+1]}:{int(hist[i])}" for i in range(len(hist) - 1)
            )
            + f", >={bins[-2]}:{int(hist[-1])}"
        )
    else:
        b_min = b_med = b_max = 0.0

    groups = propose_merge_groups_from_distance(
        B, threshold=threshold, linkage=linkage,
    )
    log.info(
        f"  Proposed {len(groups)} merge group(s) at threshold={threshold} "
        f"(linkage={linkage}): {groups}"
    )

    new_K = K - sum(len(g) - 1 for g in groups)
    if new_K < 1:
        raise RuntimeError(
            f"Proposed merges would collapse all {K} states into one; "
            f"raise --threshold or use --linkage complete."
        )

    # Always run merge_checkpoint, even with empty groups, so the output
    # path is populated and downstream rendering / reencoding has a
    # consistent file to point at. (merge_checkpoint with groups=[] writes
    # a "merged" file that's effectively a copy with provenance attached.)
    if groups:
        merge_checkpoint(
            in_path=checkpoint_in,
            out_path=checkpoint_out,
            groups=groups,
            all_snapshots=False,
        )
    else:
        log.warning(
            "  No groups proposed — auto-merge produces no checkpoint "
            "for this fit. Lower --threshold or switch --linkage to 'single' "
            "for a more aggressive partition."
        )
        checkpoint_out = checkpoint_in  # caller can detect via report.

    # Save B and a JSON report alongside the merged checkpoint.
    out_parent = (
        checkpoint_out.parent if checkpoint_out != checkpoint_in
        else checkpoint_in.parent
    )
    out_parent.mkdir(parents=True, exist_ok=True)
    b_path = out_parent / "bhattacharyya_matrix.npy"
    np.save(b_path, B)

    report = {
        "checkpoint_in": str(checkpoint_in),
        "checkpoint_out": str(checkpoint_out),
        "threshold": threshold,
        "linkage": linkage,
        "original_K": K,
        "new_K": new_K,
        "groups": groups,
        "B_path": str(b_path),
        "B_summary": {"min": b_min, "median": b_med, "max": b_max},
    }
    report_path = out_parent / "auto_merge_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Report: {report_path}; B matrix: {b_path}")
    return report


def auto_merge_topk(
    sweep_dir: Path,
    *,
    top_k: int = 5,
    rank: str = "eml",
    threshold: float = DEFAULT_BHATTACHARYYA_THRESHOLD,
    linkage: str = "complete",
    output_subdir: str = "auto_merged",
    render: bool = True,
    fps: int = 30,
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    codes_in: Path | None = None,
    # Trajectory-plot rendering kwargs forwarded to render_native_for_fit.
    # min_frequency=0.0 by default = render every syllable that has any
    # bout passing min_duration. Bump it (e.g. 0.001) if a long tail of
    # transient syllables clutters the grid.
    pre: float = 0.167,
    post: float = 0.5,
    min_frequency: float = 0.0,
    min_duration: int = 3,
    n_snapshots: int = 5,
    n_samples_per_syllable: int = 50,
    projection_planes: tuple[str, ...] = ("xy", "xz"),
    keypoint_colormap: str = "autumn",
    save_png: bool = True,
    save_gifs: bool = True,
) -> dict:
    """For each of the top-K KPMS fits, auto-merge and optionally render.

    Selection of top-K mirrors the top-K-bands / syllable-grids
    convention: read ``sweep_results.json``, sort valid fits by
    ``rank`` (``mse`` ascending or ``nmi`` descending), keep ``top_k``.

    Output structure (under ``{sweep_dir}/{output_subdir}/``):

        M{k}/
          checkpoint_merged.h5            (or absent if no merges proposed)
          auto_merge_report.json
          bhattacharyya_matrix.npy
          syllable_grids/...              (if render=True and merged)

    A top-level ``manifest.json`` indexes all M{k} entries.

    Args:
        sweep_dir: Directory containing ``sweep_results.json``.
        top_k: Number of top fits to process.
        rank: ``"eml"`` (default — Weinreb 2024: filter to fits with
            median_duration in 400 ms ±30%, rank by EML desc),
            ``"mse"`` (legacy — reconstruction MSE asc), or
            ``"nmi"`` (label-aligned NMI desc).
        threshold: Bhattacharyya cutoff (passed through).
        linkage: ``"complete"`` (default) or ``"single"``.
        output_subdir: Subdirectory of sweep_dir for outputs.
        render: If True, render syllable grids on the merged labels.
        fps: Video FPS.
    """
    sweep_dir = Path(sweep_dir).resolve()
    results_path = sweep_dir / "sweep_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"sweep_results.json not under {sweep_dir}; cannot pick top-K."
        )
    # Lazy import to avoid pulling matplotlib into pure auto-merge runs.
    from c2a.codegen.kpms.render_topk_bands import _select_topk
    from c2a.paths import get_repo_root

    with results_path.open() as f:
        results = json.load(f)
    top = _select_topk(results["all_results"], rank=rank, top_k=top_k)
    if not top:
        raise RuntimeError(f"No valid fits in {results_path}.")
    log.info(
        f"Auto-merge top-{len(top)} fits (rank={rank}, threshold={threshold}, "
        f"linkage={linkage})"
    )

    # Codes-npz auto-detection: if the user didn't pass --codes-in, look
    # for the canonical {sweep_dir}/best_codes.npz. We only re-encode it
    # for the M{k} whose (setting, seed) matches sweep_results.json's
    # best_model — the npz comes from c2a.codegen.kpms.generate_codes which
    # is keyed off that single fit. Other top-K fits don't have an
    # auto-matching codes file (they'd need generate_all_codes), so we
    # skip them silently.
    if codes_in is None:
        candidate = sweep_dir / "best_codes.npz"
        if candidate.exists():
            codes_in = candidate
            log.info(f"Auto-detected codes file: {codes_in}")
    elif not Path(codes_in).is_absolute():
        codes_in = sweep_dir / codes_in

    best_match_key: str | None = None
    if codes_in is not None:
        best = results.get("best_model") or {}
        best_setting = best.get("setting")
        best_seed = best.get("seed")
        for k, r in enumerate(top, start=1):
            if r.get("setting") == best_setting and r.get("seed") == best_seed:
                best_match_key = f"M{k}"
                log.info(
                    f"Codes file {codes_in.name} corresponds to {best_match_key} "
                    f"(setting={best_setting}, seed={best_seed}); will auto-"
                    f"re-encode it after that fit's merge."
                )
                break
        if best_match_key is None:
            log.warning(
                f"Codes file {codes_in} doesn't match any top-{top_k} fit "
                f"(best_model setting={best_setting}, seed={best_seed}); "
                f"skipping auto re-encode."
            )

    repo_root = get_repo_root()
    out_root = sweep_dir / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {
        "sweep_dir": str(sweep_dir),
        "rank": rank,
        "threshold": threshold,
        "linkage": linkage,
        "top_k": top_k,
        "fits": {},
    }

    # Lazy-import the renderer (avoid heavy deps if render=False).
    coordinates = None
    if render:
        from c2a.codegen.kpms.keypoint_loader import prepare_keypoints_for_kpms
        from c2a.codegen.kpms.native_viz import _load_keypoints_mm

        if balanced_split_path is None:
            balanced_split_path = (
                repo_root / "data" / "rodent" / "rodent_balanced_splits.json"
            )
        if reference_data_path is None:
            reference_data_path = (
                repo_root / "data" / "rodent" / "rodent_reference_clips.h5"
            )
        if stac_xml_path is None:
            stac_xml_path = repo_root / "c2a" / "assets" / "rodent.xml"

        log.info("Loading FK keypoints (one-time; ~30s)...")
        keypoints_mm, _kp_names = _load_keypoints_mm(
            reference_data_path, stac_xml_path, balanced_split_path,
        )
        coordinates, _ = prepare_keypoints_for_kpms(keypoints_mm)
        log.info(f"  {len(coordinates)} recordings (mm-scaled).")

    for k, r in enumerate(top, start=1):
        project_dir = Path(r["project_dir"])
        if not project_dir.is_absolute():
            project_dir = repo_root / project_dir
        ckpt_in = project_dir / r["model_name"] / "checkpoint.h5"
        if not ckpt_in.exists():
            log.warning(f"M{k}: checkpoint missing at {ckpt_in}; skipping.")
            manifest["fits"][f"M{k}"] = {
                "status": "skipped",
                "reason": f"checkpoint missing: {ckpt_in}",
                "setting": r.get("setting"),
                "seed": r.get("seed"),
            }
            continue

        m_dir = out_root / f"M{k}"
        m_dir.mkdir(parents=True, exist_ok=True)
        ckpt_out = m_dir / "checkpoint_merged.h5"
        if ckpt_out.exists():
            ckpt_out.unlink()  # merge_checkpoint refuses to overwrite

        log.info(f"--- M{k} (setting={r.get('setting')}, seed={r.get('seed')}) ---")
        report = auto_merge_one(
            checkpoint_in=ckpt_in,
            checkpoint_out=ckpt_out,
            threshold=threshold,
            linkage=linkage,
        )
        report["fit_setting"] = r.get("setting")
        report["fit_seed"] = r.get("seed")
        report["fit_rank"] = k
        manifest["fits"][f"M{k}"] = report

        # Auto re-encode codes for the matching fit, no manual step required.
        if (
            codes_in is not None
            and best_match_key == f"M{k}"
            and report["groups"]
        ):
            codes_out = m_dir / "best_codes_merged.npz"
            with h5py.File(ckpt_out, "r") as f:
                mp = f["merge_provenance/old_to_new"][()]
            old_to_new = {int(kk): int(vv) for kk, vv in mp}
            reencode_codes_npz(Path(codes_in), codes_out, old_to_new)
            report["codes_in"] = str(codes_in)
            report["codes_out"] = str(codes_out)
            log.info(
                f"  Wrote merged codes for trainer: {codes_out}\n"
                f"  Wire into c2a/configs/moseq_decoder.yaml as:\n"
                f"      kpms_config:\n"
                f"        code_source: kpms_merged\n"
                f"        codes_path: {codes_out}"
            )

        if render and report["groups"]:
            _render_merged_fit_native(
                fit_rank=k,
                merged_ckpt=Path(report["checkpoint_out"]),
                project_dir=project_dir,
                model_name=r["model_name"],
                coordinates=coordinates,
                output_dir=m_dir / "trajectory_plots",
                pre=pre, post=post,
                min_frequency=min_frequency, min_duration=min_duration,
                n_snapshots=n_snapshots,
                n_samples_per_syllable=n_samples_per_syllable,
                fps=fps, projection_planes=list(projection_planes),
                keypoint_colormap=keypoint_colormap,
                save_png=save_png, save_gifs=save_gifs,
            )
        elif render and not report["groups"]:
            log.info(f"  M{k}: nothing merged; skipping render for this fit.")

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Auto-merge done. Manifest: {manifest_path}")
    return manifest


def _render_merged_fit_native(
    fit_rank: int,
    merged_ckpt: Path,
    project_dir: Path,
    model_name: str,
    coordinates: dict,
    output_dir: Path,
    *,
    pre: float,
    post: float,
    min_frequency: float,
    min_duration: int,
    n_snapshots: int,
    n_samples_per_syllable: int,
    fps: int,
    projection_planes: list[str],
    keypoint_colormap: str,
    save_png: bool = True,
    save_gifs: bool = True,
) -> None:
    """Render KPMS-native trajectory plots for ONE merged fit.

    Reads the merged checkpoint's authoritative ``merge_provenance/
    old_to_new`` map (rather than rebuilding from ``groups``, which
    would diverge from ``merge_snapshot``'s descending-usage sort and
    silently mislabel the rendered syllables). Applies the map to the
    fit's ``results.h5`` syllable arrays and substitutes them into the
    ``results`` dict that KPMS's native trajectory renderer consumes.
    """
    from c2a.codegen.kpms.native_viz import render_native_for_fit
    from c2a.codegen.kpms.render_topk_bands import _load_syllables_for_fit

    try:
        sylls_orig = _load_syllables_for_fit(project_dir, model_name)
    except FileNotFoundError as e:
        log.warning(f"M{fit_rank}: {e}; skipping render.")
        return
    if not sylls_orig:
        log.warning(f"M{fit_rank}: empty results.h5; skipping render.")
        return

    if not merged_ckpt.exists():
        log.warning(
            f"M{fit_rank}: merged checkpoint {merged_ckpt} missing; skipping render."
        )
        return
    with h5py.File(merged_ckpt, "r") as f:
        if "merge_provenance/old_to_new" not in f:
            log.warning(
                f"M{fit_rank}: merged checkpoint lacks merge_provenance/old_to_new; "
                f"cannot remap syllables for render."
            )
            return
        mp = f["merge_provenance/old_to_new"][()]
    old_to_new = {int(k): int(v) for k, v in mp}
    remap = np.vectorize(old_to_new.get)
    sylls_merged = [remap(s).astype(s.dtype) for s in sylls_orig]

    render_native_for_fit(
        project_dir=project_dir,
        model_name=model_name,
        coordinates=coordinates,
        output_dir=output_dir,
        syllable_override=sylls_merged,
        pre=pre, post=post,
        min_frequency=min_frequency, min_duration=min_duration,
        n_snapshots=n_snapshots,
        n_samples_per_syllable=n_samples_per_syllable,
        fps=fps, projection_planes=projection_planes,
        keypoint_colormap=keypoint_colormap,
        save_png=save_png, save_gifs=save_gifs,
        title=f"M{fit_rank} (merged) — {model_name}",
    )


def _build_old_to_new_from_groups(
    groups: list[list[int]], K: int,
) -> dict[int, int]:
    """Rebuild merge_snapshot's old_to_new map from `groups` and K_orig.

    Mirrors merge_snapshot's logic: lone states keep their original
    indices, groups append after; final indices are reordered by usage,
    but for the syllable-renderer we only need a consistent compact
    relabeling — usage-sorting is cosmetic. We use a stable, ordered
    relabel: lone-states first (in order), then groups (sorted).
    """
    in_group = set(s for g in groups for s in g)
    lone = [s for s in range(K) if s not in in_group]
    new_idx = 0
    old_to_new: dict[int, int] = {}
    for s in lone:
        old_to_new[s] = new_idx
        new_idx += 1
    for g in sorted(groups):
        for s in g:
            old_to_new[s] = new_idx
        new_idx += 1
    return old_to_new


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        epilog=(
            "Two modes, dispatched by which args you pass: "
            "(manual)  --checkpoint X --merge '1,3' --output Y    | "
            "(auto)    --sweep-dir Z [--top-k 5 --threshold 0.3]."
        ),
    )

    # ---- Manual-mode args (original CLI, preserved for backward compat) ----
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="(manual) Input checkpoint.h5",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="(manual) Output checkpoint path.",
    )
    parser.add_argument(
        "--merge", action="append", default=None,
        help='(manual) Group of state ids, comma-separated; repeat for multiple groups.',
    )
    parser.add_argument(
        "--all-snapshots", action="store_true",
        help="(manual) Merge every snapshot (default: last only).",
    )
    parser.add_argument(
        "--codes-in", type=Path, default=None,
        help="(manual) Optional best_codes.npz to re-encode with the merge map.",
    )
    parser.add_argument(
        "--codes-out", type=Path, default=None,
        help="(manual) Where to write the re-encoded best_codes.npz.",
    )

    # ---- Auto-mode args ----
    parser.add_argument(
        "--sweep-dir", type=Path, default=None,
        help="(auto) Sweep directory containing sweep_results.json. "
             "Presence of this flag selects auto-merge mode.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="(auto)")
    parser.add_argument(
        "--rank", choices=("mse", "nmi", "eml"), default="eml",
        help="(auto) Top-K ranking. 'eml' (default, Weinreb 2024) filters "
             "fits to median_duration in [4.2, 7.8] frames then ranks by "
             "EML descending — same criterion as the sweep's best_model. "
             "'mse' ranks by reconstruction MSE ascending (legacy). "
             "'nmi' ranks by alignment with manual labels.",
    )
    parser.add_argument(
        "--threshold", type=float,
        default=DEFAULT_BHATTACHARYYA_THRESHOLD,
        help=f"(auto) Bhattacharyya cutoff (default "
             f"{DEFAULT_BHATTACHARYYA_THRESHOLD}; BC = exp(-threshold)).",
    )
    parser.add_argument(
        "--linkage", choices=("complete", "single"), default="complete",
        help="(auto) Cluster aggregation; complete is conservative.",
    )
    parser.add_argument("--output-subdir", type=str, default="auto_merged")
    parser.add_argument(
        "--no-render", dest="render", action="store_false", default=True,
        help="(auto) Skip KPMS-native trajectory-plot rendering after merging.",
    )
    parser.add_argument("--fps", type=int, default=30, help="(auto)")
    parser.add_argument(
        "--auto-codes-in", type=Path, default=None,
        help="(auto) Optional codes .npz to also re-encode after merging "
             "(default: auto-detect {sweep-dir}/best_codes.npz). The "
             "re-encoded file lands at "
             "{sweep-dir}/{output-subdir}/M{k}/best_codes_merged.npz "
             "for the M{k} that matches sweep_results.json's best_model.",
    )

    args = parser.parse_args()

    auto_args = args.sweep_dir is not None
    manual_args = (
        args.checkpoint is not None or args.output is not None
        or (args.merge and len(args.merge) > 0)
    )
    if auto_args and manual_args:
        parser.error(
            "--sweep-dir is exclusive with --checkpoint/--output/--merge. "
            "Pass one set or the other."
        )
    if not auto_args and not manual_args:
        parser.error(
            "Specify either --sweep-dir (auto mode) or --checkpoint + "
            "--merge + --output (manual mode)."
        )

    if auto_args:
        auto_merge_topk(
            sweep_dir=args.sweep_dir,
            top_k=args.top_k,
            rank=args.rank,
            threshold=args.threshold,
            linkage=args.linkage,
            output_subdir=args.output_subdir,
            render=args.render,
            fps=args.fps,
            codes_in=args.auto_codes_in,
        )
        return

    # Manual mode.
    if args.checkpoint is None or args.output is None or not args.merge:
        parser.error(
            "Manual mode requires --checkpoint, --output, and at least one --merge."
        )
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
        with h5py.File(args.output, "r") as f:
            mp = f["merge_provenance/old_to_new"][()]
        old_to_new = {int(k): int(v) for k, v in mp}
        reencode_codes_npz(args.codes_in, args.codes_out, old_to_new)


if __name__ == "__main__":
    main()
