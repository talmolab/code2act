"""Single source of truth for the codes-npz schema written by every tokenizer.

Downstream :mod:`c2a.codegen.code_sources` (specifically ``_load_kpms_npz``)
defines the rigid reader contract — every tokenizer's codegen writer must
produce a file the reader will accept. Historically that schema lived
inline in :mod:`c2a.codegen.kpms.generate_codes` and was duplicated
informally by ``merge_states.reencode_codes_npz``. Now the rules live in
one place so the kinematics_vae variants can write a compliant file
without copy-pasting npz logic.

Required fields (every tokenizer):
    train_codes   : (N_train, T) int32 — codes for the balanced train split
    test_codes    : (N_test,  T) int32
    train_indices : (N_train,) int  — original H5 clip ids, in the order
                                       the encoder visited them
    test_indices  : (N_test,)  int
    num_states    : scalar int — embedding-table size for downstream
                                  ``nn.Embed(num_states, ...)``
    model_type    : scalar bytes string — short tokenizer tag, surfaces in
                                  WandB provenance (e.g. "arhmm_kpms",
                                  "kin_vae_kmeans")

Optional fields (write when meaningful; reader gracefully ignores unknown):
    all_codes      : (N_total, T) int32 — codes for every clip in the
                                          balanced split union, in
                                          **sorted-union order**:
                                          ``all_codes[i]`` ↔ clip with
                                          H5 id ``sorted(set(train_indices) |
                                          set(test_indices))[i]``.
                                          NOT ``np.concatenate([train, test])``.
                                          Several downstream experiments
                                          (``run_gait_dynamics``,
                                          ``run_inception_distance``, …)
                                          index this by position, so writers
                                          MUST match this ordering. Pass
                                          ``all_indices`` to have the writer
                                          enforce the contract row-by-row.
    kappa          : float — smoothing hyperparam for kpms fits.
    mean_duration  : float — paper-faithful "best fit" mean state duration.
    is_merged      : bool  — produced by ``merge_states.reencode_codes_npz``;
                              forces ``code_source: kpms_merged``.
    original_num_states / new_num_states : ints — merge bookkeeping.

Plus any tokenizer-specific provenance via the ``extra`` kwarg. Keys must
not collide with the required/optional names above.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

log = logging.getLogger(__name__)


# Field names that ``c2a.codegen.code_sources._load_kpms_npz`` knows about.
# Codegen writers MUST NOT put tokenizer-specific data under these names.
_REQUIRED = frozenset({
    "train_codes", "test_codes", "train_indices", "test_indices",
    "num_states", "model_type",
})
_RESERVED_OPTIONAL = frozenset({
    "all_codes", "kappa", "mean_duration", "is_merged",
    "original_num_states", "new_num_states",
})
RESERVED = _REQUIRED | _RESERVED_OPTIONAL


def write_codes_npz(
    output_path: str | Path,
    *,
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    num_states: int,
    model_type: str,
    all_codes: np.ndarray | None = None,
    all_indices: np.ndarray | None = None,
    kappa: float | None = None,
    mean_duration: float | None = None,
    is_merged: bool = False,
    original_num_states: int | None = None,
    new_num_states: int | None = None,
    extra: Mapping[str, Any] | None = None,
    compressed: bool = False,
) -> Path:
    """Write a CodesPayload-compatible codes file.

    Validates shape / dtype / value-range BEFORE writing so a corrupt file
    is never produced. Errors raise rather than silently dropping fields.

    Args:
        output_path: Target ``.npz`` path; parent dirs are created.
        train_codes: ``(N_train, T)`` integer array. Cast to int32 on disk.
        test_codes: ``(N_test, T)`` integer array. Cast to int32 on disk.
        train_indices: ``(N_train,)`` original H5 clip ids. Cast to int64.
        test_indices: ``(N_test,)`` original H5 clip ids. Cast to int64.
        num_states: Embedding-table size; every code must be in
            ``[0, num_states)``.
        model_type: Short tag identifying the tokenizer family. Stored as
            a numpy bytes-array; downstream reads it with ``.item()``.
        all_codes: Optional ``(N_train+N_test, T)`` convenience array.
            When provided, MUST be in **sorted-union order**: ``all_codes[i]``
            corresponds to H5 clip id ``sorted(set(train_indices) |
            set(test_indices))[i]``. (NOT the same as
            ``np.concatenate([train_codes, test_codes], axis=0)``.) Several
            downstream experiments index this array by sorted-union
            position; emitting a different ordering would silently
            misalign every cross-tokenizer plot.
        all_indices: Optional ``(N_train+N_test,)`` sorted-union of
            ``train_indices ∪ test_indices``. When provided alongside
            ``all_codes``, the writer cross-checks every row: for each
            ``i in train_indices``, asserts
            ``all_codes[all_indices.tolist().index(i)] ==
            train_codes[train_indices.tolist().index(i)]``. Catches
            wrong-order ``all_codes`` arrays at write time rather than
            in some downstream plot.
        kappa, mean_duration, is_merged, original_num_states,
        new_num_states: Optional provenance carried verbatim into the
            payload's ``provenance`` dict.
        extra: Optional tokenizer-specific fields. Keys must not collide
            with the reserved names listed in :data:`RESERVED`.
        compressed: When True, use ``np.savez_compressed`` (~3-5x slower
            write, ~20x smaller files for highly compressible code
            streams). Default False to match existing kpms generate_codes.

    Returns:
        Resolved absolute ``Path`` of the written file.

    Raises:
        ValueError: required-field shape/dtype/range violation.
        KeyError: ``extra`` collides with a reserved field name.
    """
    output_path = Path(output_path)

    train_codes = np.asarray(train_codes)
    test_codes = np.asarray(test_codes)
    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)
    num_states = int(num_states)
    model_type = str(model_type)

    _validate_arrays(
        train_codes, test_codes, train_indices, test_indices, num_states,
    )

    if extra is not None:
        clash = set(extra) & RESERVED
        if clash:
            raise KeyError(
                f"`extra` contains reserved field names {sorted(clash)}. "
                f"Reserved fields: {sorted(RESERVED)}."
            )

    payload: dict[str, np.ndarray] = {
        "train_codes": train_codes.astype(np.int32, copy=False),
        "test_codes": test_codes.astype(np.int32, copy=False),
        "train_indices": train_indices.astype(np.int64, copy=False),
        "test_indices": test_indices.astype(np.int64, copy=False),
        "num_states": np.int32(num_states),
        "model_type": np.array(model_type),
    }

    if all_codes is not None:
        all_codes = np.asarray(all_codes)
        expected_total = train_codes.shape[0] + test_codes.shape[0]
        if all_codes.shape[0] != expected_total:
            raise ValueError(
                f"all_codes.shape[0]={all_codes.shape[0]} != "
                f"train+test ({expected_total})."
            )
        if all_codes.shape[1] != train_codes.shape[1]:
            raise ValueError(
                f"all_codes.shape[1]={all_codes.shape[1]} != "
                f"train_codes.shape[1] ({train_codes.shape[1]})."
            )
        if all_indices is not None:
            all_indices_arr = np.asarray(all_indices, dtype=np.int64)
            if all_indices_arr.shape[0] != expected_total:
                raise ValueError(
                    f"all_indices length {all_indices_arr.shape[0]} != "
                    f"train+test ({expected_total})."
                )
            _check_all_codes_ordering(
                all_codes=all_codes,
                all_indices=all_indices_arr,
                train_codes=train_codes,
                train_indices=train_indices,
                test_codes=test_codes,
                test_indices=test_indices,
            )
        payload["all_codes"] = all_codes.astype(np.int32, copy=False)

    if kappa is not None:
        payload["kappa"] = np.float64(kappa)
    if mean_duration is not None:
        payload["mean_duration"] = np.float64(mean_duration)
    if is_merged:
        payload["is_merged"] = np.bool_(True)
        if original_num_states is None or new_num_states is None:
            raise ValueError(
                "is_merged=True requires original_num_states and "
                "new_num_states to be provided."
            )
        payload["original_num_states"] = np.int32(original_num_states)
        payload["new_num_states"] = np.int32(new_num_states)

    if extra is not None:
        for k, v in extra.items():
            payload[k] = np.asarray(v) if not isinstance(v, np.ndarray) else v

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compressed else np.savez
    saver(str(output_path), **payload)
    log.info(
        "Wrote codes npz to %s (model_type=%s, num_states=%d, "
        "train=%s, test=%s, %d extra fields)",
        output_path, model_type, num_states,
        train_codes.shape, test_codes.shape,
        len(extra) if extra else 0,
    )
    return output_path.resolve()


def _check_all_codes_ordering(
    *,
    all_codes: np.ndarray,
    all_indices: np.ndarray,
    train_codes: np.ndarray,
    train_indices: np.ndarray,
    test_codes: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    """Verify ``all_codes`` is in sorted-union order by cross-checking
    every row against ``train_codes`` / ``test_codes``.

    Enforces the contract documented on :func:`write_codes_npz`: for each
    H5 clip id ``i`` in ``train_indices``,
    ``all_codes[all_indices.tolist().index(i)] ==
    train_codes[train_indices.tolist().index(i)]``. Same for test.

    Raises ``ValueError`` on any mismatch — refuses to write a misordered
    ``all_codes``. Cheap (O(N*T) one-time) compared to a silently corrupt
    downstream plot.
    """
    # Build a map from H5 id → row in all_codes once.
    idx_to_pos = {int(c): pos for pos, c in enumerate(all_indices.tolist())}

    def _check(side: str, codes: np.ndarray, indices: np.ndarray) -> None:
        for i_pos, h5_id in enumerate(indices.tolist()):
            all_pos = idx_to_pos.get(int(h5_id))
            if all_pos is None:
                raise ValueError(
                    f"all_indices does not contain {side}_indices[{i_pos}]"
                    f"={h5_id}; cannot align all_codes."
                )
            if not np.array_equal(all_codes[all_pos], codes[i_pos]):
                raise ValueError(
                    f"all_codes ordering mismatch: all_codes[{all_pos}] "
                    f"(corresponds to H5 clip {h5_id}) disagrees with "
                    f"{side}_codes[{i_pos}]. Pass all_codes in "
                    f"sorted-union order (see write_codes_npz docstring)."
                )

    _check("train", train_codes, train_indices)
    _check("test", test_codes, test_indices)


def _validate_arrays(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    num_states: int,
) -> None:
    """Mirror the read-time invariants in
    ``c2a.codegen.code_sources._validate_codes_shape_and_range`` so a
    write never produces a file the loader will reject."""
    if train_codes.ndim != 2:
        raise ValueError(f"train_codes must be 2D, got shape {train_codes.shape}.")
    if test_codes.ndim != 2:
        raise ValueError(f"test_codes must be 2D, got shape {test_codes.shape}.")
    if train_codes.shape[1] != test_codes.shape[1]:
        raise ValueError(
            f"train_codes.shape[1]={train_codes.shape[1]} != "
            f"test_codes.shape[1]={test_codes.shape[1]}."
        )
    if train_indices.ndim != 1 or test_indices.ndim != 1:
        raise ValueError(
            f"indices must be 1D, got train {train_indices.shape}, "
            f"test {test_indices.shape}."
        )
    if train_codes.shape[0] != train_indices.shape[0]:
        raise ValueError(
            f"train_codes rows ({train_codes.shape[0]}) != "
            f"len(train_indices) ({train_indices.shape[0]})."
        )
    if test_codes.shape[0] != test_indices.shape[0]:
        raise ValueError(
            f"test_codes rows ({test_codes.shape[0]}) != "
            f"len(test_indices) ({test_indices.shape[0]})."
        )
    if not np.issubdtype(train_codes.dtype, np.integer):
        raise ValueError(f"train_codes must be integer; got {train_codes.dtype}.")
    if not np.issubdtype(test_codes.dtype, np.integer):
        raise ValueError(f"test_codes must be integer; got {test_codes.dtype}.")
    if num_states <= 0:
        raise ValueError(f"num_states must be > 0; got {num_states}.")

    if train_codes.size > 0:
        tmin, tmax = int(train_codes.min()), int(train_codes.max())
        if tmin < 0 or tmax >= num_states:
            raise ValueError(
                f"train_codes range [{tmin}, {tmax}] out of [0, {num_states})."
            )
    if test_codes.size > 0:
        emin, emax = int(test_codes.min()), int(test_codes.max())
        if emin < 0 or emax >= num_states:
            raise ValueError(
                f"test_codes range [{emin}, {emax}] out of [0, {num_states})."
            )
