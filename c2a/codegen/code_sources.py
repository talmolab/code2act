"""Pluggable behavioral-code sources for the MoSeq decoder pipeline.

The decoder policy receives a per-frame discrete code via
``obs["kpms_code"]``. Historically that code came straight out of a
Keypoint-MoSeq fit (``best_codes.npz``).  This module abstracts the
"where do codes come from?" decision behind a single config field so
different sources can be swapped without touching the env, the
network, or the PPO loop:

    ``kpms_config.code_source`` ∈
        - ``"kpms_raw"``            — raw KPMS codes from a sweep (default).
        - ``"kpms_merged"``         — merged KPMS codes; requires the npz
                                       to carry ``is_merged=True`` (written
                                       by ``c2a.codegen.kpms.merge_states.reencode_codes_npz``).
        - ``"kinematics_vae"``      — discrete codes from K-Means on a
                                       trained vanilla kinematics-VAE's
                                       continuous latent (written by
                                       ``c2a.codegen.kinematics_vae.vae.tokenize``).
        - ``"gt_directional_turn"`` — 5-class ground-truth labels:
                                       Immobile, Rear, Walk, Turn_Left, Turn_Right.

The contract returned to the caller is fixed regardless of source:

    * ``train_codes`` and ``test_codes`` are ``np.int32`` arrays of shape
      ``(n_train, n_frames)`` and ``(n_test, n_frames)``, with values in
      ``[0, num_codes)``.  No sentinels, no NaNs.
    * ``num_codes`` is the embedding-table size the network factory must
      use (validated to bound every code in the data).

This is the **only** place in the pipeline that touches code provenance
or label-to-code mapping; ``MoSeqImitation``, ``moseq_ppo`` and the
network factories see a flat tensor and don't care where it came from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from c2a.paths import get_repo_root

log = logging.getLogger(__name__)


# Code-source identifiers (kept in sync with the YAML doc).
SOURCE_KPMS_RAW = "kpms_raw"
SOURCE_KPMS_MERGED = "kpms_merged"
SOURCE_KINEMATICS_VAE = "kinematics_vae"
SOURCE_GT_DIRECTIONAL_TURN = "gt_directional_turn"
SOURCE_MLB = "mlb"  # manually labeled behaviors: 9 flat behavior x direction
                    # codes (see c2a.codegen.mlb_segmenter)
SOURCE_RANDOM = "random"  # implicit when codes_path=null

# Sources whose ``.npz`` file format is the shared CodesPayload schema
# (written by ``c2a.codegen.shared.npz_schema.write_codes_npz``). All
# share the same loader (`_load_kpms_npz`); the ``requested_source`` arg
# is what differentiates them in WandB provenance.
_NPZ_SCHEMA_SOURCES = frozenset({
    SOURCE_KPMS_RAW,
    SOURCE_KPMS_MERGED,
    SOURCE_KINEMATICS_VAE,
    SOURCE_MLB,
})

VALID_SOURCES = {
    SOURCE_KPMS_RAW,
    SOURCE_KPMS_MERGED,
    SOURCE_KINEMATICS_VAE,
    SOURCE_GT_DIRECTIONAL_TURN,
    SOURCE_MLB,
}


# 5-class mapping for gt_directional_turn.
# Behavior indices match c2a.codegen.shared.label_alignment.BEHAVIORS:
#   0=Immobile, 1=Rear, 2=Turn, 3=Walk
# Direction indices match c2a.codegen.shared.label_alignment.DIRECTIONS:
#   0=Left, 1=Right, 2=Straight
GT_DIRECTIONAL_TURN_NUM_CODES = 5
GT_DIRECTIONAL_TURN_NAMES = (
    "Immobile",     # class 0
    "Rear",         # class 1  (any direction)
    "Walk",         # class 2  (any direction)
    "Turn_Left",    # class 3
    "Turn_Right",   # class 4
)


@dataclass
class CodesPayload:
    """Structured return value from :func:`load_codes`.

    Attributes:
        train_codes: ``[n_train, n_frames]`` int32 in ``[0, num_codes)``.
        test_codes:  ``[n_test,  n_frames]`` int32 in ``[0, num_codes)``.
        num_codes:   Embedding-table size for the network factory.
        code_source: The resolved source string (echoes config; with
            ``"random"`` substituted in when ``codes_path=null``).
        provenance:  Source-specific metadata (kappa / num_states for
            KPMS sources, class names for GT sources).  Forwarded to
            WandB config for run identification.
    """

    train_codes: np.ndarray
    test_codes: np.ndarray
    num_codes: int
    code_source: str
    provenance: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Public dispatcher
# =============================================================================


def load_codes(
    cfg,
    train_indices: list[int] | np.ndarray,
    test_indices: list[int] | np.ndarray,
    n_frames: int,
) -> CodesPayload:
    """Load (or generate) per-frame codes for the configured source.

    Args:
        cfg: OmegaConf config (full Hydra cfg). Reads ``cfg.kpms_config``.
        train_indices: Original H5 clip ids in the training subset, in the
            order ``ReferenceClips`` will preserve them. The returned
            ``train_codes`` is aligned position-by-position to this list
            (i.e. row ``i`` of ``train_codes`` corresponds to
            ``train_indices[i]``).
        test_indices: Same for the test subset.
        n_frames: H5 frames per clip (= ``cfg.env_config.clip_length``).

    Returns:
        :class:`CodesPayload` with validated arrays and ``num_codes``.

    Raises:
        FileNotFoundError: configured path does not resolve.
        RuntimeError: wrong shape / dtype / value-range / merged-flag
            mismatch.
    """
    kpms_cfg = cfg.kpms_config
    code_source = str(kpms_cfg.get("code_source", SOURCE_KPMS_RAW))
    codes_path = kpms_cfg.get("codes_path", None)

    train_indices = np.asarray(list(train_indices), dtype=np.int64)
    test_indices = np.asarray(list(test_indices), dtype=np.int64)
    n_train = int(train_indices.shape[0])
    n_test = int(test_indices.shape[0])

    # Smoke-test fallback: ``codes_path=null`` → random codes sized by
    # ``cfg.network_config.num_codes``. We only honour this for the
    # KPMS-family sources (``kpms_raw`` / ``kpms_merged``) where the
    # random codebook size is consistent with how the user set
    # ``num_codes`` in their decoder yaml. For the kinematics_vae source
    # the codebook size is determined by the *tokenizer's* training config
    # (latent-VAE K-means cluster count), NOT the decoder yaml's
    # ``num_codes`` — silently falling back would train a
    # working-but-semantically-random policy whose code distribution
    # doesn't match the intended tokenizer. Refuse the fallback there and
    # force the user to either point at a real tokenizer-output npz or
    # switch to ``kpms_raw`` for smoke runs.
    _RANDOM_FALLBACK_SOURCES = frozenset({
        SOURCE_KPMS_RAW, SOURCE_KPMS_MERGED,
    })
    if codes_path is None and code_source in _RANDOM_FALLBACK_SOURCES:
        return _generate_random(cfg, n_train, n_test, n_frames)
    if codes_path is None and code_source in (
        SOURCE_KINEMATICS_VAE,
        SOURCE_MLB,
    ):
        raise ValueError(
            f"kpms_config.code_source={code_source!r} requires "
            f"kpms_config.codes_path to be set (a tokenizer-output .npz). "
            f"Random-codes fallback is disabled for kinematics-VAE-family "
            f"sources because ``cfg.network_config.num_codes`` is unrelated "
            f"to the tokenizer's actual codebook size — silently falling "
            f"back would feed the decoder garbage. For a smoke run, set "
            f"``code_source: kpms_raw`` with ``codes_path: null`` instead."
        )

    if code_source == SOURCE_KPMS_RAW:
        return _load_kpms_npz(
            codes_path, train_indices, test_indices, n_frames,
            require_merged=False, requested_source=SOURCE_KPMS_RAW,
        )

    if code_source == SOURCE_KPMS_MERGED:
        return _load_kpms_npz(
            codes_path, train_indices, test_indices, n_frames,
            require_merged=True, requested_source=SOURCE_KPMS_MERGED,
        )

    if code_source == SOURCE_KINEMATICS_VAE:
        return _load_kpms_npz(
            codes_path, train_indices, test_indices, n_frames,
            require_merged=False, requested_source=SOURCE_KINEMATICS_VAE,
        )

    if code_source == SOURCE_MLB:
        return _load_kpms_npz(
            codes_path, train_indices, test_indices, n_frames,
            require_merged=False, requested_source=SOURCE_MLB,
        )

    if code_source == SOURCE_GT_DIRECTIONAL_TURN:
        return _load_gt_directional_turn(
            kpms_cfg, train_indices, test_indices, n_frames,
        )

    raise ValueError(
        f"Unknown kpms_config.code_source={code_source!r}. "
        f"Expected one of: {sorted(VALID_SOURCES)}."
    )


# =============================================================================
# Random codes (smoke test)
# =============================================================================


def _generate_random(cfg, n_train: int, n_test: int, n_frames: int) -> CodesPayload:
    num_codes = int(cfg.network_config.num_codes)
    log.warning(
        f"codes_path=null — generating random codes ({num_codes}) for "
        f"smoke testing"
    )
    rng = np.random.RandomState(42)
    train_codes = rng.randint(0, num_codes, size=(n_train, n_frames)).astype(
        np.int32
    )
    test_codes = rng.randint(0, num_codes, size=(n_test, n_frames)).astype(np.int32)
    return CodesPayload(
        train_codes=train_codes,
        test_codes=test_codes,
        num_codes=num_codes,
        code_source=SOURCE_RANDOM,
        provenance={"random_seed": 42},
    )


# =============================================================================
# KPMS .npz loader (raw / merged share the same file format)
# =============================================================================


def _resolve_path(p: str | Path) -> Path:
    """Resolve relative paths against the repo root."""
    pp = Path(p)
    if not pp.is_absolute():
        pp = get_repo_root() / pp
    return pp


def _load_kpms_npz(
    codes_path: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_frames: int,
    require_merged: bool,
    requested_source: str | None = None,
) -> CodesPayload:
    """Load codes from a KPMS-format .npz.

    Format (written by ``c2a.codegen.kpms.generate_codes`` / merged by
    ``c2a.codegen.kpms.merge_states.reencode_codes_npz``):

        * ``train_codes`` ``[N_train, T]`` int
        * ``test_codes``  ``[N_test,  T]`` int
        * ``train_indices`` ``[N_train]`` original H5 clip ids
        * ``test_indices``  ``[N_test]``
        * (optional) ``kappa``, ``num_states``, ``model_type``, ``mean_duration``
        * (merged only) ``is_merged=True``, ``original_num_states``,
          ``new_num_states``

    Args:
        codes_path: Path or relative path to the .npz file.
        train_indices: Expected original-id ordering for the training
            subset (must match what the npz stored).
        test_indices: Same for test.
        n_frames: Expected number of frames per clip.
        require_merged: When True, raise unless the npz carries
            ``is_merged=True``.
    """
    if codes_path is None:
        raise ValueError(
            "kpms_config.codes_path is required for KPMS code sources "
            "(set code_source: gt_directional_turn or null for smoke test)."
        )

    p = _resolve_path(codes_path)
    if not p.exists():
        raise FileNotFoundError(
            f"kpms_config.codes_path={codes_path} does not resolve to an "
            f"existing file (tried {p})."
        )

    data = dict(np.load(str(p), allow_pickle=False))
    train_codes = np.asarray(data["train_codes"])
    test_codes = np.asarray(data["test_codes"])

    # Provenance: provenance flag check first so we fail fast before doing
    # any further validation on a wrong-source file.
    is_merged = False
    if "is_merged" in data:
        is_merged = bool(np.asarray(data["is_merged"]).item())

    if require_merged and not is_merged:
        raise RuntimeError(
            f"code_source=kpms_merged requires the .npz at {p} to carry "
            f"`is_merged=True` provenance, but it does not. Re-encode the "
            f"raw codes with `c2a.codegen.kpms.merge_states.reencode_codes_npz` "
            f"(via `python -m c2a.codegen.kpms.merge_states ...`) so the file is "
            f"marked as merged. Refusing to silently treat raw codes as "
            f"merged."
        )
    if not require_merged and is_merged:
        log.warning(
            "code_source=kpms_raw but the .npz at %s is marked is_merged=True. "
            "Loading anyway, but if you intended to use the merged codes set "
            "code_source=kpms_merged.",
            p,
        )

    # Order check: the codegen scripts already align train_codes[i] to
    # train_indices[i]. If the npz's stored order doesn't match the caller's
    # expected order, the policy would see misaligned codes — silent. So we
    # verify before trusting the arrays.
    stored_train_idx = data.get("train_indices")
    stored_test_idx = data.get("test_indices")
    if stored_train_idx is not None and stored_test_idx is not None:
        if len(stored_train_idx) != len(train_indices) or not np.array_equal(
            np.asarray(stored_train_idx), train_indices
        ):
            raise RuntimeError(
                f"npz at {p} has train_indices that do not match the loader's "
                f"caller-provided train_indices. The codes file was generated "
                f"against a different balanced split. Re-run "
                f"`c2a.codegen.kpms.generate_codes` with the current split JSON."
            )
        if len(stored_test_idx) != len(test_indices) or not np.array_equal(
            np.asarray(stored_test_idx), test_indices
        ):
            raise RuntimeError(
                f"npz at {p} has test_indices that do not match the loader's "
                f"caller-provided test_indices. (See train mismatch error.)"
            )

    train_codes = train_codes.astype(np.int32, copy=False)
    test_codes = test_codes.astype(np.int32, copy=False)

    # Codebook size: the npz carries an authoritative ``num_states`` we
    # should respect. Fall back to data-derived if missing (older files).
    declared_K = data.get("num_states", None)
    if declared_K is not None:
        num_codes = int(np.asarray(declared_K).item())
    else:
        if train_codes.size == 0 and test_codes.size == 0:
            raise RuntimeError(
                f"npz at {p} has no num_states field and both code arrays "
                f"are empty; cannot derive codebook size."
            )
        num_codes = (
            int(np.maximum(train_codes.max(), test_codes.max())) + 1
        )
        log.warning(
            "npz at %s lacks 'num_states'; falling back to data-derived "
            "num_codes=%d. Older code files may underestimate the codebook "
            "size if trailing states are unused.", p, num_codes,
        )

    # Before the generic OOB check, give a more pointed hint for the
    # specific failure mode where someone manually remapped codes but
    # forgot to update num_states (e.g. legacy reencode_codes_npz before
    # the provenance patch).
    if train_codes.size > 0 or test_codes.size > 0:
        max_present = int(np.maximum(
            train_codes.max() if train_codes.size > 0 else -1,
            test_codes.max() if test_codes.size > 0 else -1,
        ))
        if max_present >= num_codes:
            raise RuntimeError(
                f"npz at {p} declares num_states={num_codes} but the codes "
                f"contain values up to {max_present}. The embedding table "
                f"would be undersized. If this file came from a remap, the "
                f"producer must update num_states (and ideally write "
                f"is_merged=True) — see c2a.codegen.kpms.merge_states.reencode_codes_npz."
            )

    _validate_codes_shape_and_range(
        train_codes, test_codes, n_train=len(train_indices),
        n_test=len(test_indices), n_frames=n_frames, num_codes=num_codes,
    )

    provenance = {
        "is_merged": is_merged,
        "kpms_kappa": _scalar_or_none(data.get("kappa")),
        "kpms_num_states": _scalar_or_none(data.get("num_states")),
        "kpms_model_type": _str_or_none(data.get("model_type")),
        "kpms_mean_duration": _scalar_or_none(data.get("mean_duration")),
        "codes_path": str(p),
    }
    if is_merged:
        provenance["original_num_states"] = _scalar_or_none(
            data.get("original_num_states")
        )
        provenance["new_num_states"] = _scalar_or_none(data.get("new_num_states"))

    # ``code_source`` echoes the user's request, NOT what the file
    # happens to carry. Auto-promoting kpms_raw -> kpms_merged because
    # the file has the flag would silently re-categorize WandB runs.
    # We log a warning above when the file is merged but the user asked
    # for raw; the actual fact lives in ``provenance.is_merged``.
    out_source = requested_source or (
        SOURCE_KPMS_MERGED if is_merged else SOURCE_KPMS_RAW
    )

    log.info(
        "Loaded KPMS codes (requested=%s, is_merged=%s) from %s: "
        "train %s, test %s, num_codes=%d",
        out_source, is_merged, p,
        train_codes.shape, test_codes.shape, num_codes,
    )

    return CodesPayload(
        train_codes=train_codes,
        test_codes=test_codes,
        num_codes=num_codes,
        code_source=out_source,
        provenance=provenance,
    )


# =============================================================================
# Ground-truth labels: 5-class directional-turn variant
# =============================================================================


def _load_gt_directional_turn(
    kpms_cfg,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_frames: int,
) -> CodesPayload:
    """Build ``gt_directional_turn`` codes from the labels CSV.

    Mapping (5 classes):

        Immobile (any direction)  -> 0
        Rear     (any direction)  -> 1
        Walk     (any direction)  -> 2
        Turn + Left               -> 3
        Turn + Right              -> 4

    ``Turn + Straight`` does not occur in the current CSV (verified
    against the balanced split). If a future CSV introduces such a
    frame, we raise rather than silently inventing a 6th class — the
    caller should pick an explicit policy first.

    UNLABELED (sentinel -1) frames are also a hard error: every frame
    in the balanced split is labeled in both tracks today, and silently
    inventing a 6th "Unlabeled" class would feed garbage embeddings to
    the policy.
    """
    # Lazy import: label_alignment lives next to KPMS code and pulling it
    # eagerly into c2a/__init__.py would couple environments together.
    from c2a.codegen.shared.label_alignment import (
        BEHAVIOR_TO_INT, DIRECTION_TO_INT, UNLABELED, load_label_frames,
    )

    csv_path = kpms_cfg.get("labels_csv_path", None)
    if csv_path is None:
        raise ValueError(
            "kpms_config.labels_csv_path must be set when "
            "code_source=gt_directional_turn."
        )
    csv_path = _resolve_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"kpms_config.labels_csv_path={csv_path} does not exist."
        )
    subsample = int(kpms_cfg.get("csv_to_h5_subsample", 2))

    # Load labels for the full possible clip range — we'll subset by
    # train/test indices below. clip_range is inclusive on both ends.
    if train_indices.size == 0 and test_indices.size == 0:
        raise RuntimeError(
            "gt_directional_turn loader received empty train and test "
            "indices; nothing to label."
        )
    max_clip_id = int(max(
        int(train_indices.max()) if train_indices.size > 0 else -1,
        int(test_indices.max()) if test_indices.size > 0 else -1,
    ))
    arr = load_label_frames(
        csv_path,
        clip_range=(0, max_clip_id),
        n_frames=n_frames,
        csv_to_h5_subsample=subsample,
    )

    # Invariant: load_label_frames returns clips in id order, so row i
    # corresponds to clip id i. If a future refactor breaks this (e.g.
    # drops unlabeled clips), our train_indices indexing would silently
    # shift the codes by however many rows were dropped. Pin the contract.
    expected_ids = np.arange(max_clip_id + 1, dtype=arr.clip_indices.dtype)
    if not np.array_equal(arr.clip_indices, expected_ids):
        raise RuntimeError(
            f"load_label_frames returned clip_indices={arr.clip_indices[:5]}... "
            f"but expected sequential ids 0..{max_clip_id}. The GT loader "
            f"relies on positional indexing; refusing to risk silent "
            f"misalignment."
        )

    # Subset to the order ReferenceClips will keep.
    train_b = arr.behavior[train_indices]
    train_d = arr.direction[train_indices]
    test_b = arr.behavior[test_indices]
    test_d = arr.direction[test_indices]

    for name, arr_ in (
        ("train", train_b), ("test", test_b),
    ):
        if (arr_ == UNLABELED).any():
            n_bad = int((arr_ == UNLABELED).sum())
            bad_clip_pos = np.unique(np.where(arr_ == UNLABELED)[0])
            raise RuntimeError(
                f"GT loader found {n_bad} UNLABELED behavior frames in the "
                f"{name} split (clip-positions {bad_clip_pos[:10].tolist()}). "
                f"The CSV must label every frame of every balanced clip in "
                f"both tracks. Refusing to invent a sentinel class."
            )
    for name, arr_ in (
        ("train", train_d), ("test", test_d),
    ):
        if (arr_ == UNLABELED).any():
            n_bad = int((arr_ == UNLABELED).sum())
            bad_clip_pos = np.unique(np.where(arr_ == UNLABELED)[0])
            raise RuntimeError(
                f"GT loader found {n_bad} UNLABELED direction frames in the "
                f"{name} split (clip-positions {bad_clip_pos[:10].tolist()}). "
                f"(Same rule as behavior.)"
            )

    # Build a (4 × 3) lookup table: M[behavior_id, direction_id] = class
    # Sentinel -1 marks "should never be hit" entries; we raise if any
    # data point lands on a sentinel slot.
    SENTINEL = -1
    M = np.full((4, 3), SENTINEL, dtype=np.int32)
    M[BEHAVIOR_TO_INT["Immobile"], :] = 0
    M[BEHAVIOR_TO_INT["Rear"], :] = 1
    M[BEHAVIOR_TO_INT["Walk"], :] = 2
    M[BEHAVIOR_TO_INT["Turn"], DIRECTION_TO_INT["Left"]] = 3
    M[BEHAVIOR_TO_INT["Turn"], DIRECTION_TO_INT["Right"]] = 4
    # M[Turn, Straight] left as SENTINEL by design.

    train_codes = M[train_b, train_d]
    test_codes = M[test_b, test_d]

    if (train_codes == SENTINEL).any():
        n_bad = int((train_codes == SENTINEL).sum())
        raise RuntimeError(
            f"GT loader hit {n_bad} (Turn+Straight) frames in the train "
            f"split that have no class in gt_directional_turn. The CSV "
            f"may have changed since the implementation was written. "
            f"Pick an explicit policy (extra class, or coalesce with "
            f"another) before continuing."
        )
    if (test_codes == SENTINEL).any():
        n_bad = int((test_codes == SENTINEL).sum())
        raise RuntimeError(
            f"GT loader hit {n_bad} (Turn+Straight) frames in the test "
            f"split. (Same rule as train.)"
        )

    # Class histograms for the run log — useful for sanity checks.
    train_hist = np.bincount(
        train_codes.ravel(), minlength=GT_DIRECTIONAL_TURN_NUM_CODES
    ).tolist()
    test_hist = np.bincount(
        test_codes.ravel(), minlength=GT_DIRECTIONAL_TURN_NUM_CODES
    ).tolist()

    _validate_codes_shape_and_range(
        train_codes, test_codes,
        n_train=len(train_indices), n_test=len(test_indices),
        n_frames=n_frames, num_codes=GT_DIRECTIONAL_TURN_NUM_CODES,
    )

    log.info(
        "Loaded gt_directional_turn codes from %s (subsample=%d): "
        "train %s, test %s, num_codes=%d.",
        csv_path, subsample, train_codes.shape, test_codes.shape,
        GT_DIRECTIONAL_TURN_NUM_CODES,
    )
    log.info("  class names: %s", GT_DIRECTIONAL_TURN_NAMES)
    log.info("  train counts: %s", train_hist)
    log.info("  test  counts: %s", test_hist)

    provenance = {
        "labels_csv_path": str(csv_path),
        "csv_to_h5_subsample": subsample,
        "gt_class_names": list(GT_DIRECTIONAL_TURN_NAMES),
        "gt_train_counts": train_hist,
        "gt_test_counts": test_hist,
    }
    return CodesPayload(
        train_codes=train_codes.astype(np.int32, copy=False),
        test_codes=test_codes.astype(np.int32, copy=False),
        num_codes=GT_DIRECTIONAL_TURN_NUM_CODES,
        code_source=SOURCE_GT_DIRECTIONAL_TURN,
        provenance=provenance,
    )


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_codes_shape_and_range(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    n_train: int,
    n_test: int,
    n_frames: int,
    num_codes: int,
) -> None:
    """Common shape / dtype / range invariants for any source.

    Enforces ``0 <= code < num_codes`` so that the embedding-table lookup
    in the policy / value networks (``nn.Embed(num_codes, ...)``) is
    guaranteed in-bounds. An OOB index is silent garbage on GPU, so this
    is the chokepoint that every code-source must pass before the
    payload escapes the loader.
    """
    expected_train = (n_train, n_frames)
    expected_test = (n_test, n_frames)
    if train_codes.shape != expected_train:
        raise RuntimeError(
            f"train_codes shape {train_codes.shape} != expected "
            f"{expected_train}."
        )
    if test_codes.shape != expected_test:
        raise RuntimeError(
            f"test_codes shape {test_codes.shape} != expected {expected_test}."
        )
    if not np.issubdtype(train_codes.dtype, np.integer):
        raise RuntimeError(
            f"train_codes must be integer; got dtype {train_codes.dtype}."
        )
    if not np.issubdtype(test_codes.dtype, np.integer):
        raise RuntimeError(
            f"test_codes must be integer; got dtype {test_codes.dtype}."
        )
    # Empty arrays make .min()/.max() raise; skip range checks on empty
    # splits and let downstream raise if it actually needs a non-empty set.
    if train_codes.size > 0:
        tmin, tmax = int(train_codes.min()), int(train_codes.max())
        if tmin < 0:
            raise RuntimeError(f"train_codes contains negative values (min={tmin}).")
        if tmax >= num_codes:
            raise RuntimeError(
                f"train_codes max ({tmax}) >= num_codes ({num_codes}); "
                f"would index out-of-bounds in nn.Embed."
            )
    if test_codes.size > 0:
        emin, emax = int(test_codes.min()), int(test_codes.max())
        if emin < 0:
            raise RuntimeError(f"test_codes contains negative values (min={emin}).")
        if emax >= num_codes:
            raise RuntimeError(
                f"test_codes max ({emax}) >= num_codes ({num_codes}); "
                f"would index out-of-bounds in nn.Embed."
            )


def _scalar_or_none(x):
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 0:
        return a.item()
    return a.tolist()


def _str_or_none(x):
    if x is None:
        return None
    a = np.asarray(x)
    if a.dtype.kind in ("U", "S"):
        return str(a.item()) if a.ndim == 0 else [str(s) for s in a.tolist()]
    return _scalar_or_none(x)
