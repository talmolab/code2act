"""Coverage metrics that compare KPMS syllable labels to human behavior labels.

Two metrics, both cardinality-agnostic so a KPMS model with any number of
syllables can be compared against a fixed 6-behavior vocabulary:

* **NMI** — ``sklearn.metrics.normalized_mutual_info_score`` with the default
  arithmetic average.  Range ``[0, 1]``; 0 = independent, 1 = deterministic
  one-to-one mapping.
* **Purity (Hungarian-matched)** — build the confusion matrix
  ``C[n_syll, n_beh]`` on labeled frames, maximize overlap via
  ``scipy.optimize.linear_sum_assignment``, report the matched-cell fraction.

Both metrics ignore frames where the behavior label is ``UNLABELED`` (the
sentinel from ``c2a.codegen.kpms.label_alignment``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

from c2a.codegen.kpms.label_alignment import UNLABELED

log = logging.getLogger(__name__)


def compute_nmi(
    syllables: np.ndarray,
    labels: np.ndarray,
    ignore: int = UNLABELED,
) -> float:
    """Normalized mutual information between syllables and behavior labels.

    Frames where ``labels == ignore`` are masked out before the metric is
    computed, so the sentinel value is never treated as a class.
    """
    mask = labels != ignore
    if not mask.any():
        return float("nan")
    return float(
        normalized_mutual_info_score(
            labels[mask].astype(np.int64),
            syllables[mask].astype(np.int64),
            average_method="arithmetic",
        )
    )


def compute_purity(
    syllables: np.ndarray,
    labels: np.ndarray,
    n_syll: int,
    n_beh: int,
    ignore: int = UNLABELED,
) -> tuple[float, dict[int, int]]:
    """Hungarian-matched cluster purity.

    Builds a rectangular confusion matrix ``C[n_syll, n_beh]`` over labeled
    frames, solves ``linear_sum_assignment(-C)`` to pair min(n_syll, n_beh)
    syllables with behaviors that maximize total overlap, and reports
    ``matched_overlap / total_labeled_frames``.

    Args:
        syllables: Integer syllable IDs.
        labels: Integer behavior labels.
        n_syll: Total number of syllable codes (rows of C).
        n_beh: Total number of behavior classes (columns of C).
        ignore: Sentinel value in ``labels`` that should be masked.

    Returns:
        ``(purity, mapping)`` where ``mapping`` is a ``{syll_id: beh_id}``
        dict for the matched syllables.  Unmatched syllables are not in the
        dict (when ``n_syll > n_beh``, some syllables have no match).
    """
    mask = labels != ignore
    if not mask.any():
        return float("nan"), {}

    s = syllables[mask].astype(np.int64)
    b = labels[mask].astype(np.int64)

    # Sanity: syllable IDs must fit the declared codebook.
    if s.max() >= n_syll or s.min() < 0:
        raise ValueError(
            f"syllable id out of range: got min={s.min()}, max={s.max()}, "
            f"expected [0, {n_syll})."
        )
    if b.max() >= n_beh or b.min() < 0:
        raise ValueError(
            f"label id out of range: got min={b.min()}, max={b.max()}, "
            f"expected [0, {n_beh})."
        )

    confusion = np.zeros((n_syll, n_beh), dtype=np.int64)
    np.add.at(confusion, (s, b), 1)

    # linear_sum_assignment minimizes; negate to maximize overlap.
    row_ind, col_ind = linear_sum_assignment(-confusion)
    matched = int(confusion[row_ind, col_ind].sum())
    total = int(confusion.sum())
    purity = matched / total if total > 0 else float("nan")

    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return float(purity), mapping


def compute_coverage(
    syllables_list: list[np.ndarray],
    frame_labels: np.ndarray,
    labeled_clip_indices: np.ndarray,
    all_clip_indices: np.ndarray,
    n_syll: int,
    n_beh: int,
) -> dict[str, Any]:
    """Aggregate NMI + purity over the labeled subset of the sweep data.

    Intersects the labeled clips with the clips actually fit by KPMS (the
    balanced-split subset), looks up each matching clip's syllable sequence
    in ``syllables_list`` (positionally indexed via ``all_clip_indices``),
    concatenates the (syllables, labels) pairs, and computes global metrics
    on the pooled array.

    Args:
        syllables_list: Per-clip KPMS label arrays, one per entry in
            ``all_clip_indices``. ``syllables_list[pos]`` has shape
            ``[n_frames]`` and corresponds to clip ``all_clip_indices[pos]``.
        frame_labels: Output of
            ``c2a.codegen.kpms.label_alignment.load_behavior_frames`` — shape
            ``[N_labeled_range, n_frames]``, dtype int32.
        labeled_clip_indices: Corresponding clip index array of shape
            ``[N_labeled_range]``.
        all_clip_indices: Clip indices passed through KPMS (balanced split),
            positionally aligned with ``syllables_list``.
        n_syll: Number of syllable states in this KPMS model.
        n_beh: Number of behavior classes (``len(BEHAVIORS)`` = 6).

    Returns:
        Dict with keys ``nmi_global, purity_global, n_labeled_clips_used,
        n_labeled_frames_used, coverage_mapping`` (str keys for JSON).
        Returns ``{}`` with a warning if no labeled clip intersects the fit
        set or no labeled frames remain.
    """
    all_set = set(int(c) for c in all_clip_indices)
    if len(all_set) != len(all_clip_indices):
        raise ValueError(
            f"all_clip_indices must be unique (got {len(all_clip_indices)} "
            f"entries, {len(all_set)} unique). Duplicate clip indices would "
            f"silently mis-map syllable rows."
        )
    all_pos = {int(c): i for i, c in enumerate(all_clip_indices)}

    paired_syllables: list[np.ndarray] = []
    paired_labels: list[np.ndarray] = []
    n_clips_used = 0

    for i, clip_idx in enumerate(labeled_clip_indices):
        ci = int(clip_idx)
        if ci not in all_set:
            continue
        pos = all_pos[ci]
        syll_row = np.asarray(syllables_list[pos])
        label_row = frame_labels[i]
        # Respect the shorter length — guards against a fit at fewer frames.
        n = min(syll_row.shape[0], label_row.shape[0])
        paired_syllables.append(syll_row[:n])
        paired_labels.append(label_row[:n])
        n_clips_used += 1

    if n_clips_used == 0:
        log.warning(
            "compute_coverage: no labeled clips intersect the fit set "
            "(%d labeled vs %d fit). Skipping coverage metrics.",
            len(labeled_clip_indices),
            len(all_clip_indices),
        )
        return {}

    syll_flat = np.concatenate(paired_syllables)
    label_flat = np.concatenate(paired_labels)
    n_labeled_frames = int((label_flat != UNLABELED).sum())

    if n_labeled_frames == 0:
        log.warning(
            "compute_coverage: %d clips intersect the fit set but 0 labeled "
            "frames survive the UNLABELED mask. Skipping coverage metrics.",
            n_clips_used,
        )
        return {}

    nmi = compute_nmi(syll_flat, label_flat)
    purity, mapping = compute_purity(syll_flat, label_flat, n_syll, n_beh)

    return {
        "nmi_global": nmi,
        "purity_global": purity,
        "n_labeled_clips_used": n_clips_used,
        "n_labeled_frames_used": n_labeled_frames,
        # JSON-compatible: str keys, int values.
        "coverage_mapping": {str(k): int(v) for k, v in mapping.items()},
    }
