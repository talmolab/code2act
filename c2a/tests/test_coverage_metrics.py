"""Tests for c2a.codegen.kpms.coverage_metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from c2a.codegen.kpms.coverage_metrics import (
    compute_coverage,
    compute_nmi,
    compute_purity,
)
from c2a.codegen.kpms.label_alignment import UNLABELED


def test_nmi_perfect_match_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    syllables = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert compute_nmi(syllables, labels) == pytest.approx(1.0)


def test_nmi_deterministic_relabeling_is_one():
    """NMI is invariant under relabeling — different IDs but same partition → 1."""
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    syllables = np.array([9, 9, 3, 3, 5, 5], dtype=np.int64)
    assert compute_nmi(syllables, labels) == pytest.approx(1.0)


def test_nmi_independent_is_near_zero():
    rng = np.random.RandomState(0)
    T = 10_000
    labels = rng.randint(0, 6, size=T).astype(np.int64)
    syllables = rng.randint(0, 20, size=T).astype(np.int64)
    nmi = compute_nmi(syllables, labels)
    assert 0.0 <= nmi < 0.05  # arithmetic NMI on independent uniform → ~0


def test_nmi_ignores_unlabeled_frames():
    labels = np.array([0, 0, UNLABELED, UNLABELED, 1, 1], dtype=np.int64)
    syllables = np.array([0, 0, 99, 99, 1, 1], dtype=np.int64)
    # If UNLABELED were treated as a class, sylable=99 vs label=-1 would
    # contribute signal. Masking → metric computed only on the labeled subset
    # which is a perfect match → 1.0.
    assert compute_nmi(syllables, labels) == pytest.approx(1.0)


def test_nmi_both_constant_is_one():
    """Degenerate case: both sequences are constant → H(X)=H(Y)=0. sklearn
    treats this as perfect agreement and returns 1.0 (the two partitions
    are trivially identical: each has exactly one cluster). This is the
    opposite of what you might naively expect, so pin it down with a test."""
    labels = np.zeros(10, dtype=np.int64)
    syllables = np.zeros(10, dtype=np.int64)
    assert compute_nmi(syllables, labels) == pytest.approx(1.0)


def test_nmi_one_constant_one_varied_is_zero():
    """If exactly one side has zero entropy (e.g. labels are all identical
    but syllables vary), the sequences are trivially independent →
    sklearn returns 0.0."""
    labels = np.zeros(10, dtype=np.int64)
    syllables = np.arange(10, dtype=np.int64) % 3
    assert compute_nmi(syllables, labels) == pytest.approx(0.0)


def test_compute_coverage_rejects_duplicate_clip_indices():
    """Defensive: duplicate all_clip_indices would silently mis-map rows."""
    all_clip_indices = np.array([10, 10, 20], dtype=np.int32)
    syllables_list = [np.zeros(5, dtype=np.int64) for _ in range(3)]
    labeled_clip_indices = np.array([10], dtype=np.int32)
    frame_labels = np.zeros((1, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="unique"):
        compute_coverage(
            syllables_list, frame_labels, labeled_clip_indices,
            all_clip_indices, n_syll=2, n_beh=2,
        )


def test_nmi_all_unlabeled_returns_nan():
    labels = np.full(10, UNLABELED, dtype=np.int64)
    syllables = np.arange(10, dtype=np.int64)
    assert math.isnan(compute_nmi(syllables, labels))


def test_purity_perfect_match_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    syllables = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    purity, mapping = compute_purity(syllables, labels, n_syll=3, n_beh=3)
    assert purity == pytest.approx(1.0)
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_purity_handles_permuted_labels():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    syllables = np.array([2, 2, 0, 0, 1, 1], dtype=np.int64)
    purity, mapping = compute_purity(syllables, labels, n_syll=3, n_beh=3)
    assert purity == pytest.approx(1.0)
    # Hungarian should recover the permutation.
    assert mapping == {0: 1, 1: 2, 2: 0}


def test_purity_cardinality_mismatch_more_syllables():
    """n_syll=10, n_beh=6 → linear_sum_assignment pairs 6 syllables with
    behaviors, returns mapping of size 6. Remaining syllables are unmatched."""
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    syllables = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    # Pretend the fit has 10 codes but only 6 were active.
    purity, mapping = compute_purity(syllables, labels, n_syll=10, n_beh=6)
    assert purity == pytest.approx(1.0)
    assert len(mapping) == 6  # min(n_syll=10, n_beh=6)


def test_purity_cardinality_mismatch_more_behaviors():
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    syllables = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=np.int64)
    # n_syll=2, n_beh=4 → only 2 matched pairs.
    purity, mapping = compute_purity(syllables, labels, n_syll=2, n_beh=4)
    assert 0.0 <= purity <= 1.0
    assert len(mapping) == 2


def test_purity_raises_on_out_of_range_ids():
    labels = np.array([0, 1], dtype=np.int64)
    syllables = np.array([0, 7], dtype=np.int64)
    with pytest.raises(ValueError, match="syllable id"):
        compute_purity(syllables, labels, n_syll=5, n_beh=2)


def test_compute_coverage_happy_path():
    """Two labeled clips both present in the fit set — check positional lookup."""
    # Fit set: clips [10, 20, 30] positionally. `syllables_list[1]` = clip 20.
    all_clip_indices = np.array([10, 20, 30], dtype=np.int32)
    syllables_list = [
        np.full(5, 7, dtype=np.int64),   # clip 10 — unlabeled clip, will be ignored
        np.array([0, 0, 1, 1, 2], dtype=np.int64),  # clip 20 — matches behavior
        np.array([1, 1, 0, 0, 2], dtype=np.int64),  # clip 30 — permuted
    ]
    # Labels: only clips 20 and 30 are labeled.
    labeled_clip_indices = np.array([20, 30], dtype=np.int32)
    frame_labels = np.array([
        [0, 0, 1, 1, 2],
        [1, 1, 0, 0, 2],
    ], dtype=np.int32)

    out = compute_coverage(
        syllables_list, frame_labels, labeled_clip_indices, all_clip_indices,
        n_syll=3, n_beh=3,
    )
    assert out["n_labeled_clips_used"] == 2
    assert out["n_labeled_frames_used"] == 10
    assert out["nmi_global"] == pytest.approx(1.0)
    assert out["purity_global"] == pytest.approx(1.0)


def test_compute_coverage_empty_intersection():
    """Labeled clips don't overlap with fit clips → returns {}, no crash."""
    all_clip_indices = np.array([1, 2, 3], dtype=np.int32)
    syllables_list = [np.zeros(5, dtype=np.int64) for _ in range(3)]
    labeled_clip_indices = np.array([100, 200], dtype=np.int32)
    frame_labels = np.zeros((2, 5), dtype=np.int32)

    out = compute_coverage(
        syllables_list, frame_labels, labeled_clip_indices, all_clip_indices,
        n_syll=3, n_beh=3,
    )
    assert out == {}


def test_compute_coverage_all_frames_unlabeled():
    """Intersection non-empty, but all frames in those clips are UNLABELED."""
    all_clip_indices = np.array([10, 20], dtype=np.int32)
    syllables_list = [np.zeros(5, dtype=np.int64), np.ones(5, dtype=np.int64)]
    labeled_clip_indices = np.array([10, 20], dtype=np.int32)
    frame_labels = np.full((2, 5), UNLABELED, dtype=np.int32)

    out = compute_coverage(
        syllables_list, frame_labels, labeled_clip_indices, all_clip_indices,
        n_syll=2, n_beh=2,
    )
    assert out == {}


def test_compute_coverage_length_mismatch_uses_shorter():
    """If syllable sequence is shorter than label row, truncate to shorter."""
    all_clip_indices = np.array([10], dtype=np.int32)
    # Only 3 frames in the fit.
    syllables_list = [np.array([0, 0, 1], dtype=np.int64)]
    labeled_clip_indices = np.array([10], dtype=np.int32)
    # 5 frames of labels; the last 2 should be discarded.
    frame_labels = np.array([[0, 0, 1, 1, 1]], dtype=np.int32)

    out = compute_coverage(
        syllables_list, frame_labels, labeled_clip_indices, all_clip_indices,
        n_syll=2, n_beh=2,
    )
    assert out["n_labeled_frames_used"] == 3
    assert out["nmi_global"] == pytest.approx(1.0)
