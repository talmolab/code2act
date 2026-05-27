"""Unit tests for c2a.experiments.shared.tokenizer_metrics.

The metric math is unit-free and tokenizer-agnostic, so we test
against constructed code arrays with known-truth expected values.
"""

import numpy as np

from c2a.experiments.shared.tokenizer_metrics import (
    TokenizerMetrics,
    compute_tokenizer_metrics,
)


def test_single_clip_two_bouts():
    """[0,0,0,1,1] -> code 0: 1 bout of 3, code 1: 1 bout of 2."""
    codes = np.array([[0, 0, 0, 1, 1]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=2, fps=50.0)
    assert m.num_codes == 2
    assert m.n_frames_total == 5
    np.testing.assert_array_equal(m.frame_frequency, [3, 2])
    np.testing.assert_array_equal(m.bout_instances, [1, 1])
    np.testing.assert_array_equal(sorted(m.bout_lengths.tolist()), [2, 3])
    assert m.n_bouts == 2
    assert m.mean_duration_frames == 2.5
    assert m.median_duration_frames == 2.5
    # 50 fps → 20 ms/frame
    assert abs(m.mean_duration_ms - 50.0) < 1e-9


def test_multi_clip_separate_bouts():
    """[[0,0],[0,0]] is two clips each with a single 2-frame bout of code 0.

    Critically, bouts do NOT span clip boundaries — even though the
    end of clip 0 and start of clip 1 are both code 0, they count as
    SEPARATE bouts because they're independent clips.
    """
    codes = np.array([[0, 0], [0, 0]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=2, fps=50.0)
    assert m.bout_instances[0] == 2
    assert m.bout_instances[1] == 0
    np.testing.assert_array_equal(m.frame_frequency, [4, 0])
    np.testing.assert_array_equal(sorted(m.bout_lengths.tolist()), [2, 2])


def test_alternating_code():
    """[0,1,0,1] -> 4 bouts of length 1 each."""
    codes = np.array([[0, 1, 0, 1]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=2, fps=50.0)
    np.testing.assert_array_equal(m.frame_frequency, [2, 2])
    np.testing.assert_array_equal(m.bout_instances, [2, 2])
    assert m.n_bouts == 4
    assert (m.bout_lengths == 1).all()
    assert m.mean_duration_frames == 1.0


def test_active_thresholds():
    """30 frames in code 0, 1 frame in code 1, 0 in codes 2-9.

    - active_any: code 0 (30) and code 1 (1) → 2
    - active_at_1pct: threshold = max(int(31*0.01), 1) = 1. Both pass → 2.
    - active_at_0p1pct: threshold = max(int(31*0.001), 1) = 1. Both pass → 2.
    With a larger setup:
    """
    codes = np.zeros((1, 31), dtype=np.int32)
    codes[0, -1] = 1
    m = compute_tokenizer_metrics(codes, num_codes=10, fps=50.0)
    assert m.active_any == 2
    # All non-empty codes pass 1-frame floor; 31 frames -> 1% = 0 raw,
    # max(0, 1) = 1.
    assert m.active_at_1pct == 2
    assert m.active_at_0p1pct == 2

    # Now scale up: 1000 frames, code 1 used only 1 time = 0.1% boundary.
    big = np.zeros((1, 1000), dtype=np.int32)
    big[0, -1] = 1
    m2 = compute_tokenizer_metrics(big, num_codes=10, fps=50.0)
    # 1% threshold = 10 frames, code 1 has 1 frame → fails.
    assert m2.active_at_1pct == 1
    # 0.1% threshold = 1 frame, code 1 has 1 frame → passes.
    assert m2.active_at_0p1pct == 2


def test_sentinel_excluded():
    """Negative codes are excluded — they don't split bouts AND don't
    contribute to frame_frequency.

    Sequence: [0, 0, -1, -1, 0, 0]. Without sentinel handling that's
    one bout of 6 frames in code 0 (zeros) plus the sentinels.
    With sentinel handling: bouts are runs of the SAME value, so
    [0,0,-1,-1,0,0] is three runs: (code=0, len=2), (code=-1, len=2),
    (code=0, len=2). We skip the sentinel run; we keep the two len-2
    code-0 bouts as separate.
    """
    codes = np.array([[0, 0, -1, -1, 0, 0]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=1, fps=50.0)
    assert m.bout_instances[0] == 2
    assert m.n_frames_total == 4  # sentinels excluded from count
    assert m.frame_frequency[0] == 4


def test_to_table_row_keys():
    codes = np.array([[0, 0, 1, 1]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=2, fps=50.0)
    row = m.to_table_row(family="test", config="dummy")
    expected_keys = {
        "family", "config", "K", "n_frames_total",
        "active_any", "active_at_1pct", "active_at_0p1pct", "usage_ratio",
        "n_bouts_total", "instances_per_code_min",
        "instances_per_code_median", "instances_per_code_max",
        "mean_duration_frames", "median_duration_frames",
        "std_duration_frames", "mean_duration_ms", "median_duration_ms",
    }
    assert set(row.keys()) == expected_keys
    assert row["family"] == "test"
    assert row["config"] == "dummy"
    assert row["K"] == 2


def test_empty_codes_raises():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        compute_tokenizer_metrics(
            np.zeros((0, 5), dtype=np.int32), num_codes=2,
        )


def test_unused_top_codes():
    """K=10 declared, but only codes 0..2 used. Confirms num_codes
    override path doesn't trim the K.
    """
    codes = np.array([[0, 1, 2, 0, 1, 2]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=10, fps=50.0)
    assert m.num_codes == 10
    assert m.frame_frequency.shape == (10,)
    assert (m.frame_frequency[3:] == 0).all()
    assert m.active_any == 3


def test_isinstance_check():
    codes = np.array([[0, 1]], dtype=np.int32)
    m = compute_tokenizer_metrics(codes, num_codes=2, fps=50.0)
    assert isinstance(m, TokenizerMetrics)
