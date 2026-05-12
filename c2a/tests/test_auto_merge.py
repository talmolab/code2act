"""Tests for the AR-HMM auto-merge criterion in c2a.codegen.kpms.merge_states.

The criterion is a closed-form Bhattacharyya distance between AR-HMM
emission distributions averaged over the empirical history. Two states
that produce nearly-identical conditional Gaussians under typical
history should get B(i,j) ≈ 0; states with different dynamics should
get B(i,j) >> 0. ``propose_merge_groups_from_distance`` then thresholds
this matrix into merge groups.

These tests build synthetic snapshots — no KPMS fit needed — to verify:
  * B(i,i) = 0 by construction.
  * Identical (Ab, Q) pairs get B ≈ 0.
  * Different (Ab, Q) pairs get B >> 0.
  * Group proposal returns connected components for 'single' linkage
    and complete-linkage groups for 'complete'.
  * old_to_new map is well-formed (every original state mapped exactly
    once, image is contiguous from 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from c2a.codegen.kpms.merge_states import (
    DEFAULT_BHATTACHARYYA_THRESHOLD,
    _bhattacharyya_pair,
    _build_history_vectors,
    _build_old_to_new_from_groups,
    compute_pairwise_bhattacharyya,
    propose_merge_groups_from_distance,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_snapshot(
    Ab: np.ndarray, Q: np.ndarray, x: np.ndarray,
) -> dict:
    """Wrap arrays into the snapshot dict shape merge_states expects."""
    K = Ab.shape[0]
    return {
        "params": {
            "Ab": Ab, "Q": Q,
            "pi": np.full((K, K), 1.0 / K),
            "betas": np.full(K, 1.0 / K),
        },
        "states": {"x": x, "z": np.zeros((x.shape[0], x.shape[1]), int)},
        "hypparams": {"ar": {"num_states": K}},
    }


# ── Bhattacharyya pair ──────────────────────────────────────────────────────


def test_bhattacharyya_self_is_zero():
    """B(i, i) must be 0 for any (Ab, Q)."""
    rng = np.random.default_rng(0)
    d, p = 4, 2
    A = rng.standard_normal((d, d * p + 1)) * 0.1
    L = rng.standard_normal((d, d)) * 0.1
    Q = L @ L.T + np.eye(d)
    z_mean = rng.standard_normal(d * p + 1)
    z_cov = np.eye(d * p + 1)
    B_self = _bhattacharyya_pair(A, Q, A, Q, z_mean, z_cov)
    assert abs(B_self) < 1e-9


def test_bhattacharyya_increases_with_distance():
    """Two AR-HMM states that are perturbations of each other should
    yield monotonically increasing B as the perturbation grows."""
    rng = np.random.default_rng(1)
    d, p = 3, 1
    A0 = rng.standard_normal((d, d * p + 1)) * 0.1
    L = rng.standard_normal((d, d)) * 0.1
    Q = L @ L.T + 0.5 * np.eye(d)
    z_mean = np.zeros(d * p + 1); z_mean[-1] = 1.0
    z_cov = np.eye(d * p + 1)

    perturbations = [0.0, 0.1, 0.5, 1.5]
    Bs = []
    for eps in perturbations:
        A1 = A0 + eps * rng.standard_normal(A0.shape)
        Bs.append(_bhattacharyya_pair(A0, Q, A1, Q, z_mean, z_cov))
    # Strictly increasing (the eps=0 case is exact zero).
    for prev, curr in zip(Bs, Bs[1:]):
        assert curr >= prev - 1e-9
    assert Bs[-1] > 0.5  # large perturbation → easily distinguishable


def test_bhattacharyya_symmetric():
    rng = np.random.default_rng(2)
    d, p = 3, 1
    A_i = rng.standard_normal((d, d * p + 1)) * 0.1
    A_j = rng.standard_normal((d, d * p + 1)) * 0.3
    Q_i = np.eye(d) + 0.3 * np.outer(rng.standard_normal(d), rng.standard_normal(d))
    Q_i = Q_i @ Q_i.T  # SPD
    Q_j = np.eye(d) * 1.5
    z_mean = np.zeros(d * p + 1); z_mean[-1] = 1.0
    z_cov = np.eye(d * p + 1)
    B_ij = _bhattacharyya_pair(A_i, Q_i, A_j, Q_j, z_mean, z_cov)
    B_ji = _bhattacharyya_pair(A_j, Q_j, A_i, Q_i, z_mean, z_cov)
    assert abs(B_ij - B_ji) < 1e-9


# ── History vector construction ──────────────────────────────────────────────


def test_history_vectors_shape_and_bias():
    """For x of shape (N, T, d) and nlags=p, output is (N*(T-p), d*p+1)
    with the trailing column all 1.0."""
    rng = np.random.default_rng(3)
    N, T, d, p = 4, 7, 3, 2
    x = rng.standard_normal((N, T, d))
    z = _build_history_vectors(x, mask=None, nlags=p)
    assert z.shape == (N * (T - p), d * p + 1)
    assert np.allclose(z[:, -1], 1.0)
    # Lag block 0 is x_{t-1}; for t = p..T-1 in clip 0 that is x[0, p-1..T-2, :].
    expected_lag0_clip0 = x[0, p - 1:T - 1, :]
    np.testing.assert_allclose(z[:T - p, :d], expected_lag0_clip0)


def test_history_vectors_raises_on_too_short():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="T="):
        _build_history_vectors(rng.standard_normal((2, 2, 3)), None, nlags=2)


# ── Pairwise B on synthetic K=4 model with two duplicate pairs ──────────────


def test_pairwise_bhattacharyya_finds_duplicates():
    """K=4 model: states 0+1 share emissions; states 2+3 share emissions
    (different from 0+1). The K×K B-matrix must show B[0,1] ≈ B[2,3] ≈ 0
    and all other off-diagonal entries clearly above threshold."""
    rng = np.random.default_rng(5)
    d, p = 3, 2

    # Two distinct emission profiles.
    A_alpha = rng.standard_normal((d, d * p + 1)) * 0.1
    L_alpha = rng.standard_normal((d, d)) * 0.1
    Q_alpha = L_alpha @ L_alpha.T + np.eye(d)

    A_beta = rng.standard_normal((d, d * p + 1)) * 0.1 + 2.0  # very different
    L_beta = rng.standard_normal((d, d)) * 0.2 + 0.5 * np.eye(d)
    Q_beta = L_beta @ L_beta.T + np.eye(d)

    Ab = np.stack([A_alpha, A_alpha, A_beta, A_beta])
    Q  = np.stack([Q_alpha, Q_alpha, Q_beta, Q_beta])
    x = rng.standard_normal((10, 30, d))
    snap = _make_snapshot(Ab, Q, x)

    B = compute_pairwise_bhattacharyya(snap)
    assert B.shape == (4, 4)
    assert np.allclose(np.diag(B), 0)
    # Symmetric.
    np.testing.assert_allclose(B, B.T, atol=1e-9)
    # Duplicate pairs ≈ 0.
    assert B[0, 1] < 1e-6
    assert B[2, 3] < 1e-6
    # Different-profile pairs are large.
    assert min(B[0, 2], B[0, 3], B[1, 2], B[1, 3]) > 0.5


# ── Group proposal ──────────────────────────────────────────────────────────


def test_propose_groups_complete_linkage():
    """B-matrix with two clear clusters → complete-linkage groups them."""
    B = np.array([
        [0.00, 0.01, 5.0,  4.5],
        [0.01, 0.00, 4.8,  5.0],
        [5.0,  4.8,  0.00, 0.02],
        [4.5,  5.0,  0.02, 0.00],
    ])
    groups = propose_merge_groups_from_distance(
        B, threshold=0.3, linkage="complete",
    )
    assert sorted(groups) == [[0, 1], [2, 3]]


def test_propose_groups_single_linkage_chains():
    """Single-linkage transitively chains via low-distance edges.

    A-B distance 0.1, B-C distance 0.1, A-C distance 1.0 → all three
    merge under 'single' (the chain), but NOT under 'complete' (since
    A-C exceeds threshold)."""
    B = np.array([
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.1],
        [1.0, 0.1, 0.0],
    ])
    groups_single = propose_merge_groups_from_distance(
        B, threshold=0.3, linkage="single",
    )
    assert groups_single == [[0, 1, 2]]
    groups_complete = propose_merge_groups_from_distance(
        B, threshold=0.3, linkage="complete",
    )
    # Complete linkage refuses to merge 0 with 2; depending on the
    # tiebreak only one of {0,1} or {1,2} is grouped, the other is lone.
    # Validate weaker invariant: no group contains both 0 AND 2.
    for g in groups_complete:
        assert not (0 in g and 2 in g)


def test_propose_groups_no_edges_returns_empty():
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert propose_merge_groups_from_distance(B, threshold=0.3) == []


def test_propose_groups_invalid_linkage():
    B = np.zeros((2, 2))
    with pytest.raises(ValueError, match="linkage"):
        propose_merge_groups_from_distance(B, threshold=0.3, linkage="bogus")


def test_propose_groups_K_lt_2():
    assert propose_merge_groups_from_distance(np.zeros((1, 1))) == []
    assert propose_merge_groups_from_distance(np.zeros((0, 0))) == []


# ── old_to_new map ───────────────────────────────────────────────────────────


def test_old_to_new_well_formed():
    """For K=6 with groups [[1,3],[2,5]], the map must:
       - cover every state in [0, 6),
       - map all members of a group to the same new id,
       - produce a contiguous image starting at 0,
       - produce |image| = K - sum(|g|-1) = 6 - 2 = 4 distinct new ids.
    """
    K, groups = 6, [[1, 3], [2, 5]]
    o2n = _build_old_to_new_from_groups(groups, K)
    assert set(o2n.keys()) == set(range(K))
    assert o2n[1] == o2n[3]
    assert o2n[2] == o2n[5]
    image = sorted(set(o2n.values()))
    assert image == list(range(len(image)))
    assert len(image) == 4


def test_old_to_new_no_groups_is_identity_renumber():
    """Empty groups → each old id gets its own new id (compact)."""
    K = 5
    o2n = _build_old_to_new_from_groups([], K)
    assert sorted(o2n.values()) == list(range(K))


# ── DEFAULT threshold sanity ────────────────────────────────────────────────


def test_default_threshold_in_reasonable_range():
    """Sanity check against module-level constant; if someone moves it
    below 0 or above 5 they need to look at this test."""
    assert 0 < DEFAULT_BHATTACHARYYA_THRESHOLD < 5
