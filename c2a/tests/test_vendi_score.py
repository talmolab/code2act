"""Tests for c2a.code2code.diversity (Vendi score)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.code2code.diversity import (
    grouped_vendi_per_step,
    grouped_vendi_per_timestep,
    vendi_score_cosine,
)


# -------------------- single-group VS sanity checks --------------------


def test_vendi_identical_returns_one():
    """All-identical sample has effective number of modes = 1."""
    Z = jnp.tile(jnp.array([1.0, 2.0, 3.0, 4.0]), (10, 1))
    vs = float(vendi_score_cosine(Z))
    assert abs(vs - 1.0) < 1e-4, f"VS for 10 identical vectors should be ~1, got {vs}"


def test_vendi_orthonormal_basis_under_cosine_shifted_kernel():
    """Orthonormal basis under cosine-shifted kernel gives a specific
    sub-n value (NOT the canonical "VS = n" — that identity holds only
    under the identity kernel, not the cosine-shifted one).

    Analytical derivation for n=8:
      cos(e_i, e_j) = δ_ij, so 0.5*(1 + cos) gives K_ii=1 and K_off=0.5.
      Eigenvalues of K:
        λ_1 = 1 + 0.5·(n−1) = 4.5    (the all-ones direction eigenvalue)
        λ_2..λ_n = 0.5                (the orthogonal-to-all-ones eigvals)
      p_1 = 4.5/8 = 0.5625, p_others = 0.5/8 = 0.0625
      H = −0.5625·log(0.5625) − 7·0.0625·log(0.0625) ≈ 1.5365
      VS = exp(H) ≈ 4.6504
    """
    n = 8
    Z = jnp.eye(n)
    vs = float(vendi_score_cosine(Z))
    expected = float(
        jnp.exp(-0.5625 * jnp.log(0.5625) - 7 * 0.0625 * jnp.log(0.0625))
    )
    assert abs(vs - expected) < 1e-3, (
        f"VS for n=8 orthonormal basis under cosine-shifted kernel should "
        f"be ≈ {expected:.4f}; got {vs:.4f}"
    )
    # Sanity bound: VS strictly between 1 (all identical) and n (canonical
    # max under the identity kernel).
    assert 1.0 < vs < n


def test_vendi_truly_distinct_under_cosine():
    """If we use vectors that are also negative of each other so cos=-1,
    the cosine-shifted kernel 0.5*(1 + cos) gives K_off = 0 → true identity
    in the shifted kernel space → VS = n."""
    # Pair each basis vector with its negation: cos(e_i, -e_i) = -1.
    n = 4
    Z = jnp.concatenate([jnp.eye(n), -jnp.eye(n)], axis=0)  # (2n, n)
    # cos(e_i, e_j) for i != j == 0, cos(e_i, -e_j) for i != j == 0,
    # cos(e_i, -e_i) == -1.
    # So K_cos has -1's pairing each item with its negation, 0's elsewhere,
    # and 1's on diagonal.  After 0.5*(K_cos + 1):
    # diagonal = 1, anti-paired (i, -i) entries = 0, all other = 0.5.
    # This is NOT the identity, so VS < 2n.  We just confirm VS is > 1 here.
    vs = float(vendi_score_cosine(Z))
    assert vs > 1.0, f"VS should exceed 1 for non-degenerate set; got {vs}"


def test_vendi_two_clusters_exact_value():
    """Two distinct equal-sized clusters: VS is the analytically derived value.

    K decomposes as the tensor product of:
        M = [[1, 0.5], [0.5, 1]]   (cluster-pair structure)
        J = ones(5, 5)             (within-cluster all-ones)
    Eigvals of M: {1.5, 0.5}.  Eigvals of J: {5, 0, 0, 0, 0}.
    Eigvals of K = M ⊗ J: {1.5·5, 0.5·5} = {7.5, 2.5} plus eight zeros.
    Trace = 10 ✓.  p = [0.75, 0.25, 0, ..., 0].
    H = −0.75·log(0.75) − 0.25·log(0.25) ≈ 0.5623
    VS = exp(H) ≈ 1.7548
    """
    A = jnp.array([1.0, 0.0, 0.0])
    B = jnp.array([0.0, 1.0, 0.0])
    Z = jnp.concatenate([jnp.tile(A, (5, 1)), jnp.tile(B, (5, 1))], axis=0)
    vs = float(vendi_score_cosine(Z))
    expected = float(jnp.exp(-0.75 * jnp.log(0.75) - 0.25 * jnp.log(0.25)))
    assert vs == pytest.approx(expected, abs=1e-3), (
        f"VS for two distinct 5+5 clusters under cosine-shifted kernel "
        f"should be ≈ {expected:.4f}; got {vs:.4f}"
    )


def test_vendi_intermediate():
    """Mix of identical and random items gives VS strictly between 1 and n."""
    rng = np.random.default_rng(0)
    A = jnp.array([1.0, 0.0, 0.0, 0.0])
    Z_identical = jnp.tile(A, (5, 1))
    Z_random = jnp.array(rng.normal(size=(5, 4)).astype(np.float32))
    Z = jnp.concatenate([Z_identical, Z_random], axis=0)
    vs = float(vendi_score_cosine(Z))
    n = Z.shape[0]
    assert 1.0 < vs < float(n), f"VS should be in (1, {n}); got {vs}"


def test_vendi_monotone_with_noise():
    """Adding noise to identical items monotonically increases VS."""
    rng = np.random.default_rng(42)
    A = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
    base = jnp.tile(A, (10, 1))
    noise_levels = [0.0, 0.01, 0.1, 0.5, 1.0]
    vs_values = []
    for σ in noise_levels:
        if σ == 0.0:
            Z = base
        else:
            noise = jnp.array(rng.normal(scale=σ, size=base.shape).astype(np.float32))
            Z = base + noise
        vs_values.append(float(vendi_score_cosine(Z)))
    # Each successive σ should give VS ≥ previous (with floating slack).
    for i in range(1, len(vs_values)):
        assert vs_values[i] >= vs_values[i - 1] - 1e-3, (
            f"VS should be monotone-nondecreasing in noise; "
            f"got {vs_values=} for {noise_levels=}"
        )


def test_vendi_single_item():
    """Edge case: n=1 → VS = 1 (no diversity possible)."""
    Z = jnp.array([[1.0, 2.0, 3.0]])
    vs = float(vendi_score_cosine(Z))
    assert abs(vs - 1.0) < 1e-5


def test_vendi_under_jit():
    """JIT-compiled version returns same result as eager."""
    rng = np.random.default_rng(7)
    Z = jnp.array(rng.normal(size=(15, 24)).astype(np.float32))
    vs_eager = float(vendi_score_cosine(Z))
    vs_jit = float(jax.jit(vendi_score_cosine)(Z))
    assert abs(vs_eager - vs_jit) < 1e-4


# -------------------- grouped per-step Vendi tests --------------------


def test_grouped_vendi_small_group_returns_zero():
    """Codes with < min_group_size members get diversity = 0."""
    # 8 envs total: 3 in code 0, 5 in code 1.
    # min_group_size=5 → code 0 should get zero, code 1 should get VS.
    z_step = jnp.eye(8)                              # 8 orthogonal vectors
    c_step = jnp.array([0, 0, 0, 1, 1, 1, 1, 1])
    r = grouped_vendi_per_step(
        z_step=z_step,
        c_step=c_step,
        num_codes=2,
        min_group_size=5,
    )
    # Code 0 envs (indices 0, 1, 2): should be 0.
    assert float(r[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(r[1]) == pytest.approx(0.0, abs=1e-5)
    assert float(r[2]) == pytest.approx(0.0, abs=1e-5)
    # Code 1 envs (indices 3..7): should be log(VS) > 0 since they're 5
    # distinct orthonormal vectors under cosine kernel.
    for i in range(3, 8):
        assert float(r[i]) > 0.0, f"Code-1 env {i} should have positive log VS"
    # All code-1 envs should have the same diversity reward (they're in
    # the same group).
    assert float(r[3]) == pytest.approx(float(r[7]), abs=1e-5)


def test_grouped_vendi_all_identical_in_group():
    """If all envs in a group have identical z's, log(VS) = log(1) = 0."""
    z_step = jnp.tile(jnp.array([1.0, 2.0]), (10, 1))
    c_step = jnp.zeros(10, dtype=jnp.int32)          # all code 0
    r = grouped_vendi_per_step(
        z_step=z_step,
        c_step=c_step,
        num_codes=1,
        min_group_size=5,
    )
    # All 10 envs in same group, all identical → VS = 1 → log VS = 0.
    for i in range(10):
        assert float(r[i]) == pytest.approx(0.0, abs=1e-4)


def test_grouped_vendi_distinct_groups_independent():
    """Two distinct code groups should get independent Vendi rewards."""
    rng = np.random.default_rng(0)
    # Group 0: 6 envs, all identical → VS = 1, log VS = 0.
    z_group0 = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (6, 1))
    # Group 1: 6 envs, all distinct (random) → VS > 1, log VS > 0.
    z_group1 = jnp.array(rng.normal(size=(6, 3)).astype(np.float32))
    z_step = jnp.concatenate([z_group0, z_group1], axis=0)
    c_step = jnp.array([0] * 6 + [1] * 6, dtype=jnp.int32)
    r = grouped_vendi_per_step(
        z_step=z_step,
        c_step=c_step,
        num_codes=2,
        min_group_size=5,
    )
    # First 6 envs: identical group → log VS = 0
    for i in range(6):
        assert float(r[i]) == pytest.approx(0.0, abs=1e-3)
    # Last 6 envs: distinct group → log VS > 0
    for i in range(6, 12):
        assert float(r[i]) > 0.0


def test_grouped_vendi_matches_single_group_vendi():
    """Tight equivalence: when every env is in the same group, the
    grouped Vendi must equal log(VS) computed directly on the same z's.

    This is the strongest correctness gate for the grouped implementation
    — it verifies the zero-padded sub-kernel + top-c eigvals strategy
    actually recovers the true per-group Vendi.
    """
    rng = np.random.default_rng(123)
    n, d = 10, 6
    Z = jnp.array(rng.normal(size=(n, d)).astype(np.float32))
    # Single group case: all envs targeted at code 0.
    c_step = jnp.zeros(n, dtype=jnp.int32)

    vs_direct = float(vendi_score_cosine(Z))
    log_vs_expected = float(jnp.log(jnp.maximum(vs_direct, 1.0)))

    r_grouped = grouped_vendi_per_step(
        z_step=Z,
        c_step=c_step,
        num_codes=1,
        min_group_size=2,
    )
    # All n envs are in the same group, so every entry of r_grouped
    # should equal log(VS_direct).
    for i in range(n):
        assert float(r_grouped[i]) == pytest.approx(log_vs_expected, abs=5e-4), (
            f"grouped log VS at env {i} = {float(r_grouped[i]):.6f}, "
            f"expected log VS = {log_vs_expected:.6f}"
        )


def test_grouped_vendi_matches_single_group_vendi_subset():
    """Even tighter: with two groups, the in-group Vendi for each group
    must match an independent computation on just that group's z's."""
    rng = np.random.default_rng(456)
    d = 5
    z_g0 = jnp.array(rng.normal(size=(6, d)).astype(np.float32))
    z_g1 = jnp.array(rng.normal(size=(7, d)).astype(np.float32))
    Z = jnp.concatenate([z_g0, z_g1], axis=0)
    c_step = jnp.array([0] * 6 + [1] * 7, dtype=jnp.int32)

    # Independent per-group VS.
    vs_g0 = float(vendi_score_cosine(z_g0))
    vs_g1 = float(vendi_score_cosine(z_g1))
    log_vs_g0_expected = float(jnp.log(jnp.maximum(vs_g0, 1.0)))
    log_vs_g1_expected = float(jnp.log(jnp.maximum(vs_g1, 1.0)))

    r_grouped = grouped_vendi_per_step(
        z_step=Z, c_step=c_step, num_codes=2, min_group_size=5,
    )
    for i in range(6):
        assert float(r_grouped[i]) == pytest.approx(log_vs_g0_expected, abs=5e-4), (
            f"group 0 env {i}: got {float(r_grouped[i]):.6f}, "
            f"expected {log_vs_g0_expected:.6f}"
        )
    for i in range(6, 13):
        assert float(r_grouped[i]) == pytest.approx(log_vs_g1_expected, abs=5e-4), (
            f"group 1 env {i}: got {float(r_grouped[i]):.6f}, "
            f"expected {log_vs_g1_expected:.6f}"
        )


def test_grouped_vendi_under_jit():
    """grouped_vendi_per_step works under JIT."""
    rng = np.random.default_rng(1)
    z_step = jnp.array(rng.normal(size=(20, 8)).astype(np.float32))
    c_step = jnp.array(rng.integers(0, 3, size=20).astype(np.int32))
    fn = jax.jit(grouped_vendi_per_step, static_argnums=(2, 3))
    r_eager = grouped_vendi_per_step(z_step, c_step, 3, 5)
    r_jit = fn(z_step, c_step, 3, 5)
    np.testing.assert_allclose(np.asarray(r_eager), np.asarray(r_jit), atol=1e-4)


def test_grouped_vendi_per_timestep_shape():
    """vmapped per-timestep version has shape (T, B)."""
    rng = np.random.default_rng(2)
    T, B, d = 4, 12, 6
    z_seq = jnp.array(rng.normal(size=(T, B, d)).astype(np.float32))
    c_seq = jnp.array(rng.integers(0, 3, size=(T, B)).astype(np.int32))
    r = grouped_vendi_per_timestep(z_seq, c_seq, num_codes=3, min_group_size=4)
    assert r.shape == (T, B), f"Expected ({T}, {B}); got {r.shape}"
