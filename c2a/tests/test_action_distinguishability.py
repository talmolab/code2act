"""Tests for c2a.code2code.distinguishability — Term 3 of the
skill-diversity objective.

Tests cover:
- Math primitives (gaussian_kl, moment_match_mixture)
- Sampling table properties (diagonal zero, row-normalized,
  distance-weighted)
- End-to-end: distinguishability is zero for a code-agnostic policy and
  positive for a code-distinguishing policy.
- nan guard
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.code2code.distinguishability import (
    compute_action_distinguishability,
    gaussian_kl,
    moment_match_mixture,
    precompute_alt_code_sampling_probs,
)


# -------------------- math primitives --------------------


def test_gaussian_kl_self_is_zero():
    """KL(N(μ, σ) || N(μ, σ)) ≈ 0 for any μ, σ."""
    rng = np.random.default_rng(0)
    mu = jnp.asarray(rng.normal(size=(8,)).astype(np.float32))
    logstd = jnp.asarray(rng.normal(size=(8,)).astype(np.float32))
    kl = float(gaussian_kl(mu, logstd, mu, logstd))
    assert abs(kl) < 1e-5


def test_gaussian_kl_positive_for_different_means():
    """KL between N(μ₁, σ) and N(μ₂, σ) is > 0 when μ₁ ≠ μ₂."""
    mu1 = jnp.array([0.0, 0.0])
    mu2 = jnp.array([1.0, 1.0])
    logstd = jnp.zeros(2)
    kl = float(gaussian_kl(mu1, logstd, mu2, logstd))
    # KL = Σ_i (μ₁_i - μ₂_i)² / (2 σ²) ; σ = e^0 = 1 → KL = 1.0
    assert kl == pytest.approx(1.0, abs=1e-5)


def test_moment_matched_mixture_of_identical_gaussians():
    """K identical Gaussians moment-match to themselves."""
    mu = jnp.array([[1.0, 2.0]] * 4)        # (K=4, action_dim=2)
    ls = jnp.array([[0.0, 0.0]] * 4)
    out_mu, out_ls = moment_match_mixture(mu, ls)
    assert jnp.allclose(out_mu, jnp.array([1.0, 2.0]))
    # Variance: σ² mean is 1, var(μ) is 0 → mixture var = 1 → log_std ≈ 0
    assert jnp.allclose(out_ls, jnp.array([0.0, 0.0]), atol=1e-4)


def test_moment_matched_mixture_increases_variance():
    """Mixture of Gaussians with different means has VAR ≥ component var."""
    mu = jnp.array([[1.0, 0.0], [-1.0, 0.0]])  # K=2, opposing means
    ls = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    out_mu, out_ls = moment_match_mixture(mu, ls)
    # Mixture mean is 0; variance is 1 (per-component) + var(means) = 1 + 1 = 2
    out_var = float(jnp.exp(2 * out_ls)[0])
    assert out_var == pytest.approx(2.0, abs=1e-4)


# -------------------- sampling table --------------------


def _make_codebook(num_codes=8, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(num_codes, dim)).astype(np.float32))


def test_sampling_probs_diagonal_zero():
    """sampling_probs[z, z] = 0 for all z (never sample yourself)."""
    codebook = _make_codebook()
    probs = precompute_alt_code_sampling_probs(codebook)
    diag = jnp.diagonal(probs)
    assert jnp.allclose(diag, jnp.zeros_like(diag), atol=1e-6)


def test_sampling_probs_row_normalized():
    """Each row of sampling_probs sums to 1.0."""
    codebook = _make_codebook()
    probs = precompute_alt_code_sampling_probs(codebook)
    row_sums = probs.sum(axis=-1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-5)


def test_sampling_probs_distance_monotone():
    """For each anchor z, sampling probability of z' is monotone
    increasing in cosine distance d(codebook[z], codebook[z'])."""
    codebook = _make_codebook(num_codes=10, dim=8, seed=42)
    probs = precompute_alt_code_sampling_probs(codebook)
    # Compute actual cosine distances.
    norms = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-9)
    cos_sim = norms @ norms.T
    cos_dist = 1.0 - cos_sim

    # For anchor z=0, check that probs are positively correlated with distances.
    z = 0
    probs_row = np.asarray(probs[z])
    dist_row = np.asarray(cos_dist[z])
    # Exclude the diagonal (probs[z, z] = 0).
    mask = np.ones(len(probs_row), dtype=bool)
    mask[z] = False
    # Pearson correlation between probs and distances should be > 0 (proportional)
    corr = np.corrcoef(probs_row[mask], dist_row[mask])[0, 1]
    assert corr > 0.99, (
        f"sampling probs should be linearly proportional to distance; "
        f"correlation = {corr:.4f}"
    )


# -------------------- end-to-end with synthetic policy --------------------


def _make_synth_policy_apply(action_dim, code_to_mean_shift=False):
    """Build a synthetic policy_apply_fn that returns Gaussian logits
    (mean, logstd) as a concatenation along the last axis.

    If ``code_to_mean_shift=False``, output is constant regardless of code.
    If ``code_to_mean_shift=True``, output mean is shifted by the
    rollout's first code (proxy for "code influences action").
    """
    def policy_apply(obs, carry, done, rng):
        # Build fake (T, B, 2*action_dim) logits.
        kpms_code = obs["kpms_code"]
        T, B = kpms_code.shape[:2]
        actual_z = kpms_code[..., 0]  # (T, B)
        # Mean = 0 (constant) or = z (code-dependent).
        if code_to_mean_shift:
            shift = actual_z[..., None] * 0.5     # (T, B, 1)
        else:
            shift = jnp.zeros((T, B, 1))
        mean = jnp.broadcast_to(shift, (T, B, action_dim))
        logstd = jnp.zeros((T, B, action_dim))
        return jnp.concatenate([mean, logstd], axis=-1)

    def split_logits(logits):
        d = logits.shape[-1] // 2
        return logits[..., :d], logits[..., d:]

    return policy_apply, split_logits


def test_distinguishability_zero_when_policy_code_agnostic():
    """Synthetic policy that ignores codes → distinguishability ≈ 0."""
    num_codes, code_dim, action_dim = 8, 4, 3
    T, B = 4, 16
    codebook = _make_codebook(num_codes, code_dim)
    sampling_probs = precompute_alt_code_sampling_probs(codebook)
    policy_apply, split_logits = _make_synth_policy_apply(
        action_dim, code_to_mean_shift=False,
    )

    rng = jax.random.PRNGKey(0)
    rng_codes, rng_run = jax.random.split(rng)
    obs = {
        "kpms_code": jax.random.randint(
            rng_codes, shape=(T, B, 5), minval=0, maxval=num_codes,
        ).astype(jnp.float32),
    }
    actual_logits = policy_apply(obs, None, None, rng_run)
    actual_mean, actual_logstd = split_logits(actual_logits)

    kl = compute_action_distinguishability(
        actual_mean=actual_mean,
        actual_logstd=actual_logstd,
        policy_apply_fn=policy_apply,
        split_logits_fn=split_logits,
        obs=obs,
        initial_carry=None,
        done=None,
        rng=rng_run,
        sampling_probs=sampling_probs,
        num_alt_codes=3,
    )
    assert float(jnp.max(kl)) < 1e-3, (
        f"code-agnostic policy should give ≈ 0 KL; got max = {float(jnp.max(kl)):.6f}"
    )


def test_distinguishability_positive_when_policy_distinguishes():
    """Synthetic policy whose mean shifts with code → KL > 0."""
    num_codes, code_dim, action_dim = 8, 4, 3
    T, B = 4, 16
    codebook = _make_codebook(num_codes, code_dim)
    sampling_probs = precompute_alt_code_sampling_probs(codebook)
    policy_apply, split_logits = _make_synth_policy_apply(
        action_dim, code_to_mean_shift=True,
    )

    rng = jax.random.PRNGKey(0)
    rng_codes, rng_run = jax.random.split(rng)
    obs = {
        "kpms_code": jax.random.randint(
            rng_codes, shape=(T, B, 5), minval=0, maxval=num_codes,
        ).astype(jnp.float32),
    }
    actual_logits = policy_apply(obs, None, None, rng_run)
    actual_mean, actual_logstd = split_logits(actual_logits)

    kl = compute_action_distinguishability(
        actual_mean=actual_mean,
        actual_logstd=actual_logstd,
        policy_apply_fn=policy_apply,
        split_logits_fn=split_logits,
        obs=obs,
        initial_carry=None,
        done=None,
        rng=rng_run,
        sampling_probs=sampling_probs,
        num_alt_codes=3,
    )
    assert float(jnp.mean(kl)) > 0.1, (
        f"code-distinguishing policy should give substantial KL; "
        f"got mean = {float(jnp.mean(kl)):.6f}"
    )


def test_nan_in_means_does_not_poison_kl():
    """NaN in policy logits propagates through gaussian_kl but is
    nan_to_num'd before being returned."""
    num_codes, action_dim = 8, 3
    T, B = 2, 4
    codebook = _make_codebook(num_codes, 4)
    sampling_probs = precompute_alt_code_sampling_probs(codebook)
    policy_apply, split_logits = _make_synth_policy_apply(
        action_dim, code_to_mean_shift=False,
    )

    rng = jax.random.PRNGKey(0)
    rng_codes, rng_run = jax.random.split(rng)
    obs = {
        "kpms_code": jax.random.randint(
            rng_codes, shape=(T, B, 5), minval=0, maxval=num_codes,
        ).astype(jnp.float32),
    }
    actual_logits = policy_apply(obs, None, None, rng_run)
    actual_mean, actual_logstd = split_logits(actual_logits)
    # Inject NaN at (t=0, b=0, dim=0).
    actual_mean = actual_mean.at[0, 0, 0].set(jnp.nan)

    kl = compute_action_distinguishability(
        actual_mean=actual_mean,
        actual_logstd=actual_logstd,
        policy_apply_fn=policy_apply,
        split_logits_fn=split_logits,
        obs=obs,
        initial_carry=None,
        done=None,
        rng=rng_run,
        sampling_probs=sampling_probs,
        num_alt_codes=3,
    )
    assert jnp.all(jnp.isfinite(kl)), (
        "nan_to_num guard should prevent NaN propagation"
    )
