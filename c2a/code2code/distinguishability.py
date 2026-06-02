"""Per-step action distinguishability reward — Term 3 of the
skill-diversity objective.

Computes ``D_KL( π(·|s, z) || π̄(·|s) )`` at every (t, b) of the rollout,
where ``π̄(·|s) = E_{z' ~ p(z|z)} [π(·|s, z')]`` is the code-marginalized
policy at state s, approximated by a moment-matched Gaussian over K
alternative codes evaluated at the same state with the same RNN carry.

Alt-code sampling is **distance-weighted**: at each (s, z), the K alt
codes are drawn from a probability distribution proportional to cosine
distance from ``codebook[z]``. This concentrates the limited sample
budget on alt codes that are semantically distant from z (where action
distributions actually differ), instead of wasting it on near-neighbor
codes whose action distributions are similar to π(·|s, z) anyway.

The signal pushes the policy toward action distributions that depend
substantively on the current code. Without this term, codes only need
to shape the long-term state distribution, and the policy is free to
use the same action regardless of code at any given state. See
``scratch/2026-06-01-skill-diversity-objective-design/README.md`` for
the design rationale and information-theoretic interpretation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


def gaussian_kl(mu1, logstd1, mu2, logstd2):
    """KL(N(μ₁, σ₁) || N(μ₂, σ₂)), summed over action dimensions.

    For diagonal Gaussians:

        KL = Σ_i [ log(σ₂_i / σ₁_i) + (σ₁_i² + (μ₁_i − μ₂_i)²) / (2 σ₂_i²) − 0.5 ]
    """
    var1 = jnp.exp(2 * logstd1)
    var2 = jnp.exp(2 * logstd2)
    kl = (logstd2 - logstd1) + (var1 + (mu1 - mu2) ** 2) / (2 * var2) - 0.5
    return jnp.sum(kl, axis=-1)


def moment_match_mixture(means, logstds):
    """Approximate a mixture of K Gaussians by a single Gaussian
    matching first two moments.

    Mean of mixture = mean of means.
    Variance of mixture = mean of variances + var of means (across-K).

    Args:
        means:   (K, ..., action_dim)
        logstds: (K, ..., action_dim)

    Returns:
        (mean, logstd): each (..., action_dim)
    """
    mean = means.mean(axis=0)
    var = (
        jnp.exp(2 * logstds).mean(axis=0)
        + ((means - mean) ** 2).mean(axis=0)
    )
    logstd = 0.5 * jnp.log(var + 1e-8)
    return mean, logstd


# ---------------------------------------------------------------------------
# Distance-weighted alt-code sampling — precomputation
# ---------------------------------------------------------------------------


def precompute_alt_code_sampling_probs(codebook):
    """Precompute the (num_codes, num_codes) row-normalized sampling
    probability table used for distance-weighted alt-code sampling.

    ``sampling_probs[z, z']`` is the probability of drawing z' as an alt
    code when the actual code is z. Probability is proportional to
    cosine distance ``d(codebook[z], codebook[z'])``. The diagonal is
    zeroed (never sample z itself), so each row is a valid probability
    distribution over the OTHER ``num_codes - 1`` codes.

    Done once at module init from the loaded VAE adapter's codebook.

    Args:
        codebook: (num_codes, code_dim) frozen tokenizer codebook.

    Returns:
        sampling_probs: (num_codes, num_codes) — each row sums to 1.0
        with a zero diagonal entry.
    """
    eps = 1e-9
    norms = codebook / (jnp.linalg.norm(codebook, axis=-1, keepdims=True) + eps)
    cos_sim = norms @ norms.T                       # (K, K) in [-1, 1]
    cos_dist = 1.0 - cos_sim                        # (K, K) in [0, 2]
    # Zero the diagonal so we don't sample z itself.
    num_codes = cos_dist.shape[0]
    diag_mask = jnp.eye(num_codes, dtype=cos_dist.dtype)
    cos_dist = cos_dist * (1.0 - diag_mask)
    # Normalize each row to a probability distribution.
    row_sums = cos_dist.sum(axis=-1, keepdims=True) + eps
    return cos_dist / row_sums


# ---------------------------------------------------------------------------
# Top-level reward computation
# ---------------------------------------------------------------------------


def compute_action_distinguishability(
    actual_mean,            # (T, B, action_dim) from main forward pass
    actual_logstd,          # (T, B, action_dim)
    policy_apply_fn,        # (obs, initial_carry, done, rng) -> logits
    split_logits_fn,        # logits -> (mean, logstd)
    obs,                    # rollout obs dict with "kpms_code"
    initial_carry,
    done,
    rng,
    sampling_probs,         # (num_codes, num_codes) precomputed
    num_alt_codes: int = 3,
) -> jnp.ndarray:
    """Compute KL(π(·|s,z) || π̄(·|s)) at each (t, b) by re-running the
    policy with K distance-weighted alt codes substituted for the actual
    target code.

    Strategy:
      1. Look up per-(t, b) sampling distribution from the precomputed
         distance-weighted table (sampling_probs[actual_z]).
      2. Sample K alt codes per (t, b) via the Gumbel-max top-K trick
         (jit-friendly weighted sampling without replacement).
      3. For each alt code, re-run policy forward (re-using the same
         RNN carry and done sequence) to get an alt mean/logstd.
      4. Moment-match the K alt Gaussians to a single Gaussian.
      5. Closed-form Gaussian KL from the actual to the moment-matched
         marginal.

    Args:
        actual_mean: (T, B, action_dim) action-distribution mean from
            the main forward pass (computed with the rollout's actual
            target codes).
        actual_logstd: (T, B, action_dim) action-distribution log-std.
        policy_apply_fn: closure that runs the policy network's
            apply_sequence with whatever obs/carry/done/rng we pass.
            Returns the policy logits tensor (mean + logstd concat).
        split_logits_fn: utility that takes the logits tensor and
            returns (mean, logstd) split along the last axis.
        obs: rollout observation dict; must contain "kpms_code" with
            shape (T, B, code_stack_size) — first entry of the stack is
            the current target code, the one we substitute.
        initial_carry: initial RNN hidden state at start of unroll.
        done: (T, B) done flags for RNN reset.
        rng: PRNG key.
        sampling_probs: (num_codes, num_codes) precomputed
            distance-weighted sampling table.
        num_alt_codes: K — number of alt codes to sample per (t, b).
            Default 3. Each adds one extra forward pass through the
            policy network.

    Returns:
        kl: (T, B) per-step KL divergence, nan-guarded.
    """
    T, B = obs["kpms_code"].shape[:2]
    # Actual target code at each (t, b) — first entry of the code stack.
    actual_z = obs["kpms_code"][..., 0].astype(jnp.int32)        # (T, B)
    # Per-(t, b) sampling distribution from the precomputed table.
    per_step_probs = sampling_probs[actual_z]                    # (T, B, K)

    # Gumbel-max trick for vectorized weighted sampling without
    # replacement: argsort(log(p) + Gumbel(0, 1)) and take the top-K
    # gives K samples drawn proportionally to p without replacement.
    rng, sub = jax.random.split(rng)
    num_codes = sampling_probs.shape[0]
    gumbel_noise = jax.random.gumbel(sub, shape=(T, B, num_codes))
    log_probs = jnp.log(per_step_probs + 1e-9)
    scored = log_probs + gumbel_noise
    alt_codes_tb = jnp.argsort(scored, axis=-1)[..., -num_alt_codes:]   # (T, B, K)
    # Reshape to (K, T, B) so we can vmap over the K axis.
    alt_codes = jnp.transpose(alt_codes_tb, (2, 0, 1))                   # (K, T, B)

    def forward_with_alt(alt_code_seq):
        # alt_code_seq: (T, B) — substitute as the first entry of the
        # code stack; later stack entries (look-ahead codes) are kept
        # unchanged.
        kpms_alt = obs["kpms_code"].at[..., 0].set(
            alt_code_seq.astype(jnp.float32)
        )
        obs_alt = {**obs, "kpms_code": kpms_alt}
        alt_logits = policy_apply_fn(obs_alt, initial_carry, done, rng)
        return split_logits_fn(alt_logits)                       # (mean, logstd)

    alt_means, alt_logstds = jax.vmap(forward_with_alt)(alt_codes)
    # Shape: (K, T, B, action_dim)
    avg_mean, avg_logstd = moment_match_mixture(alt_means, alt_logstds)
    kl = gaussian_kl(actual_mean, actual_logstd, avg_mean, avg_logstd)
    return jnp.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
