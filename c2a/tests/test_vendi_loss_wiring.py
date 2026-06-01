"""Tests for the Vendi diversity-bonus wiring in the PPO loss.

The full loss path (compute_moseq_recurrent_ppo_loss) requires a full
ppo_network and policy params, which is heavy to mock.  Instead these
tests pin down the *augmentation step* in isolation: given synthetic
state_extras, the loss's new block computes ``r + β · log(VS)`` exactly
as expected, with the right shapes and the right zero-out behavior when
disabled.

Companion test ``test_finetune_schedule.py`` already covers β(τ) values
and ``test_diversity_vendi.py`` covers grouped_vendi_per_timestep math.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from c2a.code2code.diversity import grouped_vendi_per_timestep


# ---------- helpers replicating the augmentation block in the loss ----------


def _augment_rewards(
    rewards: jnp.ndarray,
    z_seq: jnp.ndarray,
    c_seq: jnp.ndarray,
    beta_seq: jnp.ndarray,
    num_codes: int,
    min_group_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mirror of the vendi_enabled block in compute_moseq_recurrent_ppo_loss.

    Returns ``(augmented_rewards, log_vs_seq)`` for assertion convenience.
    Keep in sync with c2a/moseq_losses.py.
    """
    log_vs_seq = grouped_vendi_per_timestep(
        z_seq, c_seq, num_codes, min_group_size
    )
    log_vs_seq = jnp.nan_to_num(log_vs_seq, nan=0.0, posinf=0.0, neginf=0.0)
    diversity_reward = beta_seq * log_vs_seq
    return rewards + diversity_reward, log_vs_seq


# ---------------------------- tests ---------------------------------


def test_zero_beta_leaves_rewards_unchanged():
    """β=0 must give back the exact baseline rewards (no spurious adds)."""
    T, B, D, K = 4, 8, 6, 5
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    rewards = jax.random.normal(k1, (T, B))
    z_seq = jax.random.normal(k2, (T, B, D))
    c_seq = jax.random.randint(k3, (T, B), 0, K)
    beta_seq = jnp.zeros((T, B))

    augmented, log_vs = _augment_rewards(
        rewards, z_seq, c_seq, beta_seq, K, min_group_size=2,
    )

    assert jnp.allclose(augmented, rewards), (
        "β=0 must give back baseline rewards exactly"
    )


def test_positive_beta_adds_log_vs():
    """β=1 must add log(VS) to rewards — exact arithmetic match."""
    T, B, D, K = 3, 12, 4, 4
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    rewards = jax.random.normal(k1, (T, B))
    z_seq = jax.random.normal(k2, (T, B, D))
    # Force enough envs per code so groups are non-degenerate.
    c_seq = (jnp.arange(B)[None, :] % K).astype(jnp.int32)
    c_seq = jnp.broadcast_to(c_seq, (T, B))
    beta_seq = jnp.ones((T, B))

    augmented, log_vs = _augment_rewards(
        rewards, z_seq, c_seq, beta_seq, K, min_group_size=2,
    )

    assert jnp.allclose(augmented, rewards + log_vs), (
        "β=1 must add log_vs to rewards exactly"
    )
    # At least some envs should be in valid-size groups → some nonzero log_vs.
    assert jnp.any(log_vs > 0), (
        "expected at least one env to receive a nonzero diversity bonus"
    )


def test_per_env_beta_scaling():
    """Mixed β per env: bonus must scale linearly with β at each (t, b)."""
    T, B, D, K = 2, 10, 3, 5
    key = jax.random.PRNGKey(2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    rewards = jnp.zeros((T, B))
    z_seq = jax.random.normal(k2, (T, B, D))
    c_seq = (jnp.arange(B)[None, :] % K).astype(jnp.int32)
    c_seq = jnp.broadcast_to(c_seq, (T, B))

    beta_unit = jnp.ones((T, B))
    aug_unit, log_vs_unit = _augment_rewards(
        rewards, z_seq, c_seq, beta_unit, K, min_group_size=2,
    )

    beta_2x = 2.0 * jnp.ones((T, B))
    aug_2x, log_vs_2x = _augment_rewards(
        rewards, z_seq, c_seq, beta_2x, K, min_group_size=2,
    )

    assert jnp.allclose(log_vs_unit, log_vs_2x), (
        "log_vs depends only on z/c, not β; varying β must not change it"
    )
    assert jnp.allclose(aug_2x, 2.0 * aug_unit), (
        "doubling β must double the diversity bonus"
    )


def test_nan_in_z_does_not_poison_rewards():
    """A NaN in z must not propagate through to augmented rewards."""
    T, B, D, K = 2, 8, 4, 4
    key = jax.random.PRNGKey(3)
    rewards = jnp.ones((T, B))
    z_seq = jax.random.normal(key, (T, B, D))
    # Inject a NaN at (t=0, b=0)
    z_seq = z_seq.at[0, 0, 0].set(jnp.nan)
    c_seq = (jnp.arange(B)[None, :] % K).astype(jnp.int32)
    c_seq = jnp.broadcast_to(c_seq, (T, B))
    beta_seq = jnp.ones((T, B))

    augmented, _ = _augment_rewards(
        rewards, z_seq, c_seq, beta_seq, K, min_group_size=2,
    )

    assert jnp.all(jnp.isfinite(augmented)), (
        "nan_to_num guard must prevent NaN propagation into augmented rewards"
    )


def test_shape_preservation():
    """Augmented rewards shape must match baseline rewards shape exactly."""
    for T, B in [(1, 4), (5, 16), (40, 256)]:
        D, K = 8, 6
        key = jax.random.PRNGKey(T * 100 + B)
        k1, k2, k3 = jax.random.split(key, 3)
        rewards = jax.random.normal(k1, (T, B))
        z_seq = jax.random.normal(k2, (T, B, D))
        c_seq = jax.random.randint(k3, (T, B), 0, K)
        beta_seq = jnp.ones((T, B)) * 0.1

        augmented, log_vs = _augment_rewards(
            rewards, z_seq, c_seq, beta_seq, K, min_group_size=2,
        )

        assert augmented.shape == (T, B), (
            f"augmented rewards must be (T, B); got {augmented.shape}"
        )
        assert log_vs.shape == (T, B), (
            f"log_vs must be (T, B); got {log_vs.shape}"
        )


def test_env_wrapper_writes_beta_to_state_info():
    """The env wrapper must seed c2c_curriculum_beta in state.info at reset.

    Run is a smoke check — we don't need a full physics step, just
    verify the reset path declares the key so the unroll's pytree shape
    is consistent for jax.lax.scan.
    """
    # The env requires a heavy clips/config setup; we only inspect the
    # static contract that reset() writes c2c_curriculum_beta.  Read
    # the source to confirm rather than instantiate.
    import inspect

    from c2a import moseq_env_wrapper

    src = inspect.getsource(moseq_env_wrapper.MoSeqImitation)
    assert '"c2c_curriculum_beta"' in src, (
        'env wrapper must write "c2c_curriculum_beta" to state.info '
        '(seeded in reset, updated every step) so the PPO loss can read '
        'it via state_extras for the Vendi diversity bonus'
    )
    assert '"c2c_curriculum_alpha"' in src, (
        'env wrapper must also seed "c2c_curriculum_alpha" in state.info'
    )


def test_unroll_extracts_curriculum_beta():
    """The carry-aware unroll must extend state_extras with c2c_curriculum_beta
    so it reaches the loss via data.extras["state_extras"]."""
    import inspect

    from c2a import moseq_ppo

    src = inspect.getsource(moseq_ppo.generate_unroll_rnn_moseq)
    assert '"c2c_curriculum_beta"' in src, (
        'generate_unroll_rnn_moseq must extend state_extras to include '
        '"c2c_curriculum_beta" — otherwise the loss cannot read β(τ) '
        'and the diversity bonus silently becomes 0'
    )
    assert '"c2c_z_rollout"' in src, (
        'generate_unroll_rnn_moseq must include "c2c_z_rollout" in extras'
    )
    assert '"c2c_target_code"' in src, (
        'generate_unroll_rnn_moseq must include "c2c_target_code" in extras'
    )
