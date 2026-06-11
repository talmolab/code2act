"""Categorical PPO networks with a policy-gated additive code prior.

The high-level policy (Option C) reads a flat observation laid out as
``[proprio(277) | task_obs(52) | prior_logits(num_codes)]``. It normalizes the
``proprio+task_obs`` features (via brax's running-statistics
``preprocess_observations_fn``) but reads ``prior_logits`` **raw** (the last
``num_codes`` dims), then forms

    final_logits = sigmoid(beta) * prior_logits + delta

where ``(delta, beta)`` are the trunk's two heads. ``final_logits`` are the
parameters of a :class:`CategoricalDistribution` over the codebook.

The value network is brax's stock ``make_value_network`` over the full flat obs
(normalizing ``prior_logits`` there is harmless — it is just a value feature).
"""

from __future__ import annotations

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.training import networks as brax_networks
from brax.training import types
from brax.training.agents.ppo.networks import PPONetworks
from brax.training.agents.ppo.networks import make_inference_fn  # noqa: F401 (re-export)

from c2a.transfer.distribution import CategoricalDistribution


class _PolicyModule(nn.Module):
    """Trunk MLP -> (delta over codes, scalar gate beta) -> gated additive logits."""

    num_codes: int
    hidden: Sequence[int]
    activation: Callable = nn.swish

    @nn.compact
    def __call__(self, feats: jnp.ndarray, prior_logits: jnp.ndarray) -> jnp.ndarray:
        ki = jax.nn.initializers.lecun_uniform()
        x = feats
        for h in self.hidden:
            x = self.activation(nn.Dense(h, kernel_init=ki)(x))
        delta = nn.Dense(self.num_codes, kernel_init=ki, name="delta")(x)
        beta = nn.Dense(1, kernel_init=ki, name="beta")(x)
        return jax.nn.sigmoid(beta) * prior_logits + delta


def _make_policy_network(
    num_codes: int,
    obs_size: int,
    preprocess_observations_fn: Callable,
    hidden: Sequence[int],
) -> brax_networks.FeedForwardNetwork:
    feats_dim = obs_size - num_codes
    module = _PolicyModule(num_codes=num_codes, hidden=tuple(hidden))

    def apply(processor_params, policy_params, obs):
        prior_raw = obs[..., -num_codes:]  # RAW (unnormalized) prior logits
        obs_norm = preprocess_observations_fn(obs, processor_params)
        feats = obs_norm[..., :feats_dim]  # normalized proprio+task_obs
        return module.apply(policy_params, feats, prior_raw)

    dummy_feats = jnp.zeros((1, feats_dim))
    dummy_prior = jnp.zeros((1, num_codes))
    return brax_networks.FeedForwardNetwork(
        init=lambda key: module.init(key, dummy_feats, dummy_prior), apply=apply
    )


def make_categorical_ppo_networks(
    observation_size,
    action_size: int,
    preprocess_observations_fn: Callable = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256, 128),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 128),
) -> PPONetworks:
    """Build categorical PPO networks. ``action_size`` == ``num_codes``.

    brax calls this positionally as
    ``network_factory(obs_shape, action_size, preprocess_observations_fn=...)``.
    """
    num_codes = int(action_size)
    obs_size = (
        int(observation_size[-1])
        if hasattr(observation_size, "__len__")
        else int(observation_size)
    )
    policy = _make_policy_network(
        num_codes, obs_size, preprocess_observations_fn, policy_hidden_layer_sizes
    )
    value = brax_networks.make_value_network(
        obs_size, preprocess_observations_fn, value_hidden_layer_sizes
    )
    return PPONetworks(
        policy_network=policy,
        value_network=value,
        parametric_action_distribution=CategoricalDistribution(num_codes),
    )
