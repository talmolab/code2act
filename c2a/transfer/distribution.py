"""Categorical action distribution for brax PPO (discrete code actions).

brax ships only Normal / NormalTanh distributions. The Option-B transfer policy
acts in a **discrete codebook** (action = code index in ``[0, num_codes)``), so we
add a ``CategoricalDistribution`` satisfying brax's ``ParametricDistribution``
contract: ``param_size = num_codes`` logits, identity postprocessor (no tanh),
``event_ndims = 0`` (scalar discrete action — do NOT sum over an action axis).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from brax.training.distribution import IdentityPostprocessor, ParametricDistribution


class _CategoricalDist:
    """Inner categorical over the last logits axis."""

    def __init__(self, logits: jnp.ndarray):
        self.logits = logits

    def sample(self, seed) -> jnp.ndarray:
        return jax.random.categorical(seed, self.logits, axis=-1)

    def mode(self) -> jnp.ndarray:
        return jnp.argmax(self.logits, axis=-1)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        log_p = jax.nn.log_softmax(self.logits, axis=-1)
        return jnp.take_along_axis(
            log_p, x[..., None].astype(jnp.int32), axis=-1
        )[..., 0]

    def entropy(self) -> jnp.ndarray:
        p = jax.nn.softmax(self.logits, axis=-1)
        log_p = jax.nn.log_softmax(self.logits, axis=-1)
        return -jnp.sum(p * log_p, axis=-1)


class CategoricalDistribution(ParametricDistribution):
    """Discrete categorical parametric distribution over ``num_actions`` codes."""

    def __init__(self, num_actions: int):
        super().__init__(
            param_size=num_actions,
            postprocessor=IdentityPostprocessor(),
            event_ndims=0,
            reparametrizable=False,
        )

    def create_dist(self, parameters: jnp.ndarray) -> _CategoricalDist:
        return _CategoricalDist(parameters)
