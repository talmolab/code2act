"""Gaussian-VAE bottleneck — per-branch reparameterization.

This bottleneck is the standard ELBO-VAE choice: given the encoder's
``(mean, logvar)``, sample ``z = mean + exp(0.5*logvar) * eps``, then
emit the KL-to-unit-Gaussian and σ-cap penalty as aux losses. No
learnable parameters of its own — included as a Flax module purely so
the bottleneck-agnostic train loop's
``bottleneck.apply(bn_params, mean, logvar, rng, ...)`` signature
matches across all three variants.

Discrete codes are produced post-hoc by :mod:`c2a.codegen.
kinematics_vae.vae.tokenize` via K-Means (optionally PCA-reduced) on
``z = mean`` over the full balanced-split clip set.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from c2a.codegen.kinematics_vae.backbone.losses import (
    kl_to_unit_gaussian,
    reparameterize,
    sigma_cap_penalty,
)


class GaussianBottleneck(nn.Module):
    """Identity-shaped Gaussian bottleneck.

    Returns ``z`` of the same shape as ``mean`` plus the standard ELBO
    aux losses. Honors ``deterministic=True`` (``z = mean``) for codegen
    + eval. ``sigma_max`` is the runtime curriculum hyperparam — passed
    via ``bottleneck_kwargs`` from the train loop. When ``sigma_max=0``
    the σ-cap penalty is identically zero.
    """

    @nn.compact
    def __call__(
        self,
        mean: jnp.ndarray,
        logvar: jnp.ndarray,
        rng,
        *,
        deterministic: bool = False,
        sigma_max: float | jnp.ndarray = 0.0,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        if deterministic:
            z = mean
        else:
            z = reparameterize(rng, mean, logvar)
        kl = kl_to_unit_gaussian(mean, logvar)
        sigma_pen = sigma_cap_penalty(logvar, sigma_max)
        aux_losses = {"kl": kl, "sigma_pen": sigma_pen}
        aux_metrics: dict[str, jnp.ndarray] = {}
        return z, aux_losses, aux_metrics
