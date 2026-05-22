"""Vanilla VQ-VAE bottleneck adapter for the kinematics-VAE family.

Wraps :class:`c2a.codegen.kinematics_vae.vqvae.quantizer.VectorQuantizer`
to match the bottleneck protocol shared by the VAE / VQVAE / PS-VQVAE
variants:

    bottleneck.apply(bn_params, mean, logvar, rng, *, deterministic, **)
        -> (z, aux_losses, aux_metrics)

The encoder emits ``(mean, logvar)`` for interface uniformity. The
VQVAE ignores ``logvar`` entirely (its bottleneck is deterministic) and
uses ``mean`` as the continuous encoder output ``z_e`` that gets
quantized. ``deterministic`` and ``rng`` are accepted for signature
compatibility but unused — quantization is identical at train / eval.

Variant policy: concatenate the per-branch ``μ`` outputs into the full
``latent_dim=24`` vector, then quantize against a SINGLE codebook of
that dim. This is the "vanilla" choice contrasted with the
product-space variant which keeps separate codebooks per branch.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from c2a.codegen.kinematics_vae.vqvae.quantizer import VectorQuantizer


class VQVAEBottleneck(nn.Module):
    """Single-codebook VQ-VAE bottleneck over the concatenated latent."""

    codebook_size: int
    code_dim: int

    @nn.compact
    def __call__(
        self,
        mean: jnp.ndarray,
        logvar: jnp.ndarray,
        rng,
        *,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        # logvar / rng / deterministic exist for interface compatibility.
        # VQ is deterministic everywhere.
        del logvar, rng, deterministic

        if mean.shape[-1] != self.code_dim:
            raise ValueError(
                f"VQVAEBottleneck: mean.shape[-1]={mean.shape[-1]} != "
                f"code_dim={self.code_dim}. The encoder's concatenated "
                f"latent dim must equal the configured codebook code_dim."
            )

        quantizer = VectorQuantizer(
            codebook_size=self.codebook_size,
            code_dim=self.code_dim,
            name="quantizer",
        )
        z_st, aux_losses, aux_metrics = quantizer(mean)
        return z_st, aux_losses, aux_metrics

    @nn.compact
    def codebook_lookup(self, codes: jnp.ndarray) -> jnp.ndarray:
        """``codebook[codes]`` via the inner :class:`VectorQuantizer`.

        Used to inject a specific code into the downstream policy when
        bypassing the live encoder.
        """
        quantizer = VectorQuantizer(
            codebook_size=self.codebook_size,
            code_dim=self.code_dim,
            name="quantizer",
        )
        return quantizer.codebook_lookup(codes)
