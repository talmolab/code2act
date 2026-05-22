"""Vanilla VQ-VAE quantizer: gradient codebook, STE, commitment loss.

No EMA codebook update, no dead-code revival, no RVQ, no rotation tricks
— the simplest possible quantizer following van den Oord et al. 2017
"Neural Discrete Representation Learning" (Eq. 3-4) exactly. The
codebook is a single learned ``nn.Param`` updated via gradient descent
through the codebook loss term.

The choice to skip EMA is deliberate: matches the
``/home/jovyan/vast/kaiwen/track-mjx/vqvae_jax`` reference implementation
on the ``kevin/vqvae`` branch, avoids JAX mutable-state plumbing for
the EMA buffers, and keeps the variant strictly the "vanilla" baseline
the user asked for. The product-space variant in :mod:`c2a.codegen.
kinematics_vae.ps_vqvae` adds EMA + dead-code reseeding when needed for
paper-faithful behavior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn


class VectorQuantizer(nn.Module):
    """K-entry codebook over a ``code_dim``-D continuous latent.

    Args (Module fields):
        codebook_size: ``K`` — number of discrete codes.
        code_dim: ``D`` — dimensionality of each codebook entry.
        commit_weight: Coefficient ``β`` for the commitment term in the
            van den Oord 2017 formulation. The training loss should
            include ``β * commit`` (this module returns the raw values;
            the train loop applies weights through ``loss_weights``).
    """

    codebook_size: int
    code_dim: int

    @nn.compact
    def __call__(
        self,
        z_e: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        """Quantize ``z_e`` against the codebook.

        Args:
            z_e: ``(B, D)`` encoder output (continuous).

        Returns:
            ``z_st``: ``(B, D)`` straight-through quantized output. In the
                forward pass, ``z_st`` numerically equals ``z_q`` (the
                nearest codebook entry). In the backward pass, gradients
                flow through ``z_e`` unchanged (identity-STE).
            aux_losses: ``{"commit": <scalar>, "codebook": <scalar>}``
                — commitment term ``||z_e - sg(z_q)||²`` and codebook
                term ``||sg(z_e) - z_q||²``.
            aux_metrics: ``{"codes": (B,) int32, "perplexity": <scalar>,
                "codebook_utilization": <scalar>}``.
        """
        # Codebook: (K, D) learned param. lecun_uniform init matches the
        # Mlp default and gives well-scaled initial codes.
        codebook = self._codebook_param()

        # Pairwise squared distances: ||z_e[i] - codebook[k]||²
        #   = ||z_e[i]||² + ||codebook[k]||² - 2 z_e[i] · codebook[k]
        z_e_sq = jnp.sum(z_e**2, axis=-1, keepdims=True)        # (B, 1)
        cb_sq = jnp.sum(codebook**2, axis=-1)                   # (K,)
        dot = z_e @ codebook.T                                  # (B, K)
        dists = z_e_sq + cb_sq[None, :] - 2.0 * dot              # (B, K)

        codes = jnp.argmin(dists, axis=-1)                       # (B,)
        z_q = codebook[codes]                                    # (B, D)

        # Straight-through estimator: forward = z_q, backward = identity to z_e.
        # The classic VQ-VAE pattern from van den Oord 2017 (Eq. 2):
        #   z_st = z_e + sg(z_q − z_e)
        # so d z_st / d z_e = I and d z_st / d codebook = 0 (gradient flows
        # through the codebook only via the codebook loss term below).
        z_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        # Commitment (pulls encoder toward codebook): grad flows to z_e.
        commit = jnp.mean(jnp.sum((z_e - jax.lax.stop_gradient(z_q)) ** 2, axis=-1))
        # Codebook (pulls codebook toward encoder): grad flows to codebook.
        codebook_loss = jnp.mean(
            jnp.sum((jax.lax.stop_gradient(z_e) - z_q) ** 2, axis=-1)
        )

        # Diagnostics: per-batch codebook utilization + perplexity.
        # one_hot is on the forward path so we keep it inside the trace.
        one_hot = jax.nn.one_hot(codes, self.codebook_size)  # (B, K)
        avg_probs = jnp.mean(one_hot, axis=0)                 # (K,)
        # Perplexity = exp(entropy); maxed when usage is uniform across K.
        perplexity = jnp.exp(
            -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10))
        )
        utilization = jnp.mean((avg_probs > 0).astype(jnp.float32))

        aux_losses = {
            "commit": commit,
            "codebook": codebook_loss,
        }
        aux_metrics = {
            "perplexity": perplexity,
            "codebook_utilization": utilization,
            # codes are scattered along the leading axis; codegen.encode
            # detects per-window dim and pads to per-frame.
            "codes": codes,
        }
        return z_st, aux_losses, aux_metrics

    def _codebook_param(self):
        # Shared param accessor — same name + shape as the one created in
        # ``__call__`` so both methods share the saved param under
        # ``quantizer/codebook``.
        return self.param(
            "codebook",
            nn.initializers.lecun_uniform(),
            (self.codebook_size, self.code_dim),
        )

    @nn.compact
    def codebook_lookup(self, codes: jnp.ndarray) -> jnp.ndarray:
        """Plain codebook lookup ``codebook[codes]`` (no encoder, no quantizer).

        Used by callers that want to inject a specific code into the
        downstream policy bypassing the encoder + nearest-neighbour search.
        """
        codebook = self._codebook_param()
        return codebook[codes]
