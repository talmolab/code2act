"""Product-space VQ-VAE bottleneck adapter for the kinematics-VAE family.

Wraps :class:`c2a.codegen.kinematics_vae.ps_vqvae.quantizer.MultiGroupQuantizer`
to match the bottleneck protocol shared by VAE / VQVAE / PS-VQVAE.

Mutable-state contract
----------------------
This bottleneck declares ``mutable_collections = ("vq_state",)`` —
``c2a.codegen.kinematics_vae.backbone.train_loop`` reads that
attribute, threads ``mutable=["vq_state"]`` into ``apply``, and routes
the updated EMA buffers back into the model. VAE / VQVAE bottlenecks
leave ``mutable_collections`` empty and take the existing
pure-function fast path.

Grouping policy
---------------
Default groups follow Scott v15's body-part split:
``group_dims=(4, 8, 12)`` for (root, axial, limb). Per-group codebook
sizes default to ``group_codebook_sizes=(8, 8, 16)`` — paper-faithful
for the limb subspace (K=16, "subtype") and conservative for root +
axial (K=8 each). The user can override via the train config to match
the paper's C=2 (K1=8, K2=16) by setting ``group_dims=(12, 12)`` and
``group_codebook_sizes=(8, 16)`` (root+axial concatenated as one group,
limb as the other).
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from c2a.codegen.kinematics_vae.ps_vqvae.quantizer import MultiGroupQuantizer


class PSVQVAEBottleneck(nn.Module):
    """Product-space VQ-VAE bottleneck over Scott v15's branched latent."""

    group_dims: Sequence[int]
    group_codebook_sizes: Sequence[int]
    decay: float = 0.99
    reset_threshold: float = 1.0
    reseed_jitter: float = 0.01

    # Declared so the train_loop knows to thread mutable=["vq_state"].
    mutable_collections: tuple[str, ...] = ("vq_state",)

    @nn.compact
    def __call__(
        self,
        mean: jnp.ndarray,
        logvar: jnp.ndarray,
        rng,
        *,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], dict[str, jnp.ndarray]]:
        del logvar, rng

        if int(mean.shape[-1]) != int(sum(self.group_dims)):
            raise ValueError(
                f"PSVQVAEBottleneck: mean.shape[-1]={mean.shape[-1]} != "
                f"sum(group_dims)={sum(self.group_dims)}. The encoder's "
                f"concatenated latent dim must equal the configured "
                f"product-space group total."
            )
        if len(self.group_dims) != len(self.group_codebook_sizes):
            raise ValueError(
                f"len(group_dims)={len(self.group_dims)} != "
                f"len(group_codebook_sizes)={len(self.group_codebook_sizes)}."
            )

        mgq = MultiGroupQuantizer(
            group_dims=tuple(int(d) for d in self.group_dims),
            group_codebook_sizes=tuple(int(K) for K in self.group_codebook_sizes),
            decay=self.decay,
            reset_threshold=self.reset_threshold,
            reseed_jitter=self.reseed_jitter,
            name="mgq",
        )
        # train=True drives EMA buffer updates; eval / codegen use False.
        z_st, codes, info = mgq(mean, train=not deterministic)

        aux_losses = {"commit": info["commit"]}
        aux_metrics = {
            "perplexity": info["perplexity"],
            "codebook_utilization": info["codebook_utilization"],
            "codes": codes,
        }
        # Forward per-group diagnostics for finer logging.
        for k, v in info.items():
            if k.startswith("perplexity_g") or k.startswith("utilization_g"):
                aux_metrics[k] = v
        return z_st, aux_losses, aux_metrics
