"""Product-space VQ-VAE quantizer — JAX/Flax port of VQ-MAP.

Paper: Li, Klibaite, Akoad, Wu, Dunn 2025 — "VQ-MAP: a flexible,
versatile framework for animal behavior understanding"
(bioRxiv 10.64898/2025.12.04.692429).

This module ports the QUANTIZER side of the paper, not the encoder /
decoder / training script: we plug into Scott v15's conv1d backbone,
not the paper's channel-invariant ISA encoder. See
``scratch/2026-05-12-kinematics-vae-codegen-unify/03b-ps-vqvae-paper-aligned.md``
for the complete paper-alignment table; only deviations from that doc
are allowed in this code.

Two classes:

  :class:`QuantizeEMAReset` — single-codebook EMA-VQ with dead-code
    reseeding via tile-perturb. JAX port of ``csbev/model/quantizer.py::
    QuantizeEMAReset``. EMA buffers (``code_sum``, ``code_count``,
    ``codebook``) live in a Flax "vq_state" mutable collection, threaded
    via ``mutable=["vq_state"]`` in the training step's ``apply`` call.

  :class:`MultiGroupQuantizer` — product-space wrapper that splits
    its input along the channel axis into G groups and runs a
    :class:`QuantizeEMAReset` independently on each. Returns
    concatenated quantized output + per-group commit losses summed +
    per-group codes stacked along the last axis. JAX port of
    ``csbev/model/quantizer.py::MultiGroupQuantizer``.

The paper's headline configuration is C=2 groups, (K1, K2)=(8, 16).
For the c2a port we natively map this onto Scott v15's existing
body-part split (root=4, axial=8, limb=12) as C=3 groups — a clean,
paper-faithful realization of "product space" that respects the
backbone's structural priors.

Tokenization scheme (paper convention): per-frame codes are a tuple
(k^{(1)}, ..., k^{(C)}). Folding to a single integer for downstream
``nn.Embed`` happens at tokenize-time, not here.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def _l2_dist(z: jnp.ndarray, codebook: jnp.ndarray) -> jnp.ndarray:
    """Pairwise squared L2 distances ``||z[i] - codebook[k]||^2``.

    Args:
        z: ``(N, D)`` continuous latents.
        codebook: ``(K, D)`` codebook.

    Returns:
        ``(N, K)`` matrix of squared distances.
    """
    z_sq = jnp.sum(z ** 2, axis=-1, keepdims=True)        # (N, 1)
    cb_sq = jnp.sum(codebook ** 2, axis=-1)               # (K,)
    dot = z @ codebook.T                                   # (N, K)
    return z_sq + cb_sq[None, :] - 2.0 * dot


def _tile(x: jnp.ndarray, K: int, jitter: float = 0.01) -> jnp.ndarray:
    """Tile ``x`` along axis 0 until at least ``K`` rows, then optionally
    add small noise. Used to reseed dead codebook entries from the
    current encoder distribution.

    Mirrors ``csbev/model/quantizer.py::QuantizeEMAReset._tile``: when
    the encoder output batch ``x`` has fewer than ``K`` rows we tile it
    cyclically so every dead code slot can be reseeded.

    The jitter shouldn't reuse the global PRNG — the caller threads a
    fresh key in. Here we add deterministic noise from a hash of x's
    first entry; the caller's reseed path uses
    ``jax.random.normal(reseed_rng, ...)`` instead. Kept here for the
    eager-init path where rngs aren't yet plumbed.
    """
    n = x.shape[0]
    if n >= K:
        return x[:K]
    n_reps = (K + n - 1) // n
    tiled = jnp.tile(x, (n_reps, 1))[:K]
    if jitter > 0:
        # Tiny deterministic perturbation so duplicates aren't byte-identical.
        scale = jitter * jnp.std(x)
        seed_val = jnp.sum(x[0])
        eps = jax.random.normal(
            jax.random.PRNGKey(jnp.asarray(seed_val * 1e6, dtype=jnp.uint32)),
            shape=tiled.shape,
        )
        tiled = tiled + scale * eps
    return tiled


class QuantizeEMAReset(nn.Module):
    """Single-codebook EMA-VQ with dead-code reseeding.

    The codebook, EMA running sums, and EMA running counts live in a
    mutable Flax variable collection named ``"vq_state"``. The training
    step must call ``apply(... , mutable=["vq_state"])`` to allow these
    to update. At eval / codegen time, callers pass ``train=False`` and
    the buffers stay frozen.

    Attributes:
        codebook_size: ``K`` — number of discrete codes.
        code_dim: ``D`` — codebook entry dimensionality.
        decay: EMA decay μ (default 0.99 per ``quantizer_mg.yaml:15``).
        reset_threshold: dead-code reseed trigger (default 1.0 per
            ``quantizer.py:219``: ``usage = (code_count >= 1.0)``).
        reseed_jitter: standard-deviation scale for the noise added to
            tiled encoder samples when reseeding dead slots.
    """

    codebook_size: int
    code_dim: int
    decay: float = 0.99
    reset_threshold: float = 1.0
    reseed_jitter: float = 0.01

    def setup(self):
        # Codebook: (K, D). Initialized from a Gaussian; will be
        # re-seeded from the first batch on the first forward pass.
        self.codebook = self.variable(
            "vq_state", "codebook",
            lambda: jax.random.normal(
                jax.random.PRNGKey(0), (self.codebook_size, self.code_dim),
            ) * 0.1,
        )
        # EMA running sum of encoder outputs assigned to each code:
        # m_j in the paper (p.24 l.748–749).
        self.code_sum = self.variable(
            "vq_state", "code_sum",
            lambda: self.codebook.value.copy(),
        )
        # EMA running count of assignments per code: N_j in the paper.
        # Initialize to 1.0 so the first forward division is safe.
        self.code_count = self.variable(
            "vq_state", "code_count",
            lambda: jnp.ones((self.codebook_size,), dtype=jnp.float32),
        )

    def __call__(
        self,
        z_e: jnp.ndarray,
        *,
        train: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
        """Quantize ``z_e``; if ``train=True``, update EMA buffers.

        Args:
            z_e: ``(N, D)`` continuous latent (already flattened across
                batch + time).
            train: When True, update ``code_sum`` / ``code_count`` /
                ``codebook`` via EMA and reseed dead codes from the
                current batch. Caller must use
                ``apply(..., mutable=["vq_state"])`` to commit.

        Returns:
            ``(z_st, codes, info)`` where:
                ``z_st`` is the STE-quantized output (forward = z_q,
                    backward = identity to z_e).
                ``codes`` is ``(N,)`` int32 codebook indices.
                ``info`` is a dict with ``{"commit": <scalar>,
                    "perplexity": <scalar>, "codebook_utilization":
                    <scalar>}``.
        """
        codebook = self.codebook.value
        dists = _l2_dist(z_e, codebook)
        codes = jnp.argmin(dists, axis=-1).astype(jnp.int32)
        z_q = codebook[codes]
        # STE: forward = z_q, backward = identity to z_e.
        z_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        # Commitment loss only — EMA replaces the explicit L_embedding
        # codebook loss per paper p.24 l.748-749.
        commit = jnp.mean(jnp.sum((z_e - jax.lax.stop_gradient(z_q)) ** 2, axis=-1))

        # Diagnostics: perplexity, utilization.
        one_hot = jax.nn.one_hot(codes, self.codebook_size)  # (N, K)
        avg_probs = jnp.mean(one_hot, axis=0)                 # (K,)
        perplexity = jnp.exp(
            -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10))
        )
        utilization = jnp.mean((avg_probs > 0).astype(jnp.float32))

        if train:
            # EMA buffer updates. one_hot.T @ z_e gives, for each code,
            # the sum of encoder outputs assigned to it this batch.
            #   batch_sum_j   = sum_{i: code_i = j} z_e[i]
            #   batch_count_j = #{i: code_i = j}
            batch_sum = one_hot.T @ z_e                     # (K, D)
            batch_count = jnp.sum(one_hot, axis=0)          # (K,)

            new_sum = self.decay * self.code_sum.value + (1.0 - self.decay) * batch_sum
            new_count = self.decay * self.code_count.value + (1.0 - self.decay) * batch_count
            # Codebook centroid: m_j / N_j (paper p.24 l.748-749).
            new_codebook = new_sum / jnp.maximum(new_count[:, None], 1e-6)

            # Dead-code reseed: any code j with count < threshold gets
            # replaced by a tiled-and-jittered sample from this batch.
            usage = (new_count >= self.reset_threshold)[:, None]   # (K, 1)
            reseed = _tile(z_e, self.codebook_size, jitter=self.reseed_jitter)
            final_codebook = jnp.where(usage, new_codebook, reseed)
            # Re-initialize counts for revived codes so they don't
            # immediately die again on the next step.
            final_count = jnp.where(usage[:, 0], new_count, 1.0)
            final_sum = jnp.where(usage, new_sum, final_codebook)

            self.code_sum.value = final_sum
            self.code_count.value = final_count
            self.codebook.value = final_codebook

        info = {
            "commit": commit,
            "perplexity": perplexity,
            "codebook_utilization": utilization,
        }
        return z_st, codes, info


class MultiGroupQuantizer(nn.Module):
    """Product-space VQ: split latent along channel axis into G groups,
    quantize each independently against its own EMA codebook,
    concatenate the quantized chunks.

    Mirrors ``csbev/model/quantizer.py::MultiGroupQuantizer``. The
    paper-headline configuration is G=2 with (K_1, K_2)=(8, 16). For
    the c2a port we natively support arbitrary G with per-group
    ``(K, D)`` configuration so the body-part split (root=4, axial=8,
    limb=12) maps to G=3 groups.

    Per-frame codes are emitted as a ``(N, G)`` int32 array (one per
    group). Downstream tokenization folds these into a single product
    index — that is NOT the paper convention (the paper reports CB1
    and CB2 separately), but matches the c2a downstream
    ``nn.Embed(num_codes, ...)`` interface.

    Attributes:
        group_dims: per-group ``code_dim``. Sum must equal the input
            latent dim. e.g. ``(4, 8, 12)`` for the v15 body-part split.
        group_codebook_sizes: per-group ``K``. Same length as
            ``group_dims``. e.g. ``(8, 8, 16)``.
        decay: shared EMA decay.
        reset_threshold: shared dead-code reseed trigger.
    """

    group_dims: Sequence[int]
    group_codebook_sizes: Sequence[int]
    decay: float = 0.99
    reset_threshold: float = 1.0
    reseed_jitter: float = 0.01

    @nn.compact
    def __call__(
        self,
        z_e: jnp.ndarray,
        *,
        train: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
        if len(self.group_dims) != len(self.group_codebook_sizes):
            raise ValueError(
                f"group_dims has length {len(self.group_dims)} but "
                f"group_codebook_sizes has length "
                f"{len(self.group_codebook_sizes)} — must match."
            )
        if int(z_e.shape[-1]) != int(sum(self.group_dims)):
            raise ValueError(
                f"MultiGroupQuantizer: z_e.shape[-1]={z_e.shape[-1]} != "
                f"sum(group_dims)={sum(self.group_dims)}."
            )

        # Split along channel dim into G groups.
        offsets = np.cumsum([0] + list(self.group_dims))
        groups = [
            z_e[..., offsets[g]:offsets[g + 1]]
            for g in range(len(self.group_dims))
        ]

        z_st_groups: list[jnp.ndarray] = []
        codes_groups: list[jnp.ndarray] = []
        commits: list[jnp.ndarray] = []
        perplexities: list[jnp.ndarray] = []
        utilizations: list[jnp.ndarray] = []

        for g, (z_g, K_g, D_g) in enumerate(
            zip(groups, self.group_codebook_sizes, self.group_dims)
        ):
            q = QuantizeEMAReset(
                codebook_size=int(K_g),
                code_dim=int(D_g),
                decay=self.decay,
                reset_threshold=self.reset_threshold,
                reseed_jitter=self.reseed_jitter,
                name=f"group_{g}",
            )
            z_st_g, codes_g, info_g = q(z_g, train=train)
            z_st_groups.append(z_st_g)
            codes_groups.append(codes_g)
            commits.append(info_g["commit"])
            perplexities.append(info_g["perplexity"])
            utilizations.append(info_g["codebook_utilization"])

        z_st = jnp.concatenate(z_st_groups, axis=-1)
        # codes: (N, G) int32 — paper convention is per-group codes
        # remain separate; downstream may fold to product index.
        codes = jnp.stack(codes_groups, axis=-1).astype(jnp.int32)

        # Loss / metrics: paper sums per-group commit means.
        info = {
            "commit": sum(commits),
            "perplexity": jnp.mean(jnp.stack(perplexities)),
            "codebook_utilization": jnp.mean(jnp.stack(utilizations)),
        }
        # Also forward per-group metrics for finer-grained logging.
        for g, p in enumerate(perplexities):
            info[f"perplexity_g{g}"] = p
            info[f"utilization_g{g}"] = utilizations[g]
        return z_st, codes, info
