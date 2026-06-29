"""Bottleneck-agnostic pretraining loss for the kinematics-VAE family.

The Scott v15 ELBO loss is bottleneck-agnostic: the kinematics-VAE
plugs in its Gaussian bottleneck while sharing the encoder / decoder /
predictor stack. A bottleneck is a ``Bottleneck`` Flax module with the
following signature::

    bottleneck.apply(bn_params, mean, logvar, rng, deterministic=False)
        -> (z, aux_losses, aux_metrics)

where:
    ``z``          : ``(B, latent_dim)`` the latent passed to the decoder
                     and predictor.
    ``aux_losses`` : dict of named scalar losses (e.g. ``{"kl": ...,
                     "sigma_pen": ...}`` for the VAE). Summed into the
                     total via ``loss_weights``.
    ``aux_metrics``: dict of named scalars not added to the loss but
                     forwarded to the logger (e.g. ``{"perplexity": ...,
                     "codebook_utilization": ...}``).

This module exports the shared utilities (KL, masked MSE, σ_cap) and the
loss-assembly function :func:`pretrain_loss` every variant's training
step calls.

KL / reparam / masked-MSE / σ_cap are ported verbatim from
``origin/scott_claude/latent-mimic-handoff:track_mjx/agent/latent_ppo/
losses/pretrain_losses.py``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp


# =============================================================================
# Pure utilities (used by individual bottlenecks)
# =============================================================================


def kl_to_unit_gaussian(
    mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """Mean-over-batch KL(N(mean, exp(logvar)) || N(0, I)), sum over latent dims."""
    per_dim = -0.5 * (1.0 + logvar - jnp.square(mean) - jnp.exp(logvar))
    return jnp.mean(jnp.sum(per_dim, axis=-1))


def reparameterize(
    rng: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    eps = jax.random.normal(rng, shape=mean.shape)
    return mean + jnp.exp(0.5 * logvar) * eps


def masked_mse(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    feat_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Mean of squared error, restricted to ``feat_mask=1`` channels along
    the last axis. ``None`` means "use all channels equally"."""
    err = jnp.square(pred - target)
    if feat_mask is None:
        return jnp.mean(err)
    # err: (..., feat). Average over leading axes per dim, then mean over
    # active dims only.
    per_dim = jnp.mean(err, axis=tuple(range(err.ndim - 1)))   # (feat,)
    return (per_dim * feat_mask).sum() / jnp.maximum(feat_mask.sum(), 1.0)


def sigma_cap_penalty(
    logvar: jnp.ndarray, sigma_max: float | jnp.ndarray
) -> jnp.ndarray:
    """Hinge penalty pushing posterior std below ``sigma_max``.

    Per-dim: ``relu(logvar - log(sigma_max^2))``. Sum over latent dims,
    mean over batch. Encourages the encoder to use sharp posteriors so
    reconstructions are not blurred by reparameterization noise during
    training.

    ``sigma_max`` may be either a Python float (fixed cap) or a JAX
    scalar (curriculum schedule). Floor at ``1e-8`` to keep ``log`` finite.
    """
    sm = jnp.maximum(jnp.asarray(sigma_max, dtype=logvar.dtype), 1e-8)
    cap = 2.0 * jnp.log(sm)
    excess = jnp.maximum(logvar - cap, 0.0)   # (B, latent)
    return jnp.mean(jnp.sum(excess, axis=-1))


# =============================================================================
# Bottleneck-agnostic loss assembly
# =============================================================================


def pretrain_loss(
    encoder,
    bottleneck,
    decoder,
    predictor,
    enc_params,
    bn_variables,
    dec_params,
    pred_params,
    inputs: jnp.ndarray,        # (B, w, feat)
    targets: jnp.ndarray,       # (B, n, feat)
    rng: jax.Array,
    loss_weights: Mapping[str, float | jnp.ndarray],
    bottleneck_kwargs: Mapping[str, Any] | None = None,
    feat_mask: Optional[jnp.ndarray] = None,
    deterministic: bool = False,
    bottleneck_mutable: list[str] | None = None,
) -> tuple[jnp.ndarray, tuple[dict[str, jnp.ndarray], dict[str, Any]]]:
    """Single forward pass; return ``(total_loss, (metrics, new_bn_variables))``.

    Weights:
        ``loss_weights['recon']``  — reconstruction MSE (default callers pass 1.0).
        ``loss_weights['pred']``   — next-horizon MSE (default callers pass 0.3).
        ``loss_weights[<aux>]``    — every key the bottleneck returns in
                                      ``aux_losses`` (e.g. ``kl``, ``sigma_pen``,
                                      ``commit``, ``codebook``).

    Missing weights default to 0 so a bottleneck whose aux loss isn't in
    ``loss_weights`` contributes nothing — useful for ablations.

    ``bottleneck_kwargs`` carries per-step runtime arguments to the
    bottleneck that aren't loss coefficients — e.g. ``{"sigma_max":
    <jnp scalar>}`` for the VAE σ-cap schedule.

    ``bottleneck_mutable`` is the list of Flax variable collections the
    bottleneck mutates per forward — e.g. ``["vq_state"]`` for bottlenecks
    with EMA buffers. When provided, ``bn_variables`` must be a Flax
    variables dict (``{"params": ..., "vq_state": ...}``); when ``None``
    (default — the VAE), ``bn_variables`` is the bare params subtree as
    before.

    Returns:
        ``(total_loss, (metrics, new_bn_variables))``. For the
        non-mutable path ``new_bn_variables`` is None — the train loop
        ignores it.
    """
    mean, logvar = encoder.apply(enc_params, inputs)
    if bottleneck_mutable:
        # Mutable-state path: bn_variables is a full Flax variables dict.
        (z, aux_losses, aux_metrics), new_bn_state = bottleneck.apply(
            bn_variables, mean, logvar, rng,
            deterministic=deterministic,
            mutable=list(bottleneck_mutable),
            **(bottleneck_kwargs or {}),
        )
    else:
        # Pure-function path: bn_variables IS the params subtree.
        z, aux_losses, aux_metrics = bottleneck.apply(
            bn_variables, mean, logvar, rng,
            deterministic=deterministic,
            **(bottleneck_kwargs or {}),
        )
        new_bn_state = None
    recon = decoder.apply(dec_params, z)
    pred = predictor.apply(pred_params, z)

    recon_loss = masked_mse(recon, inputs, feat_mask)
    pred_loss = masked_mse(pred, targets, feat_mask)

    total = (
        loss_weights.get("recon", 1.0) * recon_loss
        + loss_weights.get("pred", 0.0) * pred_loss
    )
    for name, value in aux_losses.items():
        total = total + loss_weights.get(name, 0.0) * value

    metrics: dict[str, jnp.ndarray] = {
        "recon": recon_loss,
        "pred": pred_loss,
    }
    metrics.update(aux_losses)
    metrics.update(aux_metrics)
    return total, (metrics, new_bn_state)
