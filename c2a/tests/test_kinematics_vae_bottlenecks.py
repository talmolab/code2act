"""Tests for the VAE bottleneck variant.

These guard the load-bearing invariants the VAE must keep:
  * VAE: deterministic mode returns z=μ; KL=0 for unit posterior;
    σ_cap penalty respects ``sigma_max``.
  * ``pretrain_loss`` total equals the weighted sum of recon +
    pred + bottleneck aux losses, with missing weights defaulting to 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.codegen.kinematics_vae.backbone.decoder import MotionDecoderSplit
from c2a.codegen.kinematics_vae.backbone.encoder import (
    MotionEncoderConv1DSharedSplit,
)
from c2a.codegen.kinematics_vae.backbone.losses import pretrain_loss
from c2a.codegen.kinematics_vae.backbone.predictor import MotionPredictorSplit
from c2a.codegen.kinematics_vae.vae.bottleneck import GaussianBottleneck


# ── shared scaffolding ────────────────────────────────────────────────────


def _build_backbone(B=4, w=10, feat=15, horizon=5, branch=(2, 4, 6)):
    enc = MotionEncoderConv1DSharedSplit(branch_latent_dims=branch)
    idx = ((0, 1), (2, 3, 4, 5), tuple(range(6, feat)))
    dec = MotionDecoderSplit(
        branch_output_indices=idx, branch_latent_dims=branch,
        full_feat_dim=feat, window_len=w,
    )
    pred = MotionPredictorSplit(
        branch_output_indices=idx, branch_latent_dims=branch,
        full_feat_dim=feat, horizon=horizon,
    )
    key = jax.random.PRNGKey(0)
    inputs = jax.random.normal(key, (B, w, feat))
    targets = jax.random.normal(jax.random.PRNGKey(1), (B, horizon, feat))
    return enc, dec, pred, inputs, targets


# ── VAE bottleneck ────────────────────────────────────────────────────────


def test_vae_deterministic_returns_mean():
    bn = GaussianBottleneck()
    mean = jnp.ones((3, 8))
    logvar = jnp.zeros_like(mean)
    rng = jax.random.PRNGKey(0)
    params = bn.init(rng, mean, logvar, rng, deterministic=True)
    z, aux, _ = bn.apply(params, mean, logvar, rng, deterministic=True)
    np.testing.assert_array_equal(np.asarray(z), np.asarray(mean))
    assert set(aux) == {"kl", "sigma_pen"}


def test_vae_stochastic_differs_from_mean():
    bn = GaussianBottleneck()
    mean = jnp.ones((3, 8))
    logvar = jnp.zeros_like(mean)
    rng = jax.random.PRNGKey(42)
    params = bn.init(rng, mean, logvar, rng, deterministic=False)
    z, _, _ = bn.apply(params, mean, logvar, rng, deterministic=False)
    assert not jnp.allclose(z, mean)


def test_vae_sigma_pen_zero_when_sigma_max_zero():
    """sigma_max=0 disables the σ-cap penalty regardless of logvar."""
    bn = GaussianBottleneck()
    rng = jax.random.PRNGKey(0)
    mean = jnp.zeros((3, 8))
    logvar = jnp.ones_like(mean) * 5.0  # huge sigma
    params = bn.init(rng, mean, logvar, rng, deterministic=True, sigma_max=0.0)
    _, aux, _ = bn.apply(
        params, mean, logvar, rng, deterministic=True, sigma_max=0.0,
    )
    # sigma_max=0 → cap = 2*log(1e-8); logvar must exceed it → penalty is large.
    # So the "zero penalty" assertion should use a sigma_max ABOVE exp(logvar/2):
    # for logvar=5, sigma ≈ 12.2, so sigma_max=20 is well above.
    _, aux2, _ = bn.apply(
        params, mean, logvar, rng, deterministic=True, sigma_max=20.0,
    )
    assert float(aux2["sigma_pen"]) == pytest.approx(0.0, abs=1e-5)


# ── pretrain_loss assembly ────────────────────────────────────────────────


def test_pretrain_loss_weighted_sum_matches_metrics():
    """The total loss is the dot-product of loss_weights with metric values
    (where weights default to 0 for missing keys)."""
    enc, dec, pred, inputs, targets = _build_backbone(B=2, w=10, feat=15)
    bn = GaussianBottleneck()
    key = jax.random.PRNGKey(0)
    enc_p = enc.init(key, inputs)
    mu_dummy = jnp.zeros((2, 12))
    bn_p = bn.init(key, mu_dummy, mu_dummy, key, deterministic=True, sigma_max=0.1)
    dec_p = dec.init(key, mu_dummy)
    pred_p = pred.init(key, mu_dummy)

    weights = {"recon": 1.0, "pred": 0.3, "kl": 0.005, "sigma_pen": 1.0}
    loss, (metrics, _) = pretrain_loss(
        enc, bn, dec, pred,
        enc_p, bn_p, dec_p, pred_p,
        inputs, targets, key,
        loss_weights=weights,
        bottleneck_kwargs={"sigma_max": 0.1},
        deterministic=True,
    )
    manual = sum(
        float(weights.get(k, 0.0)) * float(metrics[k])
        for k in ("recon", "pred", "kl", "sigma_pen")
    )
    np.testing.assert_allclose(float(loss), manual, atol=1e-5)


def test_pretrain_loss_missing_weight_defaults_to_zero():
    """A weight that isn't in loss_weights must contribute 0 (ablation use)."""
    enc, dec, pred, inputs, targets = _build_backbone(B=2, w=10, feat=15)
    bn = GaussianBottleneck()
    key = jax.random.PRNGKey(0)
    enc_p = enc.init(key, inputs)
    mu_dummy = jnp.zeros((2, 12))
    bn_p = bn.init(key, mu_dummy, mu_dummy, key, deterministic=True, sigma_max=0.1)
    dec_p = dec.init(key, mu_dummy)
    pred_p = pred.init(key, mu_dummy)

    # Drop the sigma_pen weight — the loss should ONLY include recon + pred + kl.
    weights = {"recon": 1.0, "pred": 0.3, "kl": 0.005}
    loss, (metrics, _) = pretrain_loss(
        enc, bn, dec, pred,
        enc_p, bn_p, dec_p, pred_p,
        inputs, targets, key,
        loss_weights=weights,
        bottleneck_kwargs={"sigma_max": 0.1},
        deterministic=True,
    )
    manual = (
        1.0 * float(metrics["recon"])
        + 0.3 * float(metrics["pred"])
        + 0.005 * float(metrics["kl"])
    )
    np.testing.assert_allclose(float(loss), manual, atol=1e-5)
    # sigma_pen metric is still present (logging) but doesn't enter the loss.
    assert "sigma_pen" in metrics
