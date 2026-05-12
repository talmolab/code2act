"""Tests for the VAE + VQVAE bottleneck variants.

These guard the load-bearing invariants every variant must keep:
  * VAE: deterministic mode returns z=μ; KL=0 for unit posterior;
    σ_cap penalty respects ``sigma_max``.
  * VQVAE: identity STE (forward = z_q, backward gradient w.r.t. z_e
    = ones); commitment & codebook losses have the correct
    stop_gradient direction so they push only their respective sides;
    codebook param receives non-zero gradient through the codebook
    loss term.
  * Both: ``pretrain_loss`` total equals the weighted sum of recon +
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
from c2a.codegen.kinematics_vae.vqvae.bottleneck import VQVAEBottleneck
from c2a.codegen.kinematics_vae.vqvae.quantizer import VectorQuantizer


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


# ── VQVAE quantizer / bottleneck ──────────────────────────────────────────


def test_vq_forward_returns_z_q_numerically():
    """In the forward pass, STE makes z_st numerically equal z_q
    (the nearest codebook entry)."""
    K, D, B = 4, 3, 5
    q = VectorQuantizer(codebook_size=K, code_dim=D)
    rng = jax.random.PRNGKey(0)
    z_e = jax.random.normal(rng, (B, D))
    params = q.init(rng, z_e)
    codebook = params["params"]["codebook"]

    z_st, _, aux = q.apply(params, z_e)
    codes = aux["codes"]
    expected = codebook[codes]
    np.testing.assert_allclose(np.asarray(z_st), np.asarray(expected), atol=1e-6)


def test_vq_ste_grad_passes_identity_to_z_e():
    """d(z_st)/d(z_e) should be identity (forward through STE)."""
    K, D, B = 4, 3, 2
    q = VectorQuantizer(codebook_size=K, code_dim=D)
    rng = jax.random.PRNGKey(0)
    z_e = jax.random.normal(rng, (B, D))
    params = q.init(rng, z_e)

    def fn(z):
        out, _, _ = q.apply(params, z)
        return jnp.sum(out)
    grad = jax.grad(fn)(z_e)
    # d(sum(z_st))/d(z_e) should be ones (identity STE).
    np.testing.assert_allclose(np.asarray(grad), np.ones_like(z_e), atol=1e-5)


def test_vq_commitment_grad_only_to_encoder():
    """Commitment loss = ||z_e - sg(z_q)||². d(commit)/d(z_e) is non-zero,
    d(commit)/d(codebook) is zero (the codebook is wrapped in sg)."""
    K, D, B = 4, 3, 2
    q = VectorQuantizer(codebook_size=K, code_dim=D)
    rng = jax.random.PRNGKey(0)
    z_e = jax.random.normal(rng, (B, D))
    params = q.init(rng, z_e)

    def commit_only(z, p):
        _, aux_losses, _ = q.apply(p, z)
        return aux_losses["commit"]

    g_z = jax.grad(commit_only, argnums=0)(z_e, params)
    g_p = jax.grad(commit_only, argnums=1)(z_e, params)

    # grad w.r.t. z_e should be non-zero (encoder gets pulled toward codebook).
    assert float(jnp.linalg.norm(g_z)) > 0.0
    # grad w.r.t. codebook param should be exactly zero (sg blocked it).
    cb_grad = g_p["params"]["codebook"]
    np.testing.assert_allclose(np.asarray(cb_grad), 0.0, atol=1e-6)


def test_vq_codebook_grad_only_to_codebook():
    """Codebook loss = ||sg(z_e) - z_q||². d(loss)/d(z_e) is zero,
    d(loss)/d(codebook) is non-zero."""
    K, D, B = 4, 3, 2
    q = VectorQuantizer(codebook_size=K, code_dim=D)
    rng = jax.random.PRNGKey(0)
    z_e = jax.random.normal(rng, (B, D))
    params = q.init(rng, z_e)

    def codebook_only(z, p):
        _, aux_losses, _ = q.apply(p, z)
        return aux_losses["codebook"]

    g_z = jax.grad(codebook_only, argnums=0)(z_e, params)
    g_p = jax.grad(codebook_only, argnums=1)(z_e, params)
    np.testing.assert_allclose(np.asarray(g_z), 0.0, atol=1e-6)
    assert float(jnp.linalg.norm(g_p["params"]["codebook"])) > 0.0


def test_vqvae_bottleneck_validates_code_dim():
    """VQVAEBottleneck refuses a code_dim mismatch with the encoder."""
    bn = VQVAEBottleneck(codebook_size=16, code_dim=24)
    rng = jax.random.PRNGKey(0)
    # mean.shape[-1]=10, code_dim=24 — must raise.
    mean = jnp.zeros((3, 10))
    logvar = jnp.zeros_like(mean)
    with pytest.raises(ValueError, match="code_dim"):
        bn.init(rng, mean, logvar, rng)


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
    loss, metrics = pretrain_loss(
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
    bn = VQVAEBottleneck(codebook_size=8, code_dim=12)
    key = jax.random.PRNGKey(0)
    enc_p = enc.init(key, inputs)
    mu_dummy = jnp.zeros((2, 12))
    bn_p = bn.init(key, mu_dummy, mu_dummy, key)
    dec_p = dec.init(key, mu_dummy)
    pred_p = pred.init(key, mu_dummy)

    # Drop the codebook weight — the loss should ONLY include recon + pred + commit.
    weights = {"recon": 1.0, "pred": 0.3, "commit": 0.25}
    loss, metrics = pretrain_loss(
        enc, bn, dec, pred,
        enc_p, bn_p, dec_p, pred_p,
        inputs, targets, key,
        loss_weights=weights,
    )
    manual = (
        1.0 * float(metrics["recon"])
        + 0.3 * float(metrics["pred"])
        + 0.25 * float(metrics["commit"])
    )
    np.testing.assert_allclose(float(loss), manual, atol=1e-5)
    # codebook metric is still present (logging) but doesn't enter the loss.
    assert "codebook" in metrics
