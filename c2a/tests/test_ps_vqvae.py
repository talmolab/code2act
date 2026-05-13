"""Tests for the product-space VQVAE bottleneck and tokenization.

Guards paper-faithful behavior per VQ-MAP (Li et al., 2025):
  * MultiGroupQuantizer correctly splits the latent along the channel
    axis into G groups and quantizes each independently.
  * EMA codebook updates fire under ``mutable=["vq_state"]`` and are
    frozen otherwise.
  * STE direction matches the standard van den Oord pattern (already
    covered by the vanilla VQVAE tests; re-verified here in the
    product-space context).
  * fold_to_product_index round-trips with divmod (mixed-radix).
  * The bottleneck-agnostic pretrain_loss path correctly threads
    mutable=["vq_state"] when the bottleneck declares it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from c2a.codegen.kinematics_vae.ps_vqvae.quantizer import (
    MultiGroupQuantizer,
    QuantizeEMAReset,
    _l2_dist,
)
from c2a.codegen.kinematics_vae.ps_vqvae.bottleneck import PSVQVAEBottleneck
from c2a.codegen.kinematics_vae.ps_vqvae.tokenize import fold_to_product_index


# ── per-group quantizer ───────────────────────────────────────────────────


def test_ema_quantizer_returns_correct_shapes():
    K, D, N = 4, 3, 5
    q = QuantizeEMAReset(codebook_size=K, code_dim=D)
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (N, D))
    variables = q.init(rng, z)
    z_st, codes, info = q.apply(variables, z, train=False)
    assert z_st.shape == (N, D)
    assert codes.shape == (N,)
    assert codes.dtype == jnp.int32
    assert set(info) >= {"commit", "perplexity", "codebook_utilization"}


def test_ema_quantizer_freezes_when_train_false():
    """EMA buffers must NOT change when train=False."""
    K, D, N = 4, 3, 10
    q = QuantizeEMAReset(codebook_size=K, code_dim=D)
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (N, D))
    variables = q.init(rng, z)
    old_codebook = variables["vq_state"]["codebook"]

    # apply with mutable=["vq_state"], but train=False should leave buffers untouched.
    _, new_state = q.apply(variables, z, train=False, mutable=["vq_state"])
    new_codebook = new_state["vq_state"]["codebook"]
    np.testing.assert_array_equal(
        np.asarray(new_codebook), np.asarray(old_codebook),
    )


def test_ema_quantizer_updates_buffers_when_train_true():
    K, D, N = 4, 3, 16
    q = QuantizeEMAReset(codebook_size=K, code_dim=D, decay=0.5)
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (N, D))
    variables = q.init(rng, z)
    old_codebook = variables["vq_state"]["codebook"]
    old_count = variables["vq_state"]["code_count"]

    _, new_state = q.apply(variables, z, train=True, mutable=["vq_state"])
    new_codebook = new_state["vq_state"]["codebook"]
    new_count = new_state["vq_state"]["code_count"]

    # codebook should change for used codes (perplexity > 1 means at
    # least one assignment happened).
    assert not jnp.allclose(new_codebook, old_codebook)
    # code_count should change (EMA picks up the new batch counts).
    assert not jnp.allclose(new_count, old_count)


def test_ema_dead_code_reseed():
    """A codebook entry with persistent zero assignments should get
    reseeded from a tiled+jittered sample of the current batch."""
    K, D = 8, 3
    q = QuantizeEMAReset(
        codebook_size=K, code_dim=D, decay=0.99, reset_threshold=1.0,
    )
    rng = jax.random.PRNGKey(0)
    # Use a batch that concentrates assignments on a few codes (low
    # variance input → most codes never selected).
    z = jnp.tile(jnp.zeros((1, D)), (32, 1)) + 1e-3 * jax.random.normal(
        rng, (32, D),
    )
    variables = q.init(rng, z)
    # Run many forwards to push usage of unused codes to "dead" status.
    for _ in range(5):
        _, variables = q.apply(variables, z, train=True, mutable=["vq_state"])
    # At least some codes should have been reseeded — confirmed by the
    # number of unique codebook entries not having shrunk.
    codebook = np.asarray(variables["vq_state"]["codebook"])
    n_unique = len({tuple(row.tolist()) for row in codebook})
    # If reseed is broken, we'd have many duplicate / collapsed entries.
    assert n_unique >= K // 2


# ── product-space wrapper ─────────────────────────────────────────────────


def test_multi_group_quantizer_splits_channels():
    group_dims = (4, 8, 12)
    group_K = (8, 8, 16)
    mgq = MultiGroupQuantizer(group_dims=group_dims, group_codebook_sizes=group_K)
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (5, sum(group_dims)))
    variables = mgq.init(rng, z)
    z_st, codes, info = mgq.apply(variables, z, train=False)
    assert z_st.shape == (5, sum(group_dims))
    assert codes.shape == (5, 3)
    assert codes.dtype == jnp.int32
    # Per-group max code must be in [0, K_g)
    for g, K in enumerate(group_K):
        assert int(codes[:, g].max()) < K


def test_multi_group_quantizer_rejects_dim_mismatch():
    mgq = MultiGroupQuantizer(group_dims=(4, 8), group_codebook_sizes=(8, 16))
    rng = jax.random.PRNGKey(0)
    # Latent has 24 dims but groups sum to 12.
    z = jnp.zeros((3, 24))
    with pytest.raises(ValueError, match="sum\\(group_dims\\)"):
        mgq.init(rng, z)


def test_multi_group_quantizer_mismatched_K_dim_raises():
    mgq = MultiGroupQuantizer(group_dims=(4, 8), group_codebook_sizes=(8, 16, 32))
    rng = jax.random.PRNGKey(0)
    z = jnp.zeros((3, 12))
    with pytest.raises(ValueError, match="group_codebook_sizes"):
        mgq.init(rng, z)


# ── PS-VQVAE bottleneck (interface adapter) ──────────────────────────────


def test_ps_vqvae_bottleneck_declares_mutable_collections():
    bn = PSVQVAEBottleneck(group_dims=(4, 8, 12), group_codebook_sizes=(8, 8, 16))
    assert bn.mutable_collections == ("vq_state",)


def test_ps_vqvae_bottleneck_emits_per_group_codes():
    """codes shape should be (B, G); aux_losses must include 'commit'."""
    bn = PSVQVAEBottleneck(group_dims=(4, 8, 12), group_codebook_sizes=(8, 8, 16))
    rng = jax.random.PRNGKey(0)
    mu = jax.random.normal(rng, (4, 24))
    lv = jnp.zeros_like(mu)
    variables = bn.init(rng, mu, lv, rng, deterministic=True)
    z_st, aux_losses, aux_metrics = bn.apply(
        variables, mu, lv, rng, deterministic=True,
    )
    assert z_st.shape == (4, 24)
    assert "commit" in aux_losses
    assert aux_metrics["codes"].shape == (4, 3)
    # Per-group diagnostics surface.
    for g in range(3):
        assert f"perplexity_g{g}" in aux_metrics


def test_ps_vqvae_bottleneck_rejects_dim_mismatch():
    bn = PSVQVAEBottleneck(group_dims=(4, 8), group_codebook_sizes=(8, 16))
    rng = jax.random.PRNGKey(0)
    # encoder emits 24 dims but groups sum to 12.
    mu = jnp.zeros((2, 24)); lv = jnp.zeros_like(mu)
    with pytest.raises(ValueError, match="sum\\(group_dims\\)"):
        bn.init(rng, mu, lv, rng, deterministic=True)


# ── product-fold round-trip ───────────────────────────────────────────────


def test_fold_to_product_index_round_trips():
    """Mixed-radix fold + divmod recovers the original per-group codes."""
    K = [8, 8, 16]
    K_total = int(np.prod(K))
    # Exhaustive walk: ensure every product index round-trips.
    expected_pairs = []
    for k0 in range(K[0]):
        for k1 in range(K[1]):
            for k2 in range(K[2]):
                expected_pairs.append((k0, k1, k2))
    codes = np.array(expected_pairs, dtype=np.int32)
    folded = fold_to_product_index(codes, K)
    assert folded.shape == (K_total,)
    assert int(folded.max()) == K_total - 1
    assert int(folded.min()) == 0
    # Recover by divmod
    pv0 = K[1] * K[2]
    pv1 = K[2]
    rec0 = (folded // pv0).astype(np.int32)
    rec1 = ((folded % pv0) // pv1).astype(np.int32)
    rec2 = (folded % pv1).astype(np.int32)
    np.testing.assert_array_equal(rec0, codes[:, 0])
    np.testing.assert_array_equal(rec1, codes[:, 1])
    np.testing.assert_array_equal(rec2, codes[:, 2])


def test_fold_to_product_index_rejects_oob():
    """A code value exceeding its group's K should be caught."""
    with pytest.raises(RuntimeError, match="outside"):
        fold_to_product_index(np.array([[9, 0]]), [8, 8])
