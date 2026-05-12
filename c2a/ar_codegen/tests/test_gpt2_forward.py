"""Forward-pass tests for GPT2Generator.

These are the cheap ones that don't actually fit the model — they just
verify the architecture wires up correctly:
* output shape ``[B, T, vocab=num_codes+1]``
* deterministic forward pass (same params + same input → identical logits)
* causal mask: perturbing a future token leaves past logits unchanged
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.ar_codegen.models.gpt2 import TransformerLM, TransformerConfig


@pytest.fixture(scope="module")
def setup():
    num_codes = 8
    cfg = TransformerConfig(
        vocab_size=num_codes + 1,
        emb_dim=32, num_heads=2, num_layers=2, qkv_dim=32, mlp_dim=64,
        max_len=16, dropout_rate=0.0, attention_dropout_rate=0.0,
        deterministic=True,
    )
    model = TransformerLM(config=cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 10), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, dummy)["params"]
    return model, params, cfg, num_codes


def test_forward_shape(setup):
    model, params, cfg, num_codes = setup
    B, T = 4, 10
    np_rng = np.random.RandomState(123)
    inputs = np_rng.randint(0, num_codes, size=(B, T)).astype(np.int32) + 1  # tokens in [1, num_codes]
    logits = model.apply({"params": params}, jnp.asarray(inputs))
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape


def test_deterministic(setup):
    model, params, cfg, num_codes = setup
    inputs = jnp.asarray(
        np.random.RandomState(42).randint(1, num_codes + 1, size=(2, 8), dtype=np.int32),
    )
    a = model.apply({"params": params}, inputs)
    b = model.apply({"params": params}, inputs)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_causal_mask_blocks_future(setup):
    """Perturbing position t+ should not change logits at positions <= t."""
    model, params, cfg, num_codes = setup
    base = np.random.RandomState(7).randint(1, num_codes + 1, size=(1, 10), dtype=np.int32)
    logits_a = np.asarray(model.apply({"params": params}, jnp.asarray(base)))

    perturbed = base.copy()
    perturbed[0, 6] = (perturbed[0, 6] + 1) % (num_codes + 1)
    if perturbed[0, 6] == 0:
        perturbed[0, 6] = 1  # avoid the BOS token id
    logits_b = np.asarray(model.apply({"params": params}, jnp.asarray(perturbed)))

    # Positions 0..5 should be unchanged.
    np.testing.assert_allclose(logits_a[:, :6], logits_b[:, :6], atol=1e-5)
    # Position 6 onward MAY differ. We expect the change to be > 0 to
    # confirm the perturbation actually had downstream effect.
    assert not np.allclose(logits_a[:, 6:], logits_b[:, 6:], atol=1e-5)
