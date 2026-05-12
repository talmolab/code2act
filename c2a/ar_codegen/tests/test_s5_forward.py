"""Smoke + causality tests for S5Generator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.ar_codegen.models.s5 import S5Config, _build_model, _make_s5_init


@pytest.fixture(scope="module")
def setup():
    num_codes = 8
    cfg = S5Config(
        vocab_size=num_codes + 1, H=32, P=16, n_layers=2,
        max_len=16, dropout_rate=0.0, deterministic=True, bidirectional=False,
    )
    init_arrays = _make_s5_init(cfg.P, cfg.conj_sym)
    model = _build_model(cfg, init_arrays)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 10), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, dummy)["params"]
    return model, params, cfg, num_codes


def test_forward_shape(setup):
    model, params, cfg, num_codes = setup
    inputs = jnp.asarray(
        np.random.RandomState(2).randint(1, num_codes + 1, size=(3, 10), dtype=np.int32),
    )
    logits = model.apply({"params": params}, inputs)
    assert logits.shape == (3, 10, cfg.vocab_size), logits.shape


def test_deterministic(setup):
    model, params, cfg, num_codes = setup
    inputs = jnp.asarray(
        np.random.RandomState(43).randint(1, num_codes + 1, size=(2, 8), dtype=np.int32),
    )
    a = model.apply({"params": params}, inputs)
    b = model.apply({"params": params}, inputs)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_causal(setup):
    """Unidirectional S5 + BOS shift → past unaffected by future perturbation."""
    model, params, cfg, num_codes = setup
    base = np.random.RandomState(13).randint(1, num_codes + 1, size=(1, 10), dtype=np.int32)
    a = np.asarray(model.apply({"params": params}, jnp.asarray(base)))

    perturbed = base.copy()
    new = (perturbed[0, 6] + 1) % (num_codes + 1)
    perturbed[0, 6] = new if new != 0 else 1
    b = np.asarray(model.apply({"params": params}, jnp.asarray(perturbed)))

    np.testing.assert_allclose(a[:, :6], b[:, :6], atol=1e-4)
