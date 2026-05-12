"""Forward-pass + scan-parity tests for MambaGenerator.

The scan-parity test is the load-bearing correctness check: our
pure-JAX :func:`selective_scan` must match a slow-loop reference on
random inputs to ~1e-4. If it doesn't, the JAX port is wrong.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.ar_codegen.models.mamba import (
    MambaConfig, MambaLM, selective_scan,
)


# ---------------------------------------------------------------------------
# Slow-loop reference (mirrors deep_ssm/mixers/utils_mamba.py:selective_scan_ref)
# ---------------------------------------------------------------------------


def _selective_scan_ref_numpy(u, delta, A, B, C, D_skip, delta_bias, delta_softplus=True):
    """NumPy slow loop — the ground truth selective_scan must match."""
    B_size, D_size, L = u.shape
    N = A.shape[1]
    u_f = u.astype(np.float32)
    delta_f = delta.astype(np.float32) + delta_bias.astype(np.float32)[None, :, None]
    if delta_softplus:
        delta_f = np.log1p(np.exp(-np.abs(delta_f))) + np.maximum(delta_f, 0)  # softplus

    # discretize
    deltaA = np.exp(np.einsum("bdl,dn->bdln", delta_f, A.astype(np.float32)))  # (B,D,L,N)
    deltaBu = np.einsum("bdl,bnl,bdl->bdln", delta_f, B.astype(np.float32), u_f)

    x = np.zeros((B_size, D_size, N), dtype=np.float32)
    ys = np.zeros((B_size, D_size, L), dtype=np.float32)
    for t in range(L):
        x = deltaA[:, :, t] * x + deltaBu[:, :, t]
        y = np.einsum("bdn,bn->bd", x, C[:, :, t].astype(np.float32))
        ys[:, :, t] = y + D_skip.astype(np.float32)[None, :] * u_f[:, :, t]
    return ys


def test_selective_scan_matches_reference():
    """JAX selective_scan output ≈ NumPy reference on random inputs."""
    rng = np.random.RandomState(0)
    B_size, D_size, L, N = 2, 4, 6, 3

    u = rng.randn(B_size, D_size, L).astype(np.float32)
    delta = rng.randn(B_size, D_size, L).astype(np.float32) * 0.3
    A = -np.exp(rng.randn(D_size, N).astype(np.float32))  # negative real
    B = rng.randn(B_size, N, L).astype(np.float32)
    C = rng.randn(B_size, N, L).astype(np.float32)
    D_skip = rng.randn(D_size).astype(np.float32)
    delta_bias = rng.randn(D_size).astype(np.float32) * 0.1

    y_ref = _selective_scan_ref_numpy(u, delta, A, B, C, D_skip, delta_bias, delta_softplus=True)
    y_jax = np.asarray(
        selective_scan(
            jnp.asarray(u), jnp.asarray(delta), jnp.asarray(A),
            jnp.asarray(B), jnp.asarray(C), jnp.asarray(D_skip),
            jnp.asarray(delta_bias), delta_softplus=True,
        )
    )

    np.testing.assert_allclose(y_jax, y_ref, atol=1e-4, rtol=1e-3)


@pytest.fixture(scope="module")
def setup():
    num_codes = 8
    cfg = MambaConfig(
        vocab_size=num_codes + 1,
        d_model=32, n_layers=2, d_state=8, d_conv=3, expand=2,
        max_len=16, dropout_rate=0.0, deterministic=True,
    )
    model = MambaLM(config=cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, 10), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, dummy)["params"]
    return model, params, cfg, num_codes


def test_forward_shape(setup):
    model, params, cfg, num_codes = setup
    B, T = 4, 10
    inputs = jnp.asarray(
        np.random.RandomState(1).randint(1, num_codes + 1, size=(B, T), dtype=np.int32),
    )
    logits = model.apply({"params": params}, inputs)
    assert logits.shape == (B, T, cfg.vocab_size), logits.shape


def test_deterministic(setup):
    model, params, cfg, num_codes = setup
    inputs = jnp.asarray(
        np.random.RandomState(42).randint(1, num_codes + 1, size=(2, 8), dtype=np.int32),
    )
    a = model.apply({"params": params}, inputs)
    b = model.apply({"params": params}, inputs)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_causal_via_conv1d_padding(setup):
    """Perturbing future tokens shouldn't affect past logits."""
    model, params, cfg, num_codes = setup
    base = np.random.RandomState(11).randint(1, num_codes + 1, size=(1, 10), dtype=np.int32)
    logits_a = np.asarray(model.apply({"params": params}, jnp.asarray(base)))

    perturbed = base.copy()
    new = (perturbed[0, 7] + 1) % (num_codes + 1)
    perturbed[0, 7] = new if new != 0 else 1
    logits_b = np.asarray(model.apply({"params": params}, jnp.asarray(perturbed)))

    np.testing.assert_allclose(logits_a[:, :7], logits_b[:, :7], atol=1e-4)
