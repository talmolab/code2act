"""Sequential (recurrent) single-step inference for the Mamba code prior.

``c2a/ar_codegen/models/mamba.py`` implements Mamba's selective scan with
``jax.lax.associative_scan`` over the *full* sequence — a parallel prefix scan
with no constant-size step API. To use the trained Mamba as an **in-loop** code
prior (one token per decision step, threaded through a brax env ``State``), we
need the *sequential* form of the same linear recurrence.

This module reimplements one timestep of ``MambaLM.__call__`` as a pure function
over the trained param pytree, carrying per-layer:

* ``conv_buf`` — the last ``d_conv`` inputs to the depthwise causal conv
  (so ``conv_out[c] = Σ_k kernel[k,0,c]·conv_buf[k,c] + bias[c]``), and
* ``ssm`` — the diagonal SSM state ``x[d, n]`` (so
  ``x ← ΔA·x + ΔBu``; ``y = Σ_n C_n·x[:,n] + D·u``).

Correctness is guaranteed by ``test_mamba_recurrent_equivalence`` (step-by-step
logits == full-sequence ``MambaLM`` logits to ~1e-4). It is additive: the
parallel ``__call__`` / ``selective_scan`` / ``sample`` are untouched.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

from c2a.ar_codegen.models.mamba import BOS_ID, MambaConfig


def _rms_norm(x: jnp.ndarray, scale: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return scale * (x / rms)


def init_recurrent_state(config: MambaConfig) -> list[dict[str, jnp.ndarray]]:
    """Zero per-layer recurrent state for ONE sequence (unbatched).

    Returns a list (one entry per layer) of ``{"conv_buf": [d_conv, d_inner],
    "ssm": [d_inner, d_state]}``. Zero conv buffer reproduces the parallel conv's
    left zero-padding; zero SSM state is the scan's identity prefix.
    """
    d_inner = config.d_model * config.expand
    return [
        {
            "conv_buf": jnp.zeros((config.d_conv, d_inner), dtype=jnp.float32),
            "ssm": jnp.zeros((d_inner, config.d_state), dtype=jnp.float32),
        }
        for _ in range(config.n_layers)
    ]


def _block_step(
    bp: dict,
    cfg: MambaConfig,
    layer_state: dict[str, jnp.ndarray],
    x: jnp.ndarray,  # (d_model,)
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """One MambaBlock timestep. Returns ``(out[d_model], new_layer_state)``."""
    d_inner = cfg.d_model * cfg.expand
    dt_rank = cfg.resolved_dt_rank()

    # in_proj: (d_model,) -> (2*d_inner,); split x_main, z gate.
    xz = x @ bp["in_proj"]["kernel"]
    if "bias" in bp["in_proj"]:
        xz = xz + bp["in_proj"]["bias"]
    x_main, z = jnp.split(xz, 2, axis=-1)  # (d_inner,) each

    # Depthwise causal conv1d via the rolling buffer (oldest..newest).
    conv_buf = jnp.concatenate([layer_state["conv_buf"][1:], x_main[None]], axis=0)
    k = bp["conv1d"]["kernel"][:, 0, :]  # (d_conv, d_inner)
    conv_out = jnp.einsum("kc,kc->c", k, conv_buf)
    if "bias" in bp["conv1d"]:
        conv_out = conv_out + bp["conv1d"]["bias"]
    u = jax.nn.silu(conv_out)  # (d_inner,) — the scan input

    # x_proj -> dt, B, C
    x_dbl = u @ bp["x_proj"]["kernel"]  # (dt_rank + 2*d_state,)
    dt, B_in, C_in = jnp.split(x_dbl, [dt_rank, dt_rank + cfg.d_state], axis=-1)
    dt = dt @ bp["dt_proj"]["kernel"]  # (d_inner,)

    delta = jax.nn.softplus(dt.astype(jnp.float32) + bp["dt_bias"].astype(jnp.float32))
    A = -jnp.exp(bp["A_log"].astype(jnp.float32))  # (d_inner, d_state)
    delta_A = jnp.exp(delta[:, None] * A)  # (d_inner, d_state)
    delta_Bu = (
        delta[:, None] * B_in.astype(jnp.float32)[None, :] * u.astype(jnp.float32)[:, None]
    )  # (d_inner, d_state)
    ssm = delta_A * layer_state["ssm"] + delta_Bu
    y = jnp.einsum("dn,n->d", ssm, C_in.astype(jnp.float32))
    y = y + bp["D_skip"].astype(jnp.float32) * u.astype(jnp.float32)
    y = (y * jax.nn.silu(z.astype(jnp.float32))).astype(x.dtype)

    out = y @ bp["out_proj"]["kernel"]
    if "bias" in bp["out_proj"]:
        out = out + bp["out_proj"]["bias"]
    return out, {"conv_buf": conv_buf, "ssm": ssm}


def mamba_step(
    params: dict,
    config: MambaConfig,
    state: list[dict[str, jnp.ndarray]],
    token: jnp.ndarray,
) -> tuple[jnp.ndarray, list[dict[str, jnp.ndarray]]]:
    """One Mamba timestep (unbatched). Consume ``token``, return ``(logits[vocab],
    new_state)``.

    ``token`` is the *input* token at this position (the shifted-right id): feed
    ``BOS`` first, then the previous actual token, to obtain the next-code logits
    — mirroring ``MambaLM``'s ``shift_right`` so logits match the parallel forward.
    """
    x = params["embed"]["embedding"][token]  # (d_model,) ; dropout is identity at inference
    new_state = []
    for i in range(config.n_layers):
        normed = _rms_norm(x, params[f"norm_{i}"]["scale"])
        out, ls = _block_step(params[f"block_{i}"], config, state[i], normed)
        x = x + out
        new_state.append(ls)
    x = _rms_norm(x, params["final_norm"]["scale"])
    logits = x @ params["logits"]["kernel"]
    if "bias" in params["logits"]:
        logits = logits + params["logits"]["bias"]
    return logits.astype(jnp.float32), new_state
