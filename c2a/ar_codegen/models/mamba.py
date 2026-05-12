"""Mamba — selective state-space model, pure JAX/Flax port.

Ported from ``ekellbuch/deep_ssm/src/deep_ssm/mixers/mamba_simple.py``
(Apache-2.0 — derived from ``state-spaces/mamba`` by Tri Dao & Albert Gu).
The PyTorch original uses a fused CUDA ``selective_scan_fn`` kernel
plus a ``causal_conv1d`` CUDA kernel; this port replaces those with:

* :func:`jax.lax.associative_scan` for the per-channel diagonal SSM
  recurrence (the same operator that the lindermanlab/S5 codebase
  uses — linear in time, no kernel dependency).
* :class:`flax.linen.Conv` with left-only padding for the causal
  depthwise 1D convolution.

The forward pass is dimensionally identical to the reference; numerical
match is to ~1e-4 on random inputs (test in
``tests/test_mamba_forward.py``).

Architecture defaults are sized for ~17 k training tokens:

* ``d_model``  128
* ``n_layers``   4
* ``d_state``   16
* ``d_conv``     4
* ``expand``     2

→ ~600 k params.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state

from c2a.ar_codegen.models.base import CodeGenerator, GeneratorMeta
from c2a.ar_codegen.wandb_utils import _log_code_dist, code_histogram, log_metrics

log = logging.getLogger(__name__)


BOS_ID = 0
PARAMS_FILENAME = "params.msgpack"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@struct.dataclass
class MambaConfig:
    """Hyperparameters for the Mamba stack."""

    vocab_size: int  # = num_codes + 1 (BOS at 0)
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | str = "auto"  # int or "auto" → ceil(d_model/16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    dropout_rate: float = 0.1
    max_len: int = 256
    deterministic: bool = False
    dtype: Any = jnp.float32

    def resolved_dt_rank(self) -> int:
        if isinstance(self.dt_rank, str):
            if self.dt_rank != "auto":
                raise ValueError(f"dt_rank must be int or 'auto', got {self.dt_rank!r}")
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)


# ---------------------------------------------------------------------------
# Selective scan (the load-bearing piece)
# ---------------------------------------------------------------------------


def selective_scan(
    u: jnp.ndarray,        # (B, D, L)
    delta: jnp.ndarray,    # (B, D, L)
    A: jnp.ndarray,        # (D, N) — negative real diagonal eigenvalues
    B: jnp.ndarray,        # (B, N, L) — input-dependent
    C: jnp.ndarray,        # (B, N, L) — input-dependent
    D_skip: jnp.ndarray,   # (D,)
    delta_bias: jnp.ndarray,  # (D,)
    delta_softplus: bool = True,
) -> jnp.ndarray:
    """Diagonal selective scan — pure-JAX reference.

    Mirrors :func:`selective_scan_ref` in
    ``deep_ssm/mixers/utils_mamba.py``: real (non-complex) A only;
    variable input-dependent B, C with 3 dims; no group repetition.

    Discretization (zero-order-hold-ish; the original uses
    ``exp(Δ·A)`` directly which is the standard Mamba choice):

        ΔA[b, d, t, n] = exp((softplus(Δ[b, d, t] + Δ_bias[d])) · A[d, n])
        ΔBu[b, d, t, n] = (softplus(Δ[b, d, t] + Δ_bias[d])) · B[b, n, t] · u[b, d, t]

    Recurrence:
        x[b, d, t, n] = ΔA[b, d, t, n] · x[b, d, t-1, n] + ΔBu[b, d, t, n]
        y[b, d, t]    = Σ_n C[b, n, t] · x[b, d, t, n] + D_skip[d] · u[b, d, t]

    Returns ``y`` with shape ``(B, D, L)``.
    """
    B_size, D_size, L = u.shape
    N = A.shape[1]
    assert A.shape == (D_size, N), f"A shape {A.shape} mismatched"
    assert B.shape == (B_size, N, L), f"B shape {B.shape}"
    assert C.shape == (B_size, N, L), f"C shape {C.shape}"
    assert delta.shape == (B_size, D_size, L), f"delta shape {delta.shape}"
    assert D_skip.shape == (D_size,), f"D_skip shape {D_skip.shape}"
    assert delta_bias.shape == (D_size,), f"delta_bias shape {delta_bias.shape}"

    # cast to float32 for stability — matches the reference's u/delta cast.
    u_f = u.astype(jnp.float32)
    delta_f = delta.astype(jnp.float32) + delta_bias.astype(jnp.float32)[None, :, None]
    if delta_softplus:
        delta_f = jax.nn.softplus(delta_f)  # (B, D, L)

    # ΔA: (B, D, L, N) — diagonal selective state-transition.
    # einsum 'bdl,dn->bdln' (delta * A)
    delta_A = jnp.exp(jnp.einsum("bdl,dn->bdln", delta_f, A.astype(jnp.float32)))
    # ΔBu: (B, D, L, N) — selective input.
    delta_Bu = jnp.einsum("bdl,bnl,bdl->bdln", delta_f, B.astype(jnp.float32), u_f)

    # Associative scan over the L axis with the diagonal linear-recurrence
    # binary operator: (a1, b1) ° (a2, b2) = (a2·a1, a2·b1 + b2).
    # leaves of shape (B, D, L, N); scan along axis=2.
    def scan_op(q_i, q_j):
        a_i, b_i = q_i
        a_j, b_j = q_j
        return a_j * a_i, a_j * b_i + b_j

    # Returns the cumulative (A_prod, x_state); we only need x.
    _, x = jax.lax.associative_scan(scan_op, (delta_A, delta_Bu), axis=2)

    # y[b, d, t] = Σ_n C[b, n, t] · x[b, d, t, n]
    y = jnp.einsum("bdln,bnl->bdl", x, C.astype(jnp.float32))
    # + D skip
    y = y + D_skip.astype(jnp.float32)[None, :, None] * u_f
    return y.astype(u.dtype)


# ---------------------------------------------------------------------------
# Mamba block
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Root-mean-square layer norm (Mamba uses this, not LayerNorm)."""

    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        scale = self.param("scale", nn.initializers.ones, (dim,))
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return scale * (x / rms)


def _dt_bias_init(dt_min: float, dt_max: float, dt_init_floor: float):
    """Inverse-softplus initializer for dt bias.

    Mirrors the PyTorch reference:

        dt = exp(uniform(log(dt_min), log(dt_max))).clamp(min=dt_init_floor)
        bias = dt + log(-expm1(-dt))   # = softplus^{-1}(dt)
    """
    def init(key, shape, dtype=jnp.float32):
        u = jax.random.uniform(key, shape, dtype=jnp.float32)
        dt = jnp.exp(u * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = jnp.maximum(dt, dt_init_floor)
        # softplus^{-1}(dt) = dt + log(-expm1(-dt))
        inv_dt = dt + jnp.log(-jnp.expm1(-dt))
        return inv_dt.astype(dtype)
    return init


def _A_log_init(d_inner: int, d_state: int):
    """A_log = log(arange(1, d_state+1)) tiled across d_inner."""
    def init(key, shape, dtype=jnp.float32):
        del key
        a = jnp.arange(1, d_state + 1, dtype=jnp.float32)
        A = jnp.tile(a[None, :], (d_inner, 1))  # (d_inner, d_state)
        return jnp.log(A).astype(dtype)
    return init


class MambaBlock(nn.Module):
    """A single Mamba block (the inner layer; outer stack is in :class:`MambaLM`)."""

    config: MambaConfig

    @nn.compact
    def __call__(self, x):  # x: (B, L, D_model)
        cfg = self.config
        d_inner = cfg.d_model * cfg.expand
        dt_rank = cfg.resolved_dt_rank()

        # in_proj: D_model → 2·d_inner  (split into x, z gate).
        xz = nn.Dense(
            2 * d_inner, use_bias=cfg.bias, dtype=cfg.dtype, name="in_proj",
        )(x)  # (B, L, 2·d_inner)
        # Split last dim into x_main, z_gate.
        x_main, z = jnp.split(xz, 2, axis=-1)  # each (B, L, d_inner)

        # Depthwise causal 1D conv along time (axis=-2 of x_main).
        # Flax Conv expects (B, L, channels) layout.
        x_main = nn.Conv(
            features=d_inner,
            kernel_size=(cfg.d_conv,),
            feature_group_count=d_inner,  # depthwise
            use_bias=cfg.conv_bias,
            padding=[(cfg.d_conv - 1, 0)],  # left-pad only → causal
            dtype=cfg.dtype,
            name="conv1d",
        )(x_main)  # still (B, L, d_inner)
        x_main = jax.nn.silu(x_main)

        # x_proj: d_inner → (dt_rank + 2·d_state)
        x_dbl = nn.Dense(
            dt_rank + 2 * cfg.d_state, use_bias=False, dtype=cfg.dtype,
            name="x_proj",
        )(x_main)  # (B, L, dt_rank + 2·d_state)
        dt, B_in, C_in = jnp.split(
            x_dbl, [dt_rank, dt_rank + cfg.d_state], axis=-1,
        )
        # dt: (B, L, dt_rank) → project up to d_inner; B/C: (B, L, d_state).

        # dt_proj: dt_rank → d_inner. Bias is the special inv-softplus init.
        # IMPORTANT: PyTorch reference uses *symmetric* uniform_(-std, +std);
        # ``nn.initializers.uniform(scale)`` is asymmetric ``[0, scale)`` and
        # would bias dt projection positive at init.
        dt_init_std = dt_rank ** -0.5

        def _symmetric_uniform_init(key, shape, dtype=jnp.float32):
            return jax.random.uniform(
                key, shape, dtype=dtype,
                minval=-dt_init_std, maxval=dt_init_std,
            )

        dt = nn.Dense(
            d_inner, use_bias=False, dtype=cfg.dtype,
            kernel_init=_symmetric_uniform_init,
            name="dt_proj",
        )(dt)  # (B, L, d_inner)
        dt_bias = self.param(
            "dt_bias", _dt_bias_init(cfg.dt_min, cfg.dt_max, cfg.dt_init_floor),
            (d_inner,),
        )
        # The reference applies softplus inside selective_scan with delta_bias
        # passed separately; we keep the same convention.

        # A_log: (d_inner, d_state) — kept in fp32.
        A_log = self.param(
            "A_log", _A_log_init(d_inner, cfg.d_state), (d_inner, cfg.d_state),
        )
        A = -jnp.exp(A_log.astype(jnp.float32))  # negative real eigenvalues

        # D skip connection (d_inner,) — fp32.
        D_skip = self.param(
            "D_skip", nn.initializers.ones, (d_inner,), jnp.float32,
        )

        # selective_scan expects (B, D, L) for u/delta and (B, N, L) for B/C.
        # We have (B, L, D) and (B, L, N); transpose.
        u_bdl = jnp.transpose(x_main, (0, 2, 1))   # (B, d_inner, L)
        dt_bdl = jnp.transpose(dt, (0, 2, 1))     # (B, d_inner, L)
        B_bnl = jnp.transpose(B_in, (0, 2, 1))     # (B, d_state, L)
        C_bnl = jnp.transpose(C_in, (0, 2, 1))     # (B, d_state, L)

        y_bdl = selective_scan(
            u_bdl, dt_bdl, A, B_bnl, C_bnl, D_skip, dt_bias, delta_softplus=True,
        )  # (B, d_inner, L)
        y = jnp.transpose(y_bdl, (0, 2, 1))  # (B, L, d_inner)

        # Gate by SiLU(z) and project back to D_model.
        y = y * jax.nn.silu(z)
        out = nn.Dense(
            cfg.d_model, use_bias=cfg.bias, dtype=cfg.dtype, name="out_proj",
        )(y)
        return out


class MambaLM(nn.Module):
    """Token-embed → N × (RMSNorm + MambaBlock + residual) → norm → logits."""

    config: MambaConfig

    @nn.compact
    def __call__(self, inputs):  # (B, T) int
        cfg = self.config
        assert inputs.ndim == 2, f"inputs must be (B, T); got {inputs.shape}"
        T = inputs.shape[1]
        if T > cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {cfg.max_len}")

        # Shift-right with BOS at index 0 (same scheme as gpt2.py).
        from c2a.ar_codegen.models.gpt2 import _shift_right
        y_ids = _shift_right(inputs.astype(jnp.int32), axis=1)

        embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.d_model,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="embed",
        )
        x = embed(y_ids)  # (B, T, d_model)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)

        for layer_idx in range(cfg.n_layers):
            normed = _RMSNorm(name=f"norm_{layer_idx}")(x)
            x = x + MambaBlock(config=cfg, name=f"block_{layer_idx}")(normed)

        x = _RMSNorm(name="final_norm")(x)
        logits = nn.Dense(
            cfg.vocab_size, dtype=cfg.dtype, name="logits", use_bias=False,
        )(x)
        return logits.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Loss + training (shares the loss helper with gpt2 to avoid duplication)
# ---------------------------------------------------------------------------


def _cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    return jnp.mean(-jnp.sum(one_hot * log_probs, axis=-1))


class _TrainState(train_state.TrainState):
    pass


def _create_train_state(rng, config: MambaConfig, lr: float, wd: float, n_frames: int):
    model = MambaLM(config=config.replace(deterministic=False))
    dummy = jnp.zeros((1, n_frames), dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    params = model.init({"params": params_rng, "dropout": dropout_rng}, dummy)["params"]
    tx = optax.adamw(learning_rate=lr, weight_decay=wd)
    return _TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _train_step(state: _TrainState, batch: jnp.ndarray, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, batch, rngs={"dropout": dropout_rng},
        )
        return _cross_entropy_loss(logits, batch)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, grad_norm


def _eval_step(params, batch: jnp.ndarray, config: MambaConfig):
    model = MambaLM(config=config.replace(deterministic=True))
    logits = model.apply({"params": params}, batch)
    return _cross_entropy_loss(logits, batch)


def _predict_step(params, batch: jnp.ndarray, config: MambaConfig):
    """Teacher-forced argmax predictions (deterministic)."""
    model = MambaLM(config=config.replace(deterministic=True))
    logits = model.apply({"params": params}, batch)
    return jnp.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class MambaGenerator(CodeGenerator):
    """Selective SSM LM over per-frame KPMS codes."""

    MODEL_TYPE = "mamba"

    def __init__(self, num_codes: int, config: MambaConfig | None = None):
        super().__init__(num_codes=num_codes)
        if config is None:
            config = MambaConfig(vocab_size=num_codes + 1)
        elif config.vocab_size != num_codes + 1:
            raise ValueError(
                f"MambaConfig.vocab_size ({config.vocab_size}) must equal "
                f"num_codes+1 ({num_codes + 1})."
            )
        self.config = config
        self._params = None
        self._n_frames: int | None = None

    def fit(
        self,
        train_codes: np.ndarray,
        val_codes: np.ndarray,
        cfg: Any,
        seed: int = 0,
        wandb_run: Any = None,
    ) -> None:
        from c2a.ar_codegen.data import make_batches

        batch_size = int(cfg.get("batch_size", 16))
        num_epochs = int(cfg.get("num_epochs", 200))
        learning_rate = float(cfg.get("learning_rate", 3e-4))
        weight_decay = float(cfg.get("weight_decay", 0.01))
        log_every = int(cfg.get("log_every", 1))
        seed = int(cfg.get("seed", seed))
        early_stop_patience = int(cfg.get("early_stop_patience", 0))
        early_stop_min_delta = float(cfg.get("early_stop_min_delta", 1e-4))
        histogram_log_every = int(cfg.get("histogram_log_every", 0))

        n_train, T = train_codes.shape
        if T > self.config.max_len:
            raise ValueError(
                f"train_codes length {T} > max_len {self.config.max_len}."
            )
        self._n_frames = T

        train_tok = (train_codes.astype(np.int32) + 1).astype(np.int32)
        val_tok = (val_codes.astype(np.int32) + 1).astype(np.int32) if val_codes.size > 0 else val_codes

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        state = _create_train_state(init_rng, self.config, learning_rate, weight_decay, T)

        jit_train = jax.jit(_train_step)
        jit_eval = jax.jit(lambda p, b: _eval_step(p, b, self.config))
        jit_predict = jax.jit(lambda p, b: _predict_step(p, b, self.config))

        np_rng = np.random.default_rng(seed)
        global_step = 0

        actual_train_hist = code_histogram(train_codes, self.num_codes)
        actual_val_hist = (
            code_histogram(val_codes, self.num_codes)
            if val_codes.size > 0 else None
        )

        best_val_loss = float("inf")
        best_params = None
        best_epoch = -1
        patience_counter = 0

        log.info(
            "Mamba fit: n_train=%d, T=%d, vocab=%d, d_model=%d, n_layers=%d, "
            "d_state=%d, batch_size=%d, epochs=%d, lr=%g, "
            "early_stop_patience=%d, histogram_log_every=%d",
            n_train, T, self.config.vocab_size, self.config.d_model,
            self.config.n_layers, self.config.d_state,
            batch_size, num_epochs, learning_rate,
            early_stop_patience, histogram_log_every,
        )

        for epoch in range(num_epochs):
            batches = make_batches(train_tok, batch_size, np_rng, shuffle=True)
            epoch_losses = []
            for batch in batches:
                rng, drop_rng = jax.random.split(rng)
                state, loss, gn = jit_train(state, jnp.asarray(batch), drop_rng)
                epoch_losses.append(float(loss))
                if global_step % log_every == 0:
                    log_metrics(
                        wandb_run, global_step,
                        {
                            "train_loss": float(loss),
                            "train_perplexity": float(jnp.exp(loss)),
                            "grad_norm": float(gn),
                            "lr": learning_rate,
                        }, group="train",
                    )
                global_step += 1

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            val_loss = None
            if val_tok.size > 0:
                val_batches = make_batches(val_tok, batch_size, np_rng, shuffle=False)
                if not val_batches:
                    val_loss = float(jit_eval(state.params, jnp.asarray(val_tok)))
                else:
                    val_loss = float(np.mean(
                        [float(jit_eval(state.params, jnp.asarray(vb))) for vb in val_batches]
                    ))
                log_metrics(
                    wandb_run, global_step,
                    {
                        "val_loss": val_loss,
                        "val_perplexity": float(np.exp(val_loss)),
                        "best_val_loss": min(best_val_loss, val_loss),
                    },
                    group="loss",
                )

            if histogram_log_every > 0 and (epoch + 1) % histogram_log_every == 0:
                _log_code_dist(
                    wandb_run, global_step, state.params, self.config,
                    train_tok, val_tok, batch_size,
                    actual_train_hist, actual_val_hist, self.num_codes,
                    jit_predict,
                )

            log.info(
                "  epoch %3d/%d  train_loss=%.4f%s",
                epoch + 1, num_epochs, mean_loss,
                f"  val_loss={val_loss:.4f}" if val_loss is not None else "",
            )

            if val_loss is not None and val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                best_params = jax.tree.map(lambda x: x, state.params)
                best_epoch = epoch + 1
                patience_counter = 0
            elif val_loss is not None:
                patience_counter += 1
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    log.info(
                        "  early stop @ epoch %d: best=%.4f @ epoch %d.",
                        epoch + 1, best_val_loss, best_epoch,
                    )
                    break

        if early_stop_patience > 0 and best_params is not None:
            log.info(
                "Restoring best Mamba params from epoch %d (val_loss=%.4f).",
                best_epoch, best_val_loss,
            )
            state = state.replace(params=best_params)

        _log_code_dist(
            wandb_run, global_step, state.params, self.config,
            train_tok, val_tok, batch_size,
            actual_train_hist, actual_val_hist, self.num_codes,
            jit_predict,
        )

        self._params = state.params
        self.meta = GeneratorMeta(
            model_type=self.MODEL_TYPE,
            num_codes=self.num_codes,
            code_source=str(cfg.get("_code_source", "unknown")),
            n_frames_per_clip=T,
            config_snapshot={
                "d_model": self.config.d_model,
                "n_layers": self.config.n_layers,
                "d_state": self.config.d_state,
                "d_conv": self.config.d_conv,
                "expand": self.config.expand,
                "dt_rank": self.config.dt_rank if isinstance(self.config.dt_rank, int) else str(self.config.dt_rank),
                "max_len": self.config.max_len,
                "dropout_rate": self.config.dropout_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
            },
        )

    def sample(
        self,
        n_seq: int,
        length: int,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("MambaGenerator.sample called before fit/load.")
        if length > self.config.max_len:
            raise ValueError(
                f"length={length} > max_len={self.config.max_len}."
            )
        model = MambaLM(config=self.config.replace(deterministic=True))

        # Same O(L²) prefix-based sampling as gpt2.py.
        rng = jax.random.PRNGKey(int(seed))
        seqs = np.zeros((n_seq, length), dtype=np.int32)
        for i in range(n_seq):
            tokens = np.zeros((1, length), dtype=np.int32)
            for t in range(length):
                logits = model.apply({"params": self._params}, jnp.asarray(tokens))
                step_logits = logits[0, t] / max(float(temperature), 1e-6)
                step_logits = step_logits.at[BOS_ID].set(-1e9)
                rng, sub = jax.random.split(rng)
                tok = int(jax.random.categorical(sub, step_logits))
                tokens[0, t] = tok
            seqs[i] = tokens[0] - 1
            if (seqs[i] < 0).any() or (seqs[i] >= self.num_codes).any():
                raise RuntimeError(
                    f"Mamba sampled OOB code at seq {i}: min={seqs[i].min()}, "
                    f"max={seqs[i].max()}."
                )
        return seqs

    def _save_model_files(self, path: Path) -> None:
        from flax import serialization
        if self._params is None:
            raise RuntimeError("MambaGenerator.save before fit/load.")
        (path / PARAMS_FILENAME).write_bytes(serialization.to_bytes(self._params))

    @classmethod
    def _load_model_files(cls, path: Path, meta: GeneratorMeta) -> "MambaGenerator":
        from flax import serialization
        snap = meta.config_snapshot
        config = MambaConfig(
            vocab_size=int(meta.num_codes) + 1,
            d_model=int(snap.get("d_model", 128)),
            n_layers=int(snap.get("n_layers", 4)),
            d_state=int(snap.get("d_state", 16)),
            d_conv=int(snap.get("d_conv", 4)),
            expand=int(snap.get("expand", 2)),
            dt_rank=snap.get("dt_rank", "auto") if isinstance(snap.get("dt_rank", "auto"), int) else str(snap.get("dt_rank", "auto")),
            max_len=int(snap.get("max_len", 256)),
            dropout_rate=float(snap.get("dropout_rate", 0.1)),
        )
        gen = cls(num_codes=int(meta.num_codes), config=config)
        rng = jax.random.PRNGKey(0)
        model = MambaLM(config=config.replace(deterministic=True))
        dummy = jnp.zeros((1, meta.n_frames_per_clip), dtype=jnp.int32)
        fresh = model.init({"params": rng, "dropout": rng}, dummy)["params"]
        gen._params = serialization.from_bytes(
            fresh, (path / PARAMS_FILENAME).read_bytes(),
        )
        gen._n_frames = int(meta.n_frames_per_clip)
        return gen
