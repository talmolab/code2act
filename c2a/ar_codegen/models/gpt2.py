"""GPT-2 XS — small decoder-only transformer over per-frame codes.

Adapted from ``google/flax/examples/lm1b/models.py`` (Apache-2.0). The
original lm1b ``TransformerLM`` treats vocabulary index 0 as
EOS/padding and uses a left-pad shift to make the model autoregressive.
Our codes can legitimately take value 0, so we reserve a separate BOS
index: ``vocab_size = num_codes + 1``, code ``c`` is fed to the model
as ``c + 1``, and the model never emits index 0 (its logit is masked
out at sample time).

Architecture defaults are sized for ~17 k training tokens
(~70 clips × 250 frames, vocab ≤ 32):

* ``emb_dim``      128
* ``num_layers``     4
* ``num_heads``      4
* ``mlp_dim``      512
* ``dropout``      0.1
* ``max_len``      256

→ ~700 k params, sufficient for the data scale without overfitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state
from jax import lax

from c2a.ar_codegen.models.base import CodeGenerator, GeneratorMeta
from c2a.ar_codegen.wandb_utils import (
    _log_code_dist, code_histogram, log_code_histograms_figure, log_metrics,
)

log = logging.getLogger(__name__)


BOS_ID = 0  # vocab index reserved for "start of sequence".

PARAMS_FILENAME = "params.msgpack"


# ---------------------------------------------------------------------------
# Architecture (vendored / adapted from flax lm1b/models.py, Apache-2.0)
# ---------------------------------------------------------------------------


@struct.dataclass
class TransformerConfig:
    """Top-level hyperparameters for the GPT-2 XS transformer LM."""

    vocab_size: int  # = num_codes + 1 (BOS at 0)
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    qkv_dim: int = 128
    mlp_dim: int = 512
    max_len: int = 256
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    dtype: Any = jnp.float32


def _sinusoidal_init(max_len: int, d_feature: int) -> jnp.ndarray:
    """1-D sinusoidal positional embedding table ``[1, max_len, d_feature]``.

    Replicates ``flax/examples/lm1b/models.py::sinusoidal_init`` but is a
    pure function (not a parameter initializer) since we don't need it
    to be a flax param.
    """
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(10000.0) / (d_feature // 2 - 1)
    div_term = np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    return jnp.asarray(pe[np.newaxis, :, :])


def _shift_right(x: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """Pad left by 1 and drop the last element along ``axis``.

    For ``[c0, c1, c2]`` this yields ``[0, c0, c1]``. Combined with our
    ``+1`` token-id scheme, the padded leading 0 is the BOS token.
    """
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
    return lax.dynamic_slice_in_dim(padded, 0, padded.shape[axis] - 1, axis)


class _MlpBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        x = nn.Dense(cfg.mlp_dim, dtype=cfg.dtype)(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        x = nn.Dense(inputs.shape[-1], dtype=cfg.dtype)(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        return x


class _DecoderBlock(nn.Module):
    """One pre-norm transformer decoder block (self-attn + MLP)."""

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, decoder_mask=None):
        cfg = self.config
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            qkv_features=cfg.qkv_dim,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.attention_dropout_rate,
            deterministic=cfg.deterministic,
        )(x, mask=decoder_mask)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)
        x = x + inputs

        z = nn.LayerNorm(dtype=cfg.dtype)(x)
        z = _MlpBlock(config=cfg)(z)
        return x + z


class TransformerLM(nn.Module):
    """Decoder-only LM. Input ``[B, T]`` int → output ``[B, T, V]`` logits."""

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):  # noqa: D401
        cfg = self.config
        assert inputs.ndim == 2, f"inputs must be [B, T], got {inputs.shape}"

        # Mask: lower-triangular only (no padding mask — sequences are dense
        # and equal-length in this codebase).
        decoder_mask = nn.make_causal_mask(inputs, dtype=cfg.dtype)

        # Shift right so position t sees only positions < t.
        y = _shift_right(inputs.astype(jnp.int32), axis=1)

        # Token embedding.
        embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )
        y = embed(y)

        # Positional embedding (fixed sinusoidal).
        pe = _sinusoidal_init(cfg.max_len, cfg.emb_dim)
        y = y + pe[:, : y.shape[1], :]
        y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=cfg.deterministic)
        y = y.astype(cfg.dtype)

        for layer_idx in range(cfg.num_layers):
            y = _DecoderBlock(config=cfg, name=f"decoder_block_{layer_idx}")(
                y, decoder_mask=decoder_mask,
            )
        y = nn.LayerNorm(dtype=cfg.dtype, name="final_norm")(y)
        logits = nn.Dense(cfg.vocab_size, dtype=cfg.dtype, name="logits")(y)
        return logits.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Loss + training step
# ---------------------------------------------------------------------------


def _cross_entropy_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean cross-entropy over a [B, T, V] logits / [B, T] targets pair.

    Targets are the +1-shifted code IDs in ``[1, num_codes]`` — the BOS
    index 0 never appears as a target, so we don't need to mask it
    explicitly. (Sampling masks BOS at decode time so the model never
    emits it.)
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    nll = -jnp.sum(one_hot * log_probs, axis=-1)  # [B, T]
    return jnp.mean(nll)


class _TrainState(train_state.TrainState):
    """Standard Flax TrainState, no extra fields needed."""


def _create_train_state(
    rng: jax.Array,
    config: TransformerConfig,
    learning_rate: float,
    weight_decay: float,
    n_frames: int,
) -> _TrainState:
    model = TransformerLM(config=config.replace(deterministic=False))
    dummy = jnp.zeros((1, n_frames), dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    params = model.init({"params": params_rng, "dropout": dropout_rng}, dummy)["params"]
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return _TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _train_step(state: _TrainState, batch: jnp.ndarray, dropout_rng: jax.Array):
    """One supervised cross-entropy training step."""

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch,
            rngs={"dropout": dropout_rng},
        )
        return _cross_entropy_loss(logits, batch)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, grad_norm


def _eval_step(params, batch: jnp.ndarray, config: TransformerConfig):
    """Forward-pass cross-entropy (deterministic) for validation."""
    model = TransformerLM(config=config.replace(deterministic=True))
    logits = model.apply({"params": params}, batch)
    return _cross_entropy_loss(logits, batch)


def _predict_step(params, batch: jnp.ndarray, config: TransformerConfig):
    """Teacher-forced argmax predictions (deterministic).

    Returns int32 of shape ``[B, T]`` in ``[0, vocab_size)`` — same
    +1-shifted token space as the input batch. The caller subtracts 1
    before histogramming so the result is in ``[0, num_codes)``.
    """
    model = TransformerLM(config=config.replace(deterministic=True))
    logits = model.apply({"params": params}, batch)
    return jnp.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class GPT2Generator(CodeGenerator):
    """Small GPT-2-style decoder-only LM over KPMS codes."""

    MODEL_TYPE = "gpt2"

    def __init__(self, num_codes: int, config: TransformerConfig | None = None):
        super().__init__(num_codes=num_codes)
        if config is None:
            config = TransformerConfig(vocab_size=num_codes + 1)
        elif config.vocab_size != num_codes + 1:
            raise ValueError(
                f"TransformerConfig.vocab_size ({config.vocab_size}) must equal "
                f"num_codes+1 ({num_codes + 1}) — index 0 is reserved for BOS."
            )
        self.config = config
        self._params = None  # populated by fit/load
        self._n_frames: int | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_codes: np.ndarray,
        val_codes: np.ndarray,
        cfg: Any,
        seed: int = 0,
        wandb_run: Any = None,
    ) -> None:
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
                f"train_codes length {T} > model max_len {self.config.max_len}; "
                "increase config.max_len."
            )
        self._n_frames = T

        # ``+1`` shifts the codes into [1, num_codes], leaving 0 as BOS.
        train_tok = (train_codes.astype(np.int32) + 1).astype(np.int32)
        val_tok = (val_codes.astype(np.int32) + 1).astype(np.int32) if val_codes.size > 0 else val_codes

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        state = _create_train_state(
            init_rng, self.config, learning_rate, weight_decay, n_frames=T,
        )

        jit_train_step = jax.jit(_train_step)
        jit_eval_step = jax.jit(lambda p, b: _eval_step(p, b, self.config))
        jit_predict_step = jax.jit(lambda p, b: _predict_step(p, b, self.config))

        np_rng = np.random.default_rng(seed)
        global_step = 0

        # Precompute per-split actual histograms (constants — never change).
        actual_train_hist = code_histogram(train_codes, self.num_codes)
        actual_val_hist = (
            code_histogram(val_codes, self.num_codes)
            if val_codes.size > 0 else None
        )

        # Early-stopping state.
        best_val_loss = float("inf")
        best_params = None
        best_epoch = -1
        patience_counter = 0

        log.info(
            "GPT-2 XS fit: n_train=%d, T=%d, vocab=%d, batch_size=%d, "
            "epochs=%d, lr=%g, early_stop_patience=%d, histogram_log_every=%d",
            n_train, T, self.config.vocab_size, batch_size, num_epochs,
            learning_rate, early_stop_patience, histogram_log_every,
        )

        stopped_early = False
        last_epoch_done = -1
        for epoch in range(num_epochs):
            from c2a.ar_codegen.data import make_batches
            batches = make_batches(train_tok, batch_size, np_rng, shuffle=True)
            epoch_losses = []
            for batch in batches:
                rng, drop_rng = jax.random.split(rng)
                state, loss, grad_norm = jit_train_step(
                    state, jnp.asarray(batch), drop_rng,
                )
                epoch_losses.append(float(loss))
                if global_step % log_every == 0:
                    log_metrics(
                        wandb_run, global_step,
                        {
                            "train_loss": float(loss),
                            "train_perplexity": float(jnp.exp(loss)),
                            "grad_norm": float(grad_norm),
                            "lr": learning_rate,
                        },
                        group="train",
                    )
                global_step += 1

            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

            # Validation (no dropout).
            val_loss = None
            if val_tok.size > 0:
                val_losses = []
                val_batches = make_batches(
                    val_tok, batch_size, np_rng, shuffle=False,
                )
                if not val_batches:
                    val_loss = float(jit_eval_step(state.params, jnp.asarray(val_tok)))
                else:
                    for vb in val_batches:
                        val_losses.append(float(jit_eval_step(state.params, jnp.asarray(vb))))
                    val_loss = float(np.mean(val_losses))
                log_metrics(
                    wandb_run, global_step,
                    {
                        "val_loss": val_loss,
                        "val_perplexity": float(np.exp(val_loss)),
                        "best_val_loss": min(best_val_loss, val_loss),
                    },
                    group="loss",
                )

            # Histogram logging (every N epochs).
            if histogram_log_every > 0 and (epoch + 1) % histogram_log_every == 0:
                _log_code_dist(
                    wandb_run, global_step, state.params, self.config,
                    train_tok, val_tok, batch_size,
                    actual_train_hist, actual_val_hist, self.num_codes,
                    jit_predict_step,
                )

            log.info(
                "  epoch %3d/%d  train_loss=%.4f%s",
                epoch + 1, num_epochs, mean_loss,
                f"  val_loss={val_loss:.4f}" if val_loss is not None else "",
            )

            # Early stopping bookkeeping.
            if val_loss is not None and val_loss < best_val_loss - early_stop_min_delta:
                best_val_loss = val_loss
                # Snapshot params (a copy — jax.tree.map identity gives us
                # a tree of the same leaves so subsequent in-place updates
                # don't affect the snapshot, since JAX arrays are immutable).
                best_params = jax.tree.map(lambda x: x, state.params)
                best_epoch = epoch + 1
                patience_counter = 0
            elif val_loss is not None:
                patience_counter += 1
                if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                    log.info(
                        "  early stop @ epoch %d: val_loss has not improved by "
                        "> %g for %d epochs (best=%.4f @ epoch %d).",
                        epoch + 1, early_stop_min_delta, early_stop_patience,
                        best_val_loss, best_epoch,
                    )
                    stopped_early = True
                    last_epoch_done = epoch + 1
                    break
            last_epoch_done = epoch + 1

        # Restore best params if early stopping is enabled.
        if early_stop_patience > 0 and best_params is not None:
            log.info(
                "Restoring best params from epoch %d (val_loss=%.4f).",
                best_epoch, best_val_loss,
            )
            state = state.replace(params=best_params)

        # Final histogram log (always, when wandb is on).
        _log_code_dist(
            wandb_run, global_step, state.params, self.config,
            train_tok, val_tok, batch_size,
            actual_train_hist, actual_val_hist, self.num_codes,
            jit_predict_step,
        )

        self._params = state.params
        self.meta = GeneratorMeta(
            model_type=self.MODEL_TYPE,
            num_codes=self.num_codes,
            code_source=str(cfg.get("_code_source", "unknown")),
            n_frames_per_clip=T,
            config_snapshot={
                "emb_dim": self.config.emb_dim,
                "num_heads": self.config.num_heads,
                "num_layers": self.config.num_layers,
                "qkv_dim": self.config.qkv_dim,
                "mlp_dim": self.config.mlp_dim,
                "max_len": self.config.max_len,
                "dropout_rate": self.config.dropout_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "seed": seed,
            },
        )

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(
        self,
        n_seq: int,
        length: int,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if self._params is None:
            raise RuntimeError("GPT2Generator.sample called before fit/load.")
        if length > self.config.max_len:
            raise ValueError(
                f"length={length} > max_len={self.config.max_len}; increase "
                "config.max_len at training time."
            )
        model = TransformerLM(config=self.config.replace(deterministic=True))

        # Simple sampling loop: we run the full-sequence forward pass on a
        # growing prefix and take the last-position logits. This is O(L²)
        # in length; for L=250 it's fine. A cached-kv "decode" path could
        # speed it up but adds complexity we don't need here.
        rng = jax.random.PRNGKey(int(seed))
        seqs = np.zeros((n_seq, length), dtype=np.int32)
        for i in range(n_seq):
            tokens = np.zeros((1, length), dtype=np.int32)
            for t in range(length):
                logits = model.apply(
                    {"params": self._params},
                    jnp.asarray(tokens),
                )  # [1, length, V]
                step_logits = logits[0, t] / max(float(temperature), 1e-6)
                # Mask the BOS token — the model should never emit it.
                step_logits = step_logits.at[BOS_ID].set(-1e9)
                rng, sub = jax.random.split(rng)
                tok = int(jax.random.categorical(sub, step_logits))
                tokens[0, t] = tok
            # Strip the +1 offset: model emits in [1, num_codes], we
            # return codes in [0, num_codes).
            seqs[i] = tokens[0] - 1
            # Guard: sampled value 0 (now -1 after shift) should never
            # happen because we masked BOS. Catch silent vocab bugs.
            if (seqs[i] < 0).any() or (seqs[i] >= self.num_codes).any():
                raise RuntimeError(
                    f"GPT-2 sampled an out-of-range code at seq {i}: "
                    f"min={seqs[i].min()}, max={seqs[i].max()}. Vocab "
                    "or BOS masking is wrong."
                )
        return seqs

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_model_files(self, path: Path) -> None:
        from flax import serialization

        if self._params is None:
            raise RuntimeError("GPT2Generator.save called before fit/load.")
        bytes_blob = serialization.to_bytes(self._params)
        (path / PARAMS_FILENAME).write_bytes(bytes_blob)

    @classmethod
    def _load_model_files(cls, path: Path, meta: GeneratorMeta) -> "GPT2Generator":
        from flax import serialization

        snap = meta.config_snapshot
        config = TransformerConfig(
            vocab_size=int(meta.num_codes) + 1,
            emb_dim=int(snap.get("emb_dim", 128)),
            num_heads=int(snap.get("num_heads", 4)),
            num_layers=int(snap.get("num_layers", 4)),
            qkv_dim=int(snap.get("qkv_dim", 128)),
            mlp_dim=int(snap.get("mlp_dim", 512)),
            max_len=int(snap.get("max_len", 256)),
            dropout_rate=float(snap.get("dropout_rate", 0.1)),
        )
        gen = cls(num_codes=int(meta.num_codes), config=config)

        # Build a fresh params pytree of the right shape, then deserialize
        # the saved bytes into it.
        rng = jax.random.PRNGKey(0)
        model = TransformerLM(config=config.replace(deterministic=True))
        dummy = jnp.zeros((1, meta.n_frames_per_clip), dtype=jnp.int32)
        fresh = model.init({"params": rng, "dropout": rng}, dummy)["params"]
        gen._params = serialization.from_bytes(
            fresh, (path / PARAMS_FILENAME).read_bytes(),
        )
        gen._n_frames = int(meta.n_frames_per_clip)
        return gen
