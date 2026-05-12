"""S5 — diagonal linear SSM with parallel scan, vendored from lindermanlab/S5.

The core ``S5SSM`` and its HiPPO initialization live in :mod:`s5_ssm`
and :mod:`s5_ssm_init` (vendored verbatim with only an import-path
patch and a deprecated-typing fix — see the headers of those files).

This module wraps ``S5SSM`` into a small autoregressive language model
over per-frame KPMS codes, mirroring the GPT-2 / Mamba layout (token
embedding → BOS-shift → N × (norm + S5 layer + residual) → final norm
→ logits). The vendored ``S5SSM.__call__`` operates on a single
``(L, H)`` sequence; we ``jax.vmap`` over the batch axis.

Defaults sized for the c2a data scale:

* ``H = 128``         (model dim / features)
* ``P = 64``          (state size)
* ``n_layers = 4``
* ``conj_sym = True`` (exploits conjugate-symmetric eigenvalues)
* ``clip_eigs = True``  (LH-plane constraint; recommended for AR use)
* ``bidirectional = False``  (causal — required for AR sampling)

→ ~600 k params.
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

from c2a.ar_codegen.models.base import CodeGenerator, GeneratorMeta
from c2a.ar_codegen.models.s5_ssm import S5SSM
from c2a.ar_codegen.models.s5_ssm_init import make_DPLR_HiPPO
from c2a.ar_codegen.wandb_utils import _log_code_dist, code_histogram, log_metrics

log = logging.getLogger(__name__)


BOS_ID = 0
PARAMS_FILENAME = "params.msgpack"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@struct.dataclass
class S5Config:
    """Hyperparameters for the S5 stack."""

    vocab_size: int  # = num_codes + 1
    H: int = 128
    P: int = 64
    n_layers: int = 4
    C_init: str = "lecun_normal"
    discretization: str = "zoh"
    dt_min: float = 0.001
    dt_max: float = 0.1
    conj_sym: bool = True
    clip_eigs: bool = True
    bidirectional: bool = False  # MUST be False for AR sampling
    dropout_rate: float = 0.1
    max_len: int = 256
    deterministic: bool = False
    dtype: Any = jnp.float32


# ---------------------------------------------------------------------------
# Initialization of the (Λ_re, Λ_im, V, V^{-1}) tuple via HiPPO-LegS-DPLR
# ---------------------------------------------------------------------------


def _make_s5_init(P: int, conj_sym: bool):
    """Build (Lambda_re, Lambda_im, V, Vinv) for the S5 SSM constructor.

    Mirrors what ``lindermanlab/S5/s5/seq_model.py`` does at network
    construction time. We compute it on host (numpy) since it's a small
    fixed init.
    """
    # make_DPLR_HiPPO works in jnp; cast to numpy for parameter init values.
    Lambda, _, _, V, _ = make_DPLR_HiPPO(P)
    Vinv = V.conj().T
    if conj_sym:
        # Use the first P/2 eigenvalues (conjugate-symmetric).
        if P % 2 != 0:
            raise ValueError(f"P must be even when conj_sym=True; got {P}.")
        Lambda = Lambda[: P // 2]
        V = V[:, : P // 2]
        Vinv = Vinv[: P // 2, :]
    return np.asarray(Lambda.real), np.asarray(Lambda.imag), np.asarray(V), np.asarray(Vinv)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _S5Layer(nn.Module):
    """One S5 residual block: pre-norm → S5SSM → GLU → residual.

    Matches the residual structure used in lindermanlab/S5's
    ``seq_model.SequenceLayer`` (LayerNorm + SSM + GeLU + GLU-like
    gating). We use a simpler post-SSM nonlinearity here (just GeLU) —
    the AR data is small and we don't need every bell and whistle.
    """

    config: S5Config
    Lambda_re_init: Any
    Lambda_im_init: Any
    V: Any
    Vinv: Any

    @nn.compact
    def __call__(self, x):  # x: (B, L, H)
        cfg = self.config
        normed = nn.LayerNorm(name="norm")(x)

        # S5SSM accepts (L, H); vmap over batch.
        ssm = S5SSM(
            Lambda_re_init=self.Lambda_re_init,
            Lambda_im_init=self.Lambda_im_init,
            V=self.V,
            Vinv=self.Vinv,
            H=cfg.H,
            P=cfg.P // 2 if cfg.conj_sym else cfg.P,
            C_init=cfg.C_init,
            discretization=cfg.discretization,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            conj_sym=cfg.conj_sym,
            clip_eigs=cfg.clip_eigs,
            bidirectional=cfg.bidirectional,
            name="ssm",
        )
        # Apply over the batch axis. We unroll inside the module so vmap
        # can introspect parameters via Flax's standard vmap.
        ssm_out = nn.vmap(
            lambda mdl, seq: mdl(seq),
            variable_axes={"params": None},
            split_rngs={"params": False, "dropout": True},
            in_axes=0, out_axes=0,
        )(ssm, normed)
        ssm_out = jax.nn.gelu(ssm_out)
        ssm_out = nn.Dense(cfg.H, name="proj")(ssm_out)
        ssm_out = nn.Dropout(rate=cfg.dropout_rate)(
            ssm_out, deterministic=cfg.deterministic,
        )
        return x + ssm_out


class S5LM(nn.Module):
    """Token embed → BOS-shift → N × S5Layer → norm → logits.

    The S5 HiPPO init arrays (``Lambda_re_init``, ``Lambda_im_init``,
    ``V``, ``Vinv``) are passed as Module fields rather than computed in
    ``setup``. Computing them inside ``setup`` ran them under JAX trace
    on every ``apply`` call, which broke when ``make_DPLR_HiPPO``'s
    ``jnp.arange(N)`` was reached with a traced ``N``. Precomputing
    once at generator-construction time avoids the trace entirely
    (same pattern lindermanlab/S5 uses via ``init_S5SSM``).
    """

    config: S5Config
    Lambda_re_init: Any
    Lambda_im_init: Any
    V: Any
    Vinv: Any

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        assert inputs.ndim == 2, f"inputs must be (B, T); got {inputs.shape}"
        T = inputs.shape[1]
        if T > cfg.max_len:
            raise ValueError(f"sequence length {T} > max_len {cfg.max_len}")

        # Reuse gpt2's shift_right (BOS-pad convention).
        from c2a.ar_codegen.models.gpt2 import _shift_right
        y_ids = _shift_right(inputs.astype(jnp.int32), axis=1)

        x = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.H,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="embed",
        )(y_ids)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=cfg.deterministic)

        for layer_idx in range(cfg.n_layers):
            x = _S5Layer(
                config=cfg,
                Lambda_re_init=self.Lambda_re_init,
                Lambda_im_init=self.Lambda_im_init,
                V=self.V,
                Vinv=self.Vinv,
                name=f"layer_{layer_idx}",
            )(x)

        x = nn.LayerNorm(name="final_norm")(x)
        logits = nn.Dense(cfg.vocab_size, name="logits", use_bias=False)(x)
        return logits.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Train / eval (same shape as gpt2.py)
# ---------------------------------------------------------------------------


def _cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    return jnp.mean(-jnp.sum(one_hot * log_probs, axis=-1))


class _TrainState(train_state.TrainState):
    pass


def _build_model(config: S5Config, init_arrays):
    """Construct an S5LM with the precomputed HiPPO init arrays.

    ``init_arrays`` is ``(Lambda_re, Lambda_im, V, Vinv)`` — host-side
    numpy arrays produced by :func:`_make_s5_init`.
    """
    lr_re, lr_im, V, Vinv = init_arrays
    return S5LM(
        config=config, Lambda_re_init=lr_re, Lambda_im_init=lr_im,
        V=V, Vinv=Vinv,
    )


def _create_train_state(rng, config: S5Config, init_arrays, lr: float, wd: float, n_frames: int):
    model = _build_model(config.replace(deterministic=False), init_arrays)
    dummy = jnp.zeros((1, n_frames), dtype=jnp.int32)
    params_rng, dropout_rng = jax.random.split(rng)
    params = model.init({"params": params_rng, "dropout": dropout_rng}, dummy)["params"]
    tx = optax.adamw(learning_rate=lr, weight_decay=wd)
    return _TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _train_step(state: _TrainState, batch, dropout_rng):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params}, batch, rngs={"dropout": dropout_rng},
        )
        return _cross_entropy_loss(logits, batch)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    gn = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, loss, gn


def _eval_step(params, batch, config: S5Config, init_arrays):
    model = _build_model(config.replace(deterministic=True), init_arrays)
    logits = model.apply({"params": params}, batch)
    return _cross_entropy_loss(logits, batch)


def _predict_step(params, batch, config: S5Config, init_arrays):
    """Teacher-forced argmax predictions (deterministic)."""
    model = _build_model(config.replace(deterministic=True), init_arrays)
    logits = model.apply({"params": params}, batch)
    return jnp.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class S5Generator(CodeGenerator):
    """Linderman-lab S5 SSM as an autoregressive code-sequence prior."""

    MODEL_TYPE = "s5"

    def __init__(self, num_codes: int, config: S5Config | None = None):
        super().__init__(num_codes=num_codes)
        if config is None:
            config = S5Config(vocab_size=num_codes + 1)
        elif config.vocab_size != num_codes + 1:
            raise ValueError(
                f"S5Config.vocab_size ({config.vocab_size}) must equal "
                f"num_codes+1 ({num_codes + 1})."
            )
        if config.bidirectional:
            raise ValueError(
                "S5 must be unidirectional for autoregressive sampling "
                "(set bidirectional=False)."
            )
        self.config = config
        # Precompute the HiPPO init *once*, host-side — never under JIT.
        self._init_arrays = _make_s5_init(config.P, config.conj_sym)
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
        state = _create_train_state(
            init_rng, self.config, self._init_arrays, learning_rate, weight_decay, T,
        )

        jit_train = jax.jit(_train_step)
        jit_eval = jax.jit(lambda p, b: _eval_step(p, b, self.config, self._init_arrays))
        jit_predict = jax.jit(
            lambda p, b: _predict_step(p, b, self.config, self._init_arrays)
        )

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
            "S5 fit: n_train=%d, T=%d, vocab=%d, H=%d, P=%d, n_layers=%d, "
            "batch_size=%d, epochs=%d, lr=%g, "
            "early_stop_patience=%d, histogram_log_every=%d",
            n_train, T, self.config.vocab_size, self.config.H, self.config.P,
            self.config.n_layers, batch_size, num_epochs, learning_rate,
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
                "Restoring best S5 params from epoch %d (val_loss=%.4f).",
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
                "H": self.config.H,
                "P": self.config.P,
                "n_layers": self.config.n_layers,
                "C_init": self.config.C_init,
                "discretization": self.config.discretization,
                "dt_min": self.config.dt_min,
                "dt_max": self.config.dt_max,
                "conj_sym": self.config.conj_sym,
                "clip_eigs": self.config.clip_eigs,
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
            raise RuntimeError("S5Generator.sample called before fit/load.")
        if length > self.config.max_len:
            raise ValueError(f"length={length} > max_len={self.config.max_len}.")
        model = _build_model(self.config.replace(deterministic=True), self._init_arrays)

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
                    f"S5 sampled OOB code at seq {i}: min={seqs[i].min()}, "
                    f"max={seqs[i].max()}."
                )
        return seqs

    def _save_model_files(self, path: Path) -> None:
        from flax import serialization
        if self._params is None:
            raise RuntimeError("S5Generator.save before fit/load.")
        (path / PARAMS_FILENAME).write_bytes(serialization.to_bytes(self._params))

    @classmethod
    def _load_model_files(cls, path: Path, meta: GeneratorMeta) -> "S5Generator":
        from flax import serialization
        snap = meta.config_snapshot
        config = S5Config(
            vocab_size=int(meta.num_codes) + 1,
            H=int(snap.get("H", 128)),
            P=int(snap.get("P", 64)),
            n_layers=int(snap.get("n_layers", 4)),
            C_init=str(snap.get("C_init", "lecun_normal")),
            discretization=str(snap.get("discretization", "zoh")),
            dt_min=float(snap.get("dt_min", 0.001)),
            dt_max=float(snap.get("dt_max", 0.1)),
            conj_sym=bool(snap.get("conj_sym", True)),
            clip_eigs=bool(snap.get("clip_eigs", True)),
            max_len=int(snap.get("max_len", 256)),
            dropout_rate=float(snap.get("dropout_rate", 0.1)),
        )
        gen = cls(num_codes=int(meta.num_codes), config=config)
        rng = jax.random.PRNGKey(0)
        model = _build_model(config.replace(deterministic=True), gen._init_arrays)
        dummy = jnp.zeros((1, meta.n_frames_per_clip), dtype=jnp.int32)
        params_rng, dropout_rng = jax.random.split(rng)
        fresh = model.init({"params": params_rng, "dropout": dropout_rng}, dummy)["params"]
        gen._params = serialization.from_bytes(
            fresh, (path / PARAMS_FILENAME).read_bytes(),
        )
        gen._n_frames = int(meta.n_frames_per_clip)
        return gen
