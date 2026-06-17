"""Pluggable code priors for Option-B transfer.

A ``PriorLogitProvider`` maps ``(prev_code, state) -> (prior_logits, new_state)``
each decision step, where ``prior_logits`` is an unnormalized distribution over
the ``num_codes`` codes. The wrapper injects these logits into the policy obs and
the policy-gated additive rule combines them:
``final_logits = sigmoid(beta) * prior_logits + delta``.

Providers (staged, see ``plan.md`` "Prior menu"):

* :class:`TransitionMatrixPrior` — per-frame empirical TM ``p(c'|c)``. Stateless
  beyond ``prev_code``; trivial to thread; the de-risk default for walk_rear.
* (later) a Mamba provider — per-frame or syllable-level neural grammar; carries a
  recurrent SSM state. Same interface, swapped by config.

All providers are **frozen** (no trainable params): the prior is a fixed bias the
policy gates against, not something PPO updates.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class PriorLogitProvider(abc.ABC):
    """Maps ``(prev_code, state)`` to ``prior_logits`` over the codebook.

    ``state`` is the provider's per-env recurrent carry (empty for a TM, an SSM
    state for a Mamba). It is threaded through the env ``State.info`` and
    done-reset by the wrapper via :meth:`init_state`.
    """

    num_codes: int

    @abc.abstractmethod
    def init_state(self, batch_size: int) -> Any:
        """Initial provider state for ``batch_size`` envs (carry at reset/done)."""

    @abc.abstractmethod
    def logits(
        self, prev_code: jnp.ndarray, state: Any
    ) -> tuple[jnp.ndarray, Any]:
        """Return ``(prior_logits[..., num_codes], new_state)``.

        Args:
            prev_code: Int code(s) chosen last step, ``-1`` at first step / after
                ``done`` (caller resets). Shape ``[...]``.
            state: The provider's carry (see :meth:`init_state`).
        """


@dataclass
class TransitionMatrixPrior(PriorLogitProvider):
    """Per-frame empirical transition-matrix prior ``p(c' | c)``.

    ``log_tm[c]`` are the log-probabilities of the next code given current code
    ``c``; for the first step (``prev_code < 0``) we use the log stationary
    (marginal) distribution. Stateless: ``state`` is an unused placeholder so the
    wrapper carry pytree stays uniform across provider types.

    Attributes:
        log_tm: ``[num_codes, num_codes]`` log transition matrix (rows = current
            code, cols = next code), Laplace-smoothed.
        log_marginal: ``[num_codes]`` log marginal code distribution (first step).
        num_codes: Codebook size.
    """

    log_tm: jnp.ndarray
    log_marginal: jnp.ndarray
    num_codes: int

    def init_state(self, batch_size: int) -> jnp.ndarray:
        # Placeholder carry (no recurrence); kept for interface uniformity.
        return jnp.zeros((batch_size, 0), dtype=jnp.float32)

    def logits(
        self, prev_code: jnp.ndarray, state: Any
    ) -> tuple[jnp.ndarray, Any]:
        prev = jnp.asarray(prev_code, dtype=jnp.int32)
        safe = jnp.clip(prev, 0, self.num_codes - 1)
        row = self.log_tm[safe]  # [..., num_codes]
        first = (prev < 0)[..., None]
        out = jnp.where(first, self.log_marginal, row)
        return out, state


def transition_matrix_prior_from_codes(
    codes: np.ndarray,
    num_codes: int,
    eps: float = 1e-3,
) -> TransitionMatrixPrior:
    """Build a :class:`TransitionMatrixPrior` from code sequences.

    Args:
        codes: ``[N, T]`` int code sequences (e.g. train split of the VAE codes).
        num_codes: Codebook size.
        eps: Laplace smoothing added to every transition + marginal count.

    Returns:
        A :class:`TransitionMatrixPrior`.
    """
    codes = np.asarray(codes)
    counts = np.full((num_codes, num_codes), eps, dtype=np.float64)
    marg = np.full((num_codes,), eps, dtype=np.float64)
    for seq in codes:
        seq = seq[(seq >= 0) & (seq < num_codes)]
        if seq.size == 0:
            continue
        np.add.at(marg, seq, 1.0)
        if seq.size > 1:
            np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    tm = counts / counts.sum(axis=1, keepdims=True)
    marg = marg / marg.sum()
    return TransitionMatrixPrior(
        log_tm=jnp.asarray(np.log(tm), dtype=jnp.float32),
        log_marginal=jnp.asarray(np.log(marg), dtype=jnp.float32),
        num_codes=int(num_codes),
    )


@dataclass
class MambaPrior(PriorLogitProvider):
    """Per-frame neural prior ``p(c_t | c_{<t})`` from the trained Mamba.

    Runs the Mamba **recurrent** step (``c2a.transfer.mamba_recurrent``) one token
    per decision step, carrying the per-layer SSM + conv state. Returns
    log-probabilities over the ``num_codes`` codes (the BOS slot is dropped). The
    first step (``prev_code < 0``) feeds BOS.

    Attributes:
        params: Trained Mamba param pytree.
        config: ``MambaConfig`` matching ``params``.
        num_codes: Codebook size (= ``config.vocab_size - 1``).
    """

    params: Any
    config: Any
    num_codes: int

    def init_state(self, batch_size: int) -> Any:
        from c2a.transfer.mamba_recurrent import init_recurrent_state

        one = init_recurrent_state(self.config)
        return jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (batch_size, *x.shape)), one
        )

    def logits(self, prev_code: jnp.ndarray, state: Any) -> tuple[jnp.ndarray, Any]:
        from c2a.transfer.mamba_recurrent import mamba_step

        prev = jnp.asarray(prev_code, dtype=jnp.int32)
        token = jnp.where(prev < 0, jnp.int32(0), prev + 1)  # BOS=0 at first step

        def _one(tok, st):
            lg, new_st = mamba_step(self.params, self.config, st, tok)
            return jax.nn.log_softmax(lg[1:]), new_st  # drop BOS -> [num_codes]

        out, new_state = jax.vmap(_one)(token, state)
        return out, new_state


def mamba_prior_from_checkpoint(checkpoint_dir: str | Path) -> MambaPrior:
    """Load a trained Mamba ar_codegen checkpoint as a :class:`MambaPrior`.

    Args:
        checkpoint_dir: ``outputs/ar_codegen/mamba/<run>`` directory.

    Returns:
        A :class:`MambaPrior`.
    """
    from c2a.ar_codegen.models.base import CodeGenerator

    gen = CodeGenerator.from_checkpoint(str(checkpoint_dir))
    # Loaded params are numpy; convert to jnp so traced (vmap) token indexing of
    # the embedding table works inside the recurrent step.
    params = jax.tree_util.tree_map(jnp.asarray, gen._params)
    return MambaPrior(params=params, config=gen.config, num_codes=gen.num_codes)


def transition_matrix_prior_from_payload(
    checkpoint_cfg: Any,
    split_path: str | Path,
) -> TransitionMatrixPrior:
    """Build the TM prior from the decoder checkpoint's own code source.

    Uses ``c2a.codegen.code_sources.load_codes`` so the prior is fit on the same
    tokenizer + train split the decoder was trained on (train codes only — no
    test leakage).

    Args:
        checkpoint_cfg: The decoder checkpoint config (has ``kpms_config`` +
            ``network_config.num_codes``).
        split_path: Path to ``rodent_balanced_splits.json``.

    Returns:
        A :class:`TransitionMatrixPrior` over the checkpoint's codebook.
    """
    import json

    from c2a.codegen.code_sources import load_codes

    with open(split_path) as f:
        splits = json.load(f)
    train_idx = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_idx = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    n_frames = int(checkpoint_cfg.env_config.clip_length)
    payload = load_codes(checkpoint_cfg, train_idx, test_idx, n_frames)
    return transition_matrix_prior_from_codes(
        payload.train_codes, payload.num_codes
    )
