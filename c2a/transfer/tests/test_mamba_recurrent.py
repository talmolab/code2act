"""Equivalence test: Mamba recurrent step == parallel forward.

The in-loop Mamba prior (``c2a.transfer.mamba_recurrent.mamba_step``) must
reproduce ``MambaLM.__call__`` token-by-token. The two paths differ only in the
parallel ``selective_scan``'s internal **float32** downcast (it hardcodes
``.astype(jnp.float32)``), so we assert *functional* equivalence — exact argmax
agreement and a tiny per-step KL — rather than bitwise logit equality.

Self-contained: builds a small random-init ``MambaLM`` (no checkpoint needed).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from c2a.ar_codegen.models.gpt2 import _shift_right
from c2a.ar_codegen.models.mamba import MambaConfig, MambaLM
from c2a.transfer.mamba_recurrent import init_recurrent_state, mamba_step


def _build(num_codes: int = 12, T: int = 32, seed: int = 0):
    cfg = MambaConfig(vocab_size=num_codes + 1, d_model=32, n_layers=2, d_state=8, max_len=64)
    model = MambaLM(config=cfg.replace(deterministic=True))
    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, T), dtype=jnp.int32)
    params = model.init({"params": rng, "dropout": rng}, dummy)["params"]
    inputs = jnp.asarray(
        np.random.default_rng(seed).integers(0, num_codes, size=(1, T)).astype(np.int32)
    )
    tokens = inputs + 1  # code+1 convention
    return cfg, params, model, tokens


def test_recurrent_matches_parallel_argmax_and_kl():
    num_codes, T = 12, 32
    cfg, params, model, tokens = _build(num_codes, T)
    logits_par = model.apply({"params": params}, tokens)[0]  # (T, vocab)

    y_ids = _shift_right(tokens.astype(jnp.int32), axis=1)[0]  # [BOS, tokens[:-1]]
    state = init_recurrent_state(cfg)
    max_kl = 0.0
    argmax_agree = 0
    for t in range(T):
        lg, state = mamba_step(params, cfg, state, y_ids[t])
        assert lg.shape == (cfg.vocab_size,)
        argmax_agree += int(jnp.argmax(lg) == jnp.argmax(logits_par[t]))
        p = jax.nn.softmax(logits_par[t])
        q = jax.nn.softmax(lg)
        max_kl = max(max_kl, float(jnp.sum(p * (jnp.log(p + 1e-12) - jnp.log(q + 1e-12)))))

    assert argmax_agree == T, f"argmax mismatch: {argmax_agree}/{T}"
    assert max_kl < 1e-3, f"per-step KL too high: {max_kl}"


def test_recurrent_state_threads_and_is_finite():
    num_codes, T = 12, 16
    cfg, params, _, tokens = _build(num_codes, T, seed=1)
    y_ids = _shift_right(tokens.astype(jnp.int32), axis=1)[0]
    state = init_recurrent_state(cfg)
    for t in range(T):
        lg, state = mamba_step(params, cfg, state, y_ids[t])
        assert bool(jnp.all(jnp.isfinite(lg)))
