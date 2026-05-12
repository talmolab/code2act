"""Single shMoSeq fit on a batched syllable sequence.

Wraps ``state_moseq.hhmm_efficient`` so callers don't have to manage
PRNG keys and Gibbs-loop bookkeeping themselves.  Mirror of
:func:`c2a.codegen.kpms.fit_kpms.fit_kpms_keypoints` for the hierarchical layer.

**CRITICAL**: imports from ``state_moseq`` set ``jax_enable_x64 = True``
process-globally (consistent with the upstream ``jax_moseq`` convention).
This module must run in a separate process from the RL training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    import state_moseq as sm
    from state_moseq.hhmm_efficient import (
        initialize_params,
        resample_states,
        resample_params,
    )
    HAS_SHIMOSEQ = True
except ImportError as _e:
    HAS_SHIMOSEQ = False

from c2a.codegen.shimoseq.config import ShimoseqHyperparams, hyperparams_to_dict

log = logging.getLogger(__name__)


@dataclass
class ShimoseqFitResult:
    """Result of a single shMoSeq fit.

    Attributes:
        states: final ``(B, L)`` per-batch high-level state assignments,
            int32, values in ``[0, n_states)``.
        params: final fitted parameters dict (transition + emission).
        marginal_loglik: scalar marginal log-likelihood at the final
            iteration.  Used for ``trans_kappa`` selection per the
            shMoSeq paper recipe (log-prob peak within the duration
            window).
        loss_trace: array of per-iteration losses recorded during
            fitting.  Useful as a "did this fit converge" diagnostic.
        states_dict: ``{clip_id: 1D state sequence}`` after unbatching;
            mirrors the structure of the input ``syllables_dict``.
        durations: 1D array of state durations (in frames) collected
            across all bouts in ``states_dict``.  Used for the
            duration-target selection rule.
        usage: ``(n_states,)`` array of per-state frame counts (after
            unbatching, masked).
        active_states: number of states with usage > 1% of total
            (the paper's effective-state count).
    """

    states: np.ndarray
    params: dict
    marginal_loglik: float
    loss_trace: np.ndarray
    states_dict: dict[str, np.ndarray]
    durations: np.ndarray
    usage: np.ndarray
    active_states: int


def _runs_lengths(arr: np.ndarray) -> np.ndarray:
    """Return contiguous-run lengths of ``arr`` (a 1D integer sequence)."""
    if arr.size == 0:
        return np.array([], dtype=np.int32)
    changes = np.where(np.diff(arr) != 0)[0] + 1
    runs = np.split(arr, changes)
    return np.array([len(r) for r in runs], dtype=np.int32)


def fit_shimoseq(
    seqs: np.ndarray,
    mask: np.ndarray,
    metadata: Any,
    hyperparams: ShimoseqHyperparams,
    seed: int,
    n_syllables: int | None = None,
    parallel_fitting: bool = True,
    verbose: bool = False,
) -> ShimoseqFitResult:
    """Fit a single shMoSeq HHMM by Gibbs sampling.

    Args:
        seqs: ``(B, L)`` int — batched low-level syllable sequences from
            :func:`c2a.codegen.shimoseq.syllable_loader.batch_for_shimoseq`.
        mask: ``(B, L)`` bool — valid-timepoint mask from the same call.
        metadata: opaque metadata returned by ``state_moseq.batch``;
            needed to ``unbatch`` final states back to per-clip arrays.
        hyperparams: see :class:`ShimoseqHyperparams`.
        seed: random seed for JAX PRNG.
        parallel_fitting: forwards to ``resample_states``; True is the
            tutorial default.
        verbose: log per-iteration marginal log-likelihood.

    Returns:
        :class:`ShimoseqFitResult`.
    """
    if not HAS_SHIMOSEQ:
        raise ImportError(
            "state_moseq not installed.  Install with `pip install state-moseq`."
        )

    if n_syllables is None:
        n_syllables = int(np.asarray(seqs).max()) + 1

    hypparams = hyperparams_to_dict(hyperparams)
    # state_moseq.hhmm_efficient also requires `n_syllables` and `random_seed`
    # as top-level keys; add them here.
    hypparams["n_syllables"] = int(n_syllables)
    hypparams["random_seed"] = int(seed)

    data = {
        "syllables": jnp.asarray(seqs, dtype=jnp.int32),
        "mask": jnp.asarray(mask, dtype=jnp.bool_),
    }

    key = jr.PRNGKey(seed)
    init_key, key = jr.split(key)

    # Initial random states + fitted params from the data.
    states = jr.randint(
        init_key, data["syllables"].shape, 0, hypparams["n_states"],
    )
    params = initialize_params(data, hypparams)

    losses_per_iter: list[float] = []
    final_marginal_ll = float("nan")

    for it in range(hypparams["num_iters"]):
        sub1, sub2, key = jr.split(key, 3)
        params, _losses = resample_params(sub1, data, states, hypparams, params)
        # state_moseq.hhmm_efficient.resample_states accepts ``parallel``,
        # not ``parallel_fitting`` (the tutorial example used the latter
        # keyword in narrative; the actual API name is shorter).
        states, marginal_ll = resample_states(
            sub2, data, params, parallel=parallel_fitting,
        )
        try:
            losses_per_iter.append(float(np.asarray(_losses).mean()))
        except Exception:
            losses_per_iter.append(float("nan"))
        final_marginal_ll = float(np.asarray(marginal_ll))
        if verbose and (it % 10 == 0 or it == hypparams["num_iters"] - 1):
            log.info(
                f"  iter {it:>3d}: marginal_ll={final_marginal_ll:.2f}, "
                f"loss={losses_per_iter[-1]:.4f}"
            )

    states_np = np.asarray(states, dtype=np.int32)
    states_dict = sm.unbatch(states_np, *metadata)

    # Aggregate diagnostics across the unbatched per-clip arrays.
    all_durations: list[np.ndarray] = []
    n_states = int(hypparams["n_states"])
    usage = np.zeros(n_states, dtype=np.int64)
    for arr in states_dict.values():
        a = np.asarray(arr, dtype=np.int32).reshape(-1)
        all_durations.append(_runs_lengths(a))
        np.add.at(usage, a, 1)
    durations = np.concatenate(all_durations) if all_durations else np.array([])
    active_states = int(np.sum(usage > 0.01 * usage.sum()))

    return ShimoseqFitResult(
        states=states_np,
        params=params,
        marginal_loglik=final_marginal_ll,
        loss_trace=np.asarray(losses_per_iter, dtype=np.float64),
        states_dict={k: np.asarray(v, dtype=np.int32) for k, v in states_dict.items()},
        durations=durations,
        usage=usage,
        active_states=active_states,
    )
