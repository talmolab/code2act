"""Invariants for decoder-only (code-driven) rollouts.

Every experiment that drives the policy *purely from the discrete code stream*
— holding / scripting codes for attractor, basin, and hidden-dynamics analyses —
MUST satisfy two invariants. This module centralizes their enforcement so all
such experiments share one setup and one source of truth.

Invariant 1 — deterministic actions
    ``deterministic=True`` so the action is the policy distribution's **mode**,
    not a sample. Sampled actions inject per-step exploration noise that
    contaminates any dynamical-systems readout (Floquet, perturbation→return,
    PCA trajectories). Two legacy population scripts historically used
    ``deterministic=False`` + ``sample_no_postprocessing`` — that is a bug for
    dynamics analysis, not a feature.

Invariant 2 — the frozen MIMIC encoder cannot influence the policy
    The continuous encoder (``IntentionEncoder`` → ``z_e``) must contribute
    nothing to the action or the GRU recurrence. Depending on the checkpoint's
    architecture flags this holds for *different reasons*:

      * ``use_continuous_encoder=False``  → encoder is never instantiated/run.
      * ``use_distillation_head=True``     → ``z_e`` is a pure KL target;
        ``z_e_arg=None`` and ``decoder_input`` excludes ``z_e`` (see
        ``MoSeqRecurrentDecoderNetwork.__call__``). It never enters the action
        path regardless of ``z_e_scale``.
      * otherwise                          → ``z_e_scaled = z_e * z_e_scale``
        enters the GRU input (legacy) or the action head; only ``z_e_scale=0``
        severs it.

    Because the mechanism differs per checkpoint — and ``load_moseq_checkpoint``
    may even auto-flip ``use_continuous_encoder`` for distill ckpts — a single
    flag check is unreliable. We instead verify the **behavioural** invariant:
    the encoder's only private input is ``obs["task_obs"]`` (``code_emb`` comes
    from ``kpms_code``; ``proprioception`` is a legitimate decoder input;
    ``task_obs`` feeds *only* the encoder, via ``_encode``). So perturbing
    ``task_obs`` and asserting the action + next hidden state are bit-identical
    is a definitive, mode-agnostic test that nothing from the encoder reaches
    the policy.

Usage in an experiment ``main``::

    from c2a.experiments.shared.decoder_only import (
        assert_decoder_only_settings, describe_decoder_only_mode,
        verify_encoder_severed,
    )

    assert_decoder_only_settings(z_e_scale, deterministic=True)
    log.info(describe_decoder_only_mode(ckpt_cfg.network_config))
    inf_fn = make_inference_fn(ppo_networks, use_rnn=True,
                               deterministic=True, z_e_scale=z_e_scale)
    # once, on any real (batched) obs from env.reset:
    verify_encoder_severed(inf_fn, params, batched_obs, hidden, key)
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import jax
import numpy as np

log = logging.getLogger(__name__)

# The encoder's sole private observation key (see
# MoSeqRecurrentDecoderNetwork._encode: imitation_target = obs["task_obs"]).
ENCODER_PRIVATE_OBS_KEY = "task_obs"


def assert_decoder_only_settings(
    z_e_scale: float, deterministic: bool
) -> None:
    """Static guard for the two decoder-only invariants.

    Raises:
        ValueError: if ``deterministic`` is not True or ``z_e_scale`` is not
            exactly 0.0. (``z_e_scale=0`` is the lever for the non-distillation
            paths; for ``use_continuous_encoder=False`` / distillation ckpts it
            is a harmless no-op, so requiring it costs nothing and keeps every
            decoder-only experiment on one identical setting.)
    """
    if not bool(deterministic):
        raise ValueError(
            "decoder-only rollouts require deterministic=True (actions = "
            "distribution mode); got deterministic=False. Sampled actions add "
            "per-step noise that contaminates dynamics analysis."
        )
    if float(z_e_scale) != 0.0:
        raise ValueError(
            f"decoder-only rollouts require z_e_scale=0.0; got {z_e_scale}. "
            "Any nonzero value lets the frozen MIMIC encoder drive the policy."
        )


def describe_decoder_only_mode(network_config: Any) -> str:
    """Return a human-readable reason the encoder is (or isn't) severed.

    Purely informational — for logging at experiment startup so the run record
    states *why* the rollout is decoder-only for this particular checkpoint.
    """
    g = network_config.get
    use_enc = bool(g("use_continuous_encoder", False))
    use_distill = bool(g("use_distillation_head", False))
    use_pre = bool(g("use_pretrained_decoder", False))
    z_at_head = bool(g("z_e_at_action_head", False))
    if not use_enc:
        return ("decoder-only mechanism: use_continuous_encoder=False — encoder "
                "never instantiated; z_e_scale is a no-op.")
    if use_distill:
        return ("decoder-only mechanism: use_distillation_head=True — encoder is "
                "a frozen KL target only (z_e_arg=None); z_e never enters the "
                "action path. Pretrained decoder=%s." % use_pre)
    where = "action head" if z_at_head else "GRU input"
    return (f"decoder-only mechanism: encoder active; z_e enters the {where} "
            f"scaled by z_e_scale — MUST be 0.0 to sever it. "
            f"use_pretrained_decoder={use_pre}.")


def verify_encoder_severed(
    inference_fn: Callable,
    params: Any,
    obs: dict,
    hidden: Any,
    key: jax.Array,
    *,
    atol: float = 0.0,
    n_probes: int = 2,
    base_seed: int = 0,
) -> dict:
    """Behaviourally verify the encoder cannot influence the policy.

    Runs ``inference_fn`` on ``obs`` unperturbed, then on copies whose
    ``task_obs`` leaf is replaced with large random values, and asserts the
    action and every hidden-state array are identical (within ``atol``; default
    exact). This is a closed test of invariant 2 that holds regardless of which
    architecture mechanism provides decoder-only behaviour.

    Args:
        inference_fn: RNN inference fn ``(params, obs, hidden, key) ->
            (action, extras, new_hidden)`` (as built by ``make_inference_fn``).
        params: Inference params, e.g. ``(normalizer_state, policy_params)``.
        obs: A single example obs dict, batched exactly as ``inference_fn``
            expects it during rollout (leading batch dim already present).
        hidden: GRU hidden state matching the obs batch.
        key: A PRNG key (the *same* key is reused across perturbations, so any
            difference is attributable to ``task_obs`` alone).
        atol: Allowed absolute difference. 0.0 = require bit-identical (correct
            when z_e is zeroed: ``0 * W = 0`` contributes nothing).
        n_probes: Number of independent random ``task_obs`` perturbations.
        base_seed: Seed for the perturbation RNG (kept fixed for reproducibility).

    Returns:
        A report dict with ``severed`` (bool, always True if it returns),
        ``mechanism`` (str), and ``max_abs_diff`` (float).

    Raises:
        RuntimeError: if any perturbation changes the action or hidden state —
            i.e. the encoder DOES influence the policy (decoder-only is broken).
    """
    if ENCODER_PRIVATE_OBS_KEY not in obs:
        log.info(
            "verify_encoder_severed: no %r key in obs — encoder has no private "
            "input wired (use_continuous_encoder=False). Trivially severed.",
            ENCODER_PRIVATE_OBS_KEY,
        )
        return {"severed": True, "mechanism": "no task_obs key",
                "max_abs_diff": 0.0}

    action0, _, hidden0 = inference_fn(params, obs, hidden, key)
    action0 = np.asarray(action0)
    hidden0 = [np.asarray(h) for h in jax.tree_util.tree_leaves(hidden0)]

    task_obs = np.asarray(obs[ENCODER_PRIVATE_OBS_KEY])
    rng = np.random.RandomState(base_seed)
    max_diff = 0.0
    for p in range(n_probes):
        # Large, structurally-valid garbage in the encoder's input only.
        perturbed = (rng.randn(*task_obs.shape) * 1e3).astype(task_obs.dtype)
        probe_obs = dict(obs)
        probe_obs[ENCODER_PRIVATE_OBS_KEY] = jax.numpy.asarray(perturbed)

        action1, _, hidden1 = inference_fn(params, probe_obs, hidden, key)
        action1 = np.asarray(action1)
        hidden1 = [np.asarray(h) for h in jax.tree_util.tree_leaves(hidden1)]

        a_diff = float(np.max(np.abs(action0 - action1))) if action0.size else 0.0
        h_diff = max(
            (float(np.max(np.abs(a - b))) for a, b in zip(hidden0, hidden1)),
            default=0.0,
        )
        max_diff = max(max_diff, a_diff, h_diff)
        if a_diff > atol or h_diff > atol:
            raise RuntimeError(
                "verify_encoder_severed FAILED: perturbing task_obs changed the "
                f"policy output (probe {p}: |Δaction|={a_diff:.3e}, "
                f"|Δhidden|={h_diff:.3e}, atol={atol}). The frozen MIMIC encoder "
                "IS influencing the policy — this rollout is NOT decoder-only. "
                "Check z_e_scale (must be 0.0) and the checkpoint's encoder "
                "architecture flags."
            )

    log.info(
        "verify_encoder_severed OK: task_obs perturbations left action+hidden "
        "unchanged (max|Δ|=%.2e over %d probes). Encoder confirmed severed.",
        max_diff, n_probes,
    )
    return {"severed": True, "mechanism": "behavioural-invariance",
            "max_abs_diff": max_diff}


def verify_encoder_severed_from_env(
    inference_fn: Callable,
    params: Any,
    ppo_networks: Any,
    env: Any,
    key: jax.Array,
    *,
    clip_idx: int = 0,
    **kwargs,
) -> dict:
    """Convenience wrapper: build a real batched obs from ``env.reset`` and run
    :func:`verify_encoder_severed`.

    Constructs the obs exactly as a rollout does (``env.reset`` then add a
    leading batch dim) and the matching ``init_hidden(1)``, so an experiment can
    verify the decoder-only invariant in one line right after building its
    inference fn. Extra kwargs are forwarded to :func:`verify_encoder_severed`.
    """
    import jax.numpy as jnp

    state = env.reset(key, clip_idx=jnp.int32(clip_idx))
    obs = jax.tree_util.tree_map(lambda x: x[None], state.obs)
    hidden = ppo_networks.policy_network.init_hidden(1)
    return verify_encoder_severed(inference_fn, params, obs, hidden, key, **kwargs)
