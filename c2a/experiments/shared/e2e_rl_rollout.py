"""Real RL rollouts for an e2e moseq-decoder checkpoint.

Two rollout modes:

* **Live**  — normal env rollout. The e2e policy reads
  ``obs["motion_window_stack"]`` (supplied by the env from the GT clip's
  reference future), runs it through the e2e VQ-VAE encoder + bottleneck,
  and uses ``z_q = codebook[live_code]`` for the action decoder. The codes
  the policy actually used at each step are recorded via the
  ``live_code_idx`` field of the logging inference fn's ``extras``.

* **Fixed** — same env rollout, but obs is augmented with
  ``code_override`` (a per-frame code-stack pulled from the pretrained
  tokenizer's saved encoding of this clip). The policy bypasses the
  encoder/bottleneck entirely and uses ``codebook[code_override]`` for
  ``z_q``. The action decoder + GRU + VAE-head + IntentionDecoder still
  run, the MuJoCo step still happens — the only thing that's different
  is which code the policy was told to act on.

Both rollouts return per-frame ``qpos``, so the divergence between live
and fixed shows up as a real physical trajectory difference, not a
kinematic guess.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


def run_e2e_rollout(
    env: Any,
    inference_fn: Callable,
    params: Any,
    ppo_networks: Any,
    key: jax.Array,
    *,
    max_steps: int,
    code_override: np.ndarray | None = None,
    reset_clip_idx: int | None = None,
    jit_reset: Callable | None = None,
    jit_step: Callable | None = None,
) -> dict[str, np.ndarray]:
    """Run one e2e-policy rollout, optionally forcing a fixed code stream.

    Args:
        env: pre-built ``MoSeqImitation`` env (see
            :func:`c2a.experiments.run_labeled_hidden_dynamics_controlled._build_pose_env`).
        inference_fn: jit-compiled logging policy fn from
            :func:`c2a.experiments.shared.checkpoint_utils.make_inference_fn`.
        params: ``(normalizer_state, policy_params)``.
        ppo_networks: returned by the e2e network factory; only
            ``init_hidden`` is used here.
        key: PRNG key for env reset + action sampling.
        max_steps: rollout length (frames).
        code_override: optional ``(T,)`` int32 array of pretrained-tokenizer
            code ids. When provided, each step the policy receives an
            ``obs["code_override"]`` stack ``code_override[t:t+stack]``
            (last index clamped at ``T-1``), and the policy bypasses the
            e2e encoder/bottleneck — see ``MoSeqRecurrentDecoderNetwork._e2e_forward``.
        reset_clip_idx: forwarded to ``env.reset(clip_idx=...)``.
        jit_reset / jit_step: cached jitted env reset/step. Created on
            the fly when ``None``.

    Returns:
        dict with keys:
            - ``qpos``: ``(T, nq)`` float64
            - ``live_code_idx``: ``(T,)`` int32  — current code (= code_override
              when override mode, = live encoder code otherwise)
            - ``code_idx``: ``(T,)`` int32  — env-supplied kpms_code[0] (the
              pretrained-tokenizer code carried by the env, useful as a
              baseline reference even in live mode)
            - ``rewards``: ``(T,)`` float32
            - ``survival``: number of steps before ``state.done`` first fired
              (= ``max_steps`` if it never did).
    """
    if jit_reset is None:
        jit_reset = jax.jit(env.reset)
    if jit_step is None:
        jit_step = jax.jit(env.step)

    key, reset_key = jax.random.split(key)
    if reset_clip_idx is not None:
        state = env.reset(reset_key, clip_idx=jnp.int32(reset_clip_idx))
    else:
        state = jit_reset(reset_key)

    hidden = ppo_networks.policy_network.init_hidden(1)
    code_stack_size = int(state.obs["kpms_code"].shape[-1])

    qpos_list: list[np.ndarray] = []
    live_code_list: list[int] = []
    env_code_list: list[int] = []
    reward_list: list[float] = []
    survival = max_steps

    for t in range(max_steps):
        if code_override is not None:
            stacked = []
            for si in range(code_stack_size):
                idx = min(t + si, int(code_override.shape[0]) - 1)
                stacked.append(int(code_override[idx]))
            new_obs = OrderedDict(state.obs)
            new_obs["code_override"] = jnp.array(stacked, dtype=jnp.int32)
            state = state.replace(obs=new_obs)

        key, subkey = jax.random.split(key)
        batched_obs = jax.tree.map(lambda x: x[None], state.obs)
        action, extras, hidden = inference_fn(
            params, batched_obs, hidden, subkey,
        )
        action = jax.tree.map(lambda x: x[0], action)
        extras = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x,
            extras,
        )

        qpos_list.append(np.array(state.data.qpos))
        reward_list.append(float(state.reward))
        if "live_code_idx" in extras:
            live_code_list.append(int(extras["live_code_idx"]))
        elif "code_idx" in extras:
            live_code_list.append(int(extras["code_idx"]))
        env_code_list.append(int(extras.get("code_idx", -1)))

        state = jit_step(state, action)
        if state.done and survival == max_steps:
            survival = t + 1

    return {
        "qpos": np.asarray(qpos_list),
        "live_code_idx": np.asarray(live_code_list, dtype=np.int32),
        "code_idx": np.asarray(env_code_list, dtype=np.int32),
        "rewards": np.asarray(reward_list, dtype=np.float32),
        "survival": int(survival),
    }
