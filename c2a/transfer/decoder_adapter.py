"""Load and drive the frozen GRU decoder from an external code stream.

The Option-B transfer stack freezes a trained ``MoSeqRecurrentDecoderNetwork``
(checkpoint ``model_checkpoints/260603_180852_140475`` — ``kinematics_vae``, 25
codes, 4-layer GRU, distillation + pretrained-VAE-decoder) and drives it
decoder-only (``z_e_scale=0``, deterministic). The high-level task policy chooses
a code; this adapter turns ``(code, proprioception, hidden)`` into a 38-dim torque
action and the next GRU hidden state, with the frozen MIMIC encoder severed.

This module is **env-agnostic and prior-agnostic** — it only knows how to talk to
the frozen decoder. The task env wrapper (``wrappers.py``) and the prior
(``priors.py``) build on top of it.

Decoder obs schema (from the checkpoint's ``network_config.obs_sizes``):

    * ``proprioception`` — 277-d, **raw** (the policy applies the frozen
      normalizer internally). In a ``vnl_playground`` task env this is the full
      ``_get_proprioception`` vector, exposed as ``privileged_state.proprioception``
      (its ravel order matches the imitation env's ``state.proprioception`` that
      the decoder's normalizer was fit on — both come from
      ``RodentEnv._get_proprioception``).
    * ``task_obs`` — 640-d, **severed**: fed as zeros (encoder off; ``z_e=0``).
    * ``kpms_code`` — ``code_stack_size`` (=5) copies of the chosen code, float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from c2a.transfer import _compat  # noqa: F401  (record_scalar shim, before orbax)

import jax
import jax.numpy as jnp
from jax import flatten_util

from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    make_inference_fn,
)


@dataclass(frozen=True)
class FrozenDecoder:
    """A loaded, frozen decoder ready to be driven by a code stream.

    Attributes:
        cfg: Full checkpoint config (Hydra ``DictConfig``).
        params: ``(normalizer_state, policy_params)`` for the inference fn.
        inf_fn: ``fn(params, obs_batched, hidden, key) -> (action, extras,
            new_hidden)`` — JIT-compiled, decoder-only (``z_e_scale=0``,
            ``deterministic=True``).
        ppo_networks: The reconstructed ``MoSeqRecurrentPPONetworks`` (exposes
            ``policy_network.init_hidden``).
        num_codes: Codebook size (25).
        code_stack_size: Number of stacked codes the decoder expects (5).
        proprio_size: Flat proprioception dim the normalizer expects (277).
        task_obs_size: Flat (severed) task-obs dim the decoder expects (640).
        action_size: Torque action dim (38).
    """

    cfg: Any
    params: tuple[Any, Any]
    inf_fn: Callable
    ppo_networks: Any
    num_codes: int
    code_stack_size: int
    proprio_size: int
    task_obs_size: int
    action_size: int

    def init_hidden(self, batch_size: int) -> list[jnp.ndarray]:
        """Zero GRU hidden state for ``batch_size`` rollouts."""
        return self.ppo_networks.policy_network.init_hidden(batch_size)


def load_frozen_decoder(
    checkpoint_path: str | Path,
    step: int | None = None,
) -> FrozenDecoder:
    """Load a trained RNN decoder and build its decoder-only inference fn.

    Args:
        checkpoint_path: Directory with ``MoSeqPPONetwork_*`` folders. Resolved
            to an absolute path (orbax requires absolute paths).
        step: Checkpoint step to restore (``None`` = latest).

    Returns:
        A :class:`FrozenDecoder`.

    Raises:
        ValueError: if the checkpoint is not an RNN decoder (the transfer stack
            requires the recurrent decoder).
    """
    checkpoint_path = str(Path(checkpoint_path).resolve())
    cfg, normalizer_state, policy_params, ppo_networks = load_moseq_checkpoint(
        checkpoint_path, step=step
    )
    net = cfg.network_config
    if not bool(net.get("use_rnn_decoder", False)):
        raise ValueError(
            "load_frozen_decoder requires an RNN decoder checkpoint "
            f"(use_rnn_decoder=True); got use_rnn_decoder="
            f"{net.get('use_rnn_decoder')}."
        )

    inf_fn = make_inference_fn(
        ppo_networks, use_rnn=True, deterministic=True, z_e_scale=0.0
    )
    obs_sizes = net.obs_sizes
    return FrozenDecoder(
        cfg=cfg,
        params=(normalizer_state, policy_params),
        inf_fn=inf_fn,
        ppo_networks=ppo_networks,
        num_codes=int(net.num_codes),
        code_stack_size=int(net.code_stack_size),
        proprio_size=int(obs_sizes["proprioception"]),
        task_obs_size=int(obs_sizes["task_obs"]),
        action_size=int(net.get("action_size", 38)),
    )


def extract_decoder_proprio(task_env_obs: Any) -> jnp.ndarray:
    """Flatten a ``vnl_playground`` task-env obs into the decoder's proprio vector.

    The decoder consumes the full ``RodentEnv._get_proprioception`` vector
    (277-d). In the nested task-env obs that is ``privileged_state.proprioception``
    (the non-privileged ``state.proprioception`` is a reduced subset). The ravel
    order matches what the decoder's normalizer was fit on because both envs use
    the same base ``_get_proprioception``.

    Args:
        task_env_obs: The task env's observation (a Mapping with a
            ``privileged_state.proprioception`` entry).

    Returns:
        Flat proprioception array, shape ``[proprio_size]``.
    """
    proprio = task_env_obs["privileged_state"]["proprioception"]
    return jnp.nan_to_num(flatten_util.ravel_pytree(proprio)[0])


def build_decoder_obs(
    code: jnp.ndarray,
    proprio_flat: jnp.ndarray,
    frozen: FrozenDecoder,
) -> dict[str, jnp.ndarray]:
    """Assemble the decoder's obs dict for a single (unbatched) step.

    Args:
        code: Scalar int code in ``[0, num_codes)`` (the chosen code this step).
        proprio_flat: Flat proprioception, shape ``[proprio_size]``, **raw**.
        frozen: The loaded :class:`FrozenDecoder`.

    Returns:
        ``{"kpms_code", "proprioception", "task_obs"}`` for one step (no batch dim).
    """
    code = jnp.asarray(code)
    kpms_code = jnp.full(
        (frozen.code_stack_size,), code, dtype=jnp.float32
    )  # replicate across the stack (no causal lookahead at transfer time)
    return {
        "kpms_code": kpms_code,
        "proprioception": proprio_flat.astype(jnp.float32),
        "task_obs": jnp.zeros((frozen.task_obs_size,), dtype=jnp.float32),
    }


def decode_step(
    frozen: FrozenDecoder,
    code: jnp.ndarray,
    proprio_flat: jnp.ndarray,
    hidden: list[jnp.ndarray],
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, list[jnp.ndarray]]:
    """One decoder step: ``(code, proprio, hidden) -> (action, code_idx, hidden')``.

    Adds/removes a batch dim around the batched inference fn (the wrapper calls
    this per-env; brax vmaps the env).

    Args:
        frozen: The loaded :class:`FrozenDecoder`.
        code: Scalar int code in ``[0, num_codes)``.
        proprio_flat: Flat proprioception, shape ``[proprio_size]``, raw.
        hidden: GRU hidden state (list of ``[1, H]`` arrays).
        key: PRNG key (unused at ``deterministic=True``, threaded for API parity).

    Returns:
        ``(action, code_idx, new_hidden)`` — ``action`` shape ``[action_size]``.
    """
    obs = build_decoder_obs(code, proprio_flat, frozen)
    obs_b = jax.tree_util.tree_map(lambda x: x[None], obs)
    action, extras, new_hidden = frozen.inf_fn(frozen.params, obs_b, hidden, key)
    action = jax.tree_util.tree_map(lambda x: x[0], action)
    return action, extras, new_hidden
