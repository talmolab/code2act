"""Env wrapper that drives the frozen GRU decoder from a code-prior + policy.

``CodePriorDecoderWrapper`` wraps a bare ``vnl_playground`` task env (e.g.
``Rearing``). The high-level PPO policy emits a discrete **code** per decision
step; this wrapper:

1. holds the code for ``dwell`` control steps, driving the **frozen GRU decoder**
   each control step (``decode_step``) to produce torques and stepping the base
   env (matches the decoder's 50 Hz training cadence: dwell=2 at ctrl 100 Hz);
2. advances the **frozen Mamba prior** once per decision (consuming the chosen
   code) to produce ``prior_logits`` for the next step;
3. exposes a flat observation ``[proprio(277) | task_obs | prior_logits]`` — the
   policy reads ``prior_logits`` raw and slices proprio from the front;
4. threads the GRU hidden + Mamba state + rng through ``state.info`` and
   **zeros/​re-inits them on ``done``** (the auto-reset wrapper does NOT reset
   custom info fields with ``full_reset=False``).

Carries live under ``t_*`` keys to avoid colliding with the base env's info.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import flatten_util
from mujoco_playground._src import wrapper

from c2a.transfer.decoder_adapter import (
    FrozenDecoder,
    decode_step,
    extract_decoder_proprio,
)
from c2a.transfer.priors import PriorLogitProvider


class CodePriorDecoderWrapper(wrapper.Wrapper):
    """Discrete code-prior + frozen-decoder transfer wrapper.

    Args:
        env: bare task env (must return ``{state:{task_obs,proprioception},
            privileged_state:{...}}`` obs).
        frozen_decoder: loaded :class:`FrozenDecoder`.
        prior: a :class:`PriorLogitProvider` (TM or Mamba).
        dwell: control steps to hold each code (2 = match 50 Hz code rate).
    """

    def __init__(
        self,
        env,
        frozen_decoder: FrozenDecoder,
        prior: PriorLogitProvider,
        dwell: int = 2,
    ):
        self._fd = frozen_decoder
        self._prior = prior
        self._dwell = int(dwell)
        self._num_codes = int(frozen_decoder.num_codes)
        self._proprio_size = int(frozen_decoder.proprio_size)
        super().__init__(env)
        # BOS reset branch is a constant (pure fn of frozen prior params) —
        # precompute once instead of recomputing a full Mamba step every env-step.
        self._bos_logits, self._bos_state = self._prior_first()

    @property
    def action_size(self) -> int:
        return self._num_codes

    # -- obs assembly -------------------------------------------------------
    def _task_obs(self, base_obs) -> jnp.ndarray:
        t = base_obs["state"]["task_obs"] if "state" in base_obs else base_obs["task_obs"]
        return jnp.nan_to_num(flatten_util.ravel_pytree(t)[0])

    def _build_obs(self, base_obs, prior_logits: jnp.ndarray) -> jnp.ndarray:
        proprio = extract_decoder_proprio(base_obs)  # 277
        return jnp.concatenate([proprio, self._task_obs(base_obs), prior_logits], axis=-1)

    def _prior_first(self):
        """prior_logits for a fresh episode (feed BOS) + the post-BOS state."""
        st = self._prior.init_state(1)
        pl, st = self._prior.logits(jnp.array([-1], dtype=jnp.int32), st)
        return pl[0], st

    # -- env API ------------------------------------------------------------
    def reset(self, rng):
        rng, body_rng = jax.random.split(rng)
        state = self.env.reset(body_rng)
        hidden = self._fd.init_hidden(1)
        obs = self._build_obs(state.obs, self._bos_logits)
        info = dict(state.info)
        info["t_decoder_hidden"] = hidden
        info["t_mamba_state"] = self._bos_state
        info["t_rng"] = rng
        return state.replace(obs=obs, info=info)

    def step(self, state, action):
        code = jnp.clip(jnp.round(action).astype(jnp.int32), 0, self._num_codes - 1)
        hidden = state.info["t_decoder_hidden"]
        mamba_state = state.info["t_mamba_state"]
        rng = state.info["t_rng"]
        # proprio (277) is the front of our flat obs; same vector the decoder wants.
        proprio0 = state.obs[..., : self._proprio_size]
        # Rebuild the base env's NESTED obs so the dwell scan carry has a
        # consistent pytree (base step returns nested obs; our incoming obs is flat).
        base_state = state.replace(obs=self.env._get_obs(state.data, state.info))

        def control_step(carry, _):
            st, hid, key, proprio = carry
            key, sk = jax.random.split(key)
            torque, _extras, hid = decode_step(self._fd, code, proprio, hid, sk)
            st = self.env.step(st, torque)  # one base control step
            proprio_next = extract_decoder_proprio(st.obs)
            return (st, hid, key, proprio_next), st.reward

        (state2, hidden, rng, _proprio), rewards = jax.lax.scan(
            control_step, (base_state, hidden, rng, proprio0), None, length=self._dwell
        )
        reward = jnp.sum(rewards)
        done = state2.done

        # advance prior with the chosen code; reset branch (cached BOS) on
        # inner-env done. Episode-length TRUNCATION is handled by the outer
        # BraxAutoResetWrapper(full_reset=True), which re-inits all t_* carries.
        pl_cont, ms_cont = self._prior.logits(code[None], mamba_state)
        prior_logits = jnp.where(done, self._bos_logits, pl_cont[0])
        mamba_state = jax.tree_util.tree_map(
            lambda r, c: jnp.where(done, r, c), self._bos_state, ms_cont
        )
        hidden = [jnp.where(done, jnp.zeros_like(h), h) for h in hidden]

        obs = self._build_obs(state2.obs, prior_logits)
        info = dict(state2.info)
        info["t_decoder_hidden"] = hidden
        info["t_mamba_state"] = mamba_state
        info["t_rng"] = rng
        return state2.replace(obs=obs, reward=reward, done=done, info=info)
