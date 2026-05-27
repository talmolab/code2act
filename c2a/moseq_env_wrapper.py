"""Imitation environment subclass that includes pre-computed KPMS codes.

Subclasses ``Imitation`` to inject ``kpms_code`` directly in ``_get_obs``,
and flattens the v0.0.13 nested obs structure to a flat dict. This avoids
the pitfalls of an env *wrapper* approach where:

* ``BraxDomainRandomizationVmapWrapper`` bypasses wrappers via ``env.unwrapped``
* ``EpisodeWrapper.step``'s ``jax.lax.scan`` requires carry-in / carry-out to
  have identical pytree structure, but the base env's ``_get_obs`` would strip
  the extra key.

By producing the flat ``{"task_obs": ..., "proprioception": ..., "kpms_code": ...}``
inside ``_get_obs``, every wrapper (Vmap, DR, Episode, AutoReset) sees a consistent
obs structure at all times.

code2code reward mode
---------------------
When a :class:`c2a.code2code.C2CTokenizerAdapter` is supplied and
``c2c_reward_mode != "off"``, the env additionally:

* Maintains a rolling ``(window_len, 7 + n_joints)`` qpos buffer in
  ``state.info["motion_window"]`` so the encoder always sees a full
  window of recent rollout motion.
* On every step, encodes the buffer, computes the distance to the
  codebook entry for the target code, and adds the (or replaces with
  the) resulting reward term per ``c2c_reward_mode``:
    - ``additive``: ``r = track_weight * r_track + alpha * match``
    - ``replace``:  ``r = alpha * match + replace_alive_bonus``
* Logs ``c2c_*`` diagnostics in ``state.info`` for downstream WandB
  logging.

The adapter and its codebook are **frozen** — PPO never updates them.
This is the architectural property that makes code2code immune to the
private-code-language drift that killed e2e_mode. See
``scratch/2026-05-26-code2code-mode/`` for the full design.
"""

import collections
from typing import Any, Literal, Mapping, Optional

import jax
import jax.numpy as jp
import jax.numpy as jnp
from mujoco import mjx
from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.code2code.adapter_base import C2CTokenizerAdapter, DistanceForm


C2CRewardMode = Literal["off", "additive", "replace"]
C2CRewardForm = Literal["exp_neg_distance", "neg_distance"]


class MoSeqImitation(Imitation):
    """Imitation env whose observations include pre-computed KPMS syllable codes.

    Overrides ``_get_obs`` to:
    1. Inject ``kpms_code`` into the obs dict.
    2. Flatten nested obs values to 1D arrays.
    3. Strip the ``state`` hierarchy (v0.0.13 nesting).

    This produces a flat dict ``{"task_obs": 1D, "proprioception": 1D, "kpms_code": 1D}``
    at all times, avoiding wrapper-chain pitfalls.

    Attributes:
        _kpms_codes: Pre-computed code array ``[n_clips, n_frames]``.
        _code_stack_size: Number of consecutive codes to stack (1 = current only,
            5 = current + 4 future codes).
        _c2c_adapter: Frozen tokenizer adapter for the code2code reward
            (None to disable).
        _c2c_active: True iff a c2c reward is actively shaping reward.
    """

    def __init__(
        self,
        config: Any,
        clips: ReferenceClips,
        kpms_codes: jnp.ndarray,
        code_stack_size: int = 1,
        *,
        c2c_adapter: Optional[C2CTokenizerAdapter] = None,
        c2c_alpha: float = 0.5,
        c2c_track_weight: float = 1.0,
        c2c_temperature: float = 1.0,
        c2c_distance_form: DistanceForm = "cosine",
        c2c_reward_form: C2CRewardForm = "exp_neg_distance",
        c2c_reward_mode: C2CRewardMode = "off",
        c2c_replace_alive_bonus: float = 0.0,
    ):
        super().__init__(config=config, clips=clips)
        self._kpms_codes = jnp.asarray(kpms_codes, dtype=jnp.int32)
        self._code_stack_size = int(code_stack_size)

        if c2c_reward_mode not in ("off", "additive", "replace"):
            raise ValueError(
                f"c2c_reward_mode must be 'off', 'additive', or 'replace'; "
                f"got {c2c_reward_mode!r}"
            )
        if c2c_reward_form not in ("exp_neg_distance", "neg_distance"):
            raise ValueError(
                f"c2c_reward_form must be 'exp_neg_distance' or 'neg_distance'; "
                f"got {c2c_reward_form!r}"
            )
        if c2c_distance_form not in ("l2", "cosine"):
            raise ValueError(
                f"c2c_distance_form must be 'l2' or 'cosine'; "
                f"got {c2c_distance_form!r}"
            )

        self._c2c_adapter = c2c_adapter
        self._c2c_alpha = float(c2c_alpha)
        self._c2c_track_weight = float(c2c_track_weight)
        self._c2c_temperature = float(c2c_temperature)
        self._c2c_distance_form = c2c_distance_form
        self._c2c_reward_form = c2c_reward_form
        self._c2c_reward_mode = c2c_reward_mode
        self._c2c_replace_alive_bonus = float(c2c_replace_alive_bonus)

        # Master enable flag: the env shapes reward only if an adapter
        # is configured AND reward_mode is not "off". Both reset() and
        # step() check this; the pytree structure of state.info is
        # uniform across the env's lifetime (set at construction).
        self._c2c_active = (
            c2c_adapter is not None and c2c_reward_mode != "off"
        )

        if c2c_reward_mode != "off" and c2c_adapter is None:
            raise ValueError(
                f"c2c_reward_mode={c2c_reward_mode!r} requires a c2c_adapter; "
                f"got None."
            )

    # ------------------------------------------------------------------
    # Override _get_obs to inject kpms_code
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = super()._get_obs(data, info)
        frame = self._get_cur_frame(data, info)
        n_frames = self._kpms_codes.shape[1]
        clip_idx = info["reference_clip"]

        # Inject kpms_code (pretrained tokenizer codes)
        if self._code_stack_size <= 1:
            frame = jnp.clip(frame, 0, n_frames - 1)
            code = self._kpms_codes[clip_idx, frame]
            kpms_code = code[..., None].astype(jnp.float32)
        else:
            offsets = jnp.arange(self._code_stack_size)
            frames = jnp.clip(frame[..., None] + offsets, 0, n_frames - 1)
            codes = self._kpms_codes[clip_idx[..., None], frames]
            kpms_code = codes.astype(jnp.float32)

        # Flatten nested obs and strip "state" hierarchy:
        # v0.0.13 returns {"state": OrderedDict(task_obs=nested, proprioception=nested)}
        # We produce {"task_obs": 1D, "proprioception": 1D, "kpms_code": 1D}
        state_obs = obs["state"]
        flat = collections.OrderedDict()
        for k, v in state_obs.items():
            flat[k] = jnp.nan_to_num(jax.flatten_util.ravel_pytree(v)[0])
        flat["kpms_code"] = kpms_code

        return flat

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            jax.flatten_util.ravel_pytree(obs_size["proprioception"])[0]
        )

    @property
    def observation_size(self):
        obs = self.non_flattened_observation_size
        return jp.sum(jax.flatten_util.ravel_pytree(obs)[0])

    # ------------------------------------------------------------------
    # code2code reward mode: extend reset / step
    # ------------------------------------------------------------------

    def reset(self, rng, *args, **kwargs):
        state = super().reset(rng, *args, **kwargs)
        if not self._c2c_active:
            return state

        # Initialize the rolling motion buffer with `w` copies of the
        # reset qpos. This makes the first encoder forward see a
        # constant-pose window (causal zero-padding equivalent).
        w = self._c2c_adapter.window_len
        init_qpos = state.data.qpos                            # (nq,)
        buf = jnp.broadcast_to(init_qpos[None, :], (w, init_qpos.shape[0]))

        new_info = dict(state.info)
        new_info["motion_window"] = buf
        # Mirror the c2c scalars in state.info for any per-step
        # consumer that inspects raw env output.
        new_info["episode_c2c_reward"] = jnp.float32(0.0)
        new_info["episode_c2c_distance"] = jnp.float32(0.0)
        new_info["episode_c2c_discrete_match"] = jnp.float32(0.0)
        new_info["episode_c2c_track_reward"] = state.reward.astype(jnp.float32)

        # Adding the c2c keys to state.metrics is what surfaces them in
        # the WandB eval dict. track-mjx's eval aggregator iterates
        # state.metrics.keys() at EpisodeWrapper.reset, initializes
        # state.info['episode_metrics'][k] for each, and accumulates
        # state.metrics[k] per-step. The aggregator then emits
        # eval/episode_<k> = mean of (sum of state.metrics[k] over
        # the episode) across parallel envs.
        state.metrics["c2c_reward"] = jnp.float32(0.0)
        state.metrics["c2c_distance"] = jnp.float32(0.0)
        state.metrics["c2c_discrete_match"] = jnp.float32(0.0)
        state.metrics["c2c_track_reward"] = state.reward.astype(jnp.float32)

        return state.replace(info=new_info)

    def step(self, state, action):
        state = super().step(state, action)
        if not self._c2c_active:
            return state

        adapter = self._c2c_adapter

        # 1. Roll buffer: drop oldest, append new qpos.
        new_qpos = state.data.qpos                            # (nq,)
        buf = state.info["motion_window"]                     # (w, nq)
        new_buf = jnp.concatenate(
            [buf[1:], new_qpos[None, :]], axis=0,
        )

        # 2. Encode the buffer to z (single-window, single-sample).
        z = adapter.encode_from_qpos(new_buf)                 # (code_dim,)

        # 3. Target code: first-in-stack from kpms_code obs.
        c_target = jnp.round(state.obs["kpms_code"][..., 0]).astype(jnp.int32)

        # 4. Distance + match score.
        # Guard against NaN z (from NaN qpos during physics failure): replace
        # the match score with 0 so the reward stays finite and gradients are
        # clean.  The episode is already being terminated by nan_termination,
        # so a zero match is the correct reward for a degenerate step.
        d = adapter.distance(z, c_target, distance_form=self._c2c_distance_form)
        if self._c2c_reward_form == "exp_neg_distance":
            match = jnp.nan_to_num(jnp.exp(-d / self._c2c_temperature), nan=0.0)
        else:  # "neg_distance"
            match = jnp.nan_to_num(-d, nan=0.0)

        # 5. Modify reward per reward_mode.
        if self._c2c_reward_mode == "additive":
            new_reward = (
                self._c2c_track_weight * state.reward + self._c2c_alpha * match
            )
        else:  # "replace"
            new_reward = (
                self._c2c_alpha * match + self._c2c_replace_alive_bonus
            )

        # 6. Diagnostic: argmin over the full codebook (reward-hacking
        # detector). Cheap — KxD matmul once per step.
        all_dists = adapter.distances_to_all_codes(
            z, distance_form=self._c2c_distance_form,
        )
        all_dists = jnp.nan_to_num(all_dists, nan=jnp.inf)
        nearest = jnp.argmin(all_dists)
        discrete_match = (nearest == c_target).astype(jnp.float32)

        new_info = dict(state.info)
        new_info["motion_window"] = new_buf
        scaled_reward = (self._c2c_alpha * match).astype(jnp.float32)
        scaled_distance = d.astype(jnp.float32)
        track_reward = state.reward.astype(jnp.float32)
        new_info["episode_c2c_reward"] = scaled_reward
        new_info["episode_c2c_distance"] = scaled_distance
        new_info["episode_c2c_discrete_match"] = discrete_match
        new_info["episode_c2c_track_reward"] = track_reward

        # Mirror into state.metrics so EpisodeWrapper accumulates and
        # the eval aggregator surfaces as eval/episode_c2c_*. See
        # reset() for the full plumbing comment.
        state.metrics["c2c_reward"] = scaled_reward
        state.metrics["c2c_distance"] = scaled_distance
        state.metrics["c2c_discrete_match"] = discrete_match
        state.metrics["c2c_track_reward"] = track_reward

        return state.replace(reward=new_reward, info=new_info)
