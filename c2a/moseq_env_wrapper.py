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


C2CRewardMode = Literal["off", "additive", "replace", "curriculum"]
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
        # Curriculum-mode knobs (only consulted when reward_mode == 'curriculum')
        c2c_curriculum_alpha_fn: Optional[Any] = None,
        c2c_curriculum_target_total_steps: Optional[int] = None,
    ):
        super().__init__(config=config, clips=clips)
        self._kpms_codes = jnp.asarray(kpms_codes, dtype=jnp.int32)
        self._code_stack_size = int(code_stack_size)

        if c2c_reward_mode not in ("off", "additive", "replace", "curriculum"):
            raise ValueError(
                f"c2c_reward_mode must be 'off', 'additive', 'replace', or "
                f"'curriculum'; got {c2c_reward_mode!r}"
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

        # Curriculum-mode plumbing.
        self._c2c_curriculum_alpha_fn = c2c_curriculum_alpha_fn
        if c2c_reward_mode == "curriculum":
            if c2c_curriculum_alpha_fn is None:
                raise ValueError(
                    "c2c_reward_mode='curriculum' requires "
                    "c2c_curriculum_alpha_fn (build via "
                    "c2a.code2code.schedules.make_curriculum_schedules)."
                )
            if (
                c2c_curriculum_target_total_steps is None
                or c2c_curriculum_target_total_steps <= 0
            ):
                raise ValueError(
                    "c2c_reward_mode='curriculum' requires "
                    "c2c_curriculum_target_total_steps > 0."
                )
            self._c2c_curriculum_target_total_steps = float(
                c2c_curriculum_target_total_steps
            )
        else:
            self._c2c_curriculum_target_total_steps = 1.0

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

        # Rate alignment between the rollout buffer (control-rate) and
        # the tokenizer (mocap-rate). The VAE/VQ encoder was fit on
        # windows sampled at the reference's mocap_hz; the env runs at
        # 1/ctrl_dt Hz. To feed the encoder a window that spans the
        # same physical duration with the same per-frame spacing as
        # training, we keep a (window_len * stride)-long buffer at
        # control-rate and subsample [stride-1::stride] at encode time.
        # stride = round(1 / (ctrl_dt * mocap_hz)).
        if self._c2c_active:
            rate_ratio = 1.0 / (config.ctrl_dt * config.mocap_hz)
            stride = round(rate_ratio)
            if abs(stride - rate_ratio) > 1e-9:
                raise ValueError(
                    f"c2c rate alignment: 1/(ctrl_dt*mocap_hz) = "
                    f"{rate_ratio} is not an integer. Cannot align the "
                    f"rollout buffer to the tokenizer's training rate; "
                    f"either retrain the tokenizer at control rate or "
                    f"set ctrl_dt = 1/(mocap_hz * k) for some integer k."
                )
            self._c2c_buffer_stride = int(stride)
        else:
            self._c2c_buffer_stride = 1

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

        # Initialize the rolling motion buffer with ``w * stride`` copies
        # of the reset qpos. The buffer accumulates one qpos per control
        # step; we subsample it to mocap-rate spacing at encode time
        # (see step() and __init__ for the rate-alignment rationale).
        # Buffer size = adapter.window_len * c2c_buffer_stride so that
        # buf[stride-1::stride] yields exactly window_len entries at
        # mocap-rate spacing.
        w = self._c2c_adapter.window_len
        stride = self._c2c_buffer_stride
        buf_size = w * stride
        init_qpos = state.data.qpos                            # (nq,)
        buf = jnp.broadcast_to(init_qpos[None, :], (buf_size, init_qpos.shape[0]))

        new_info = dict(state.info)
        new_info["motion_window"] = buf
        # Curriculum-mode env step counter (incremented once per step()
        # call from 0 at reset).  Used to compute τ = step / target and
        # therefore α(τ) inside step().  Always present so the pytree
        # shape is uniform across the env's lifetime.
        new_info["c2c_env_step"] = jnp.int32(0)
        # Per-step c2c reward components — needed by the curriculum
        # PPO-loss path for Vendi diversity reward.  Default to zeros
        # so the state.info pytree shape stays uniform.
        new_info["c2c_z_rollout"] = jnp.zeros(
            (self._c2c_adapter.code_dim,), dtype=jnp.float32
        )
        new_info["c2c_target_code"] = jnp.int32(0)
        # Mirror the c2c scalars in state.info for any per-step
        # consumer that inspects raw env output.
        new_info["episode_c2c_reward"] = jnp.float32(0.0)
        new_info["episode_c2c_distance"] = jnp.float32(0.0)
        new_info["episode_c2c_discrete_match"] = jnp.float32(0.0)
        new_info["episode_c2c_track_reward"] = state.reward.astype(jnp.float32)
        # Per-step "reconstructed" code: the frozen tokenizer's nearest
        # codebook entry for the rollout's actual motion. Logging the
        # raw value (alongside the binary ``discrete_match``) lets the
        # render pipeline draw a second timeline bar comparing the
        # target code to what the agent's motion actually maps to.
        # Sentinel ``-1`` at reset; populated by ``step`` once a motion
        # window has accrued.
        new_info["c2c_reconstructed_code"] = jnp.int32(-1)

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
        # Curriculum-mode metrics — always present for uniform pytree.
        # Populated meaningfully only when reward_mode == 'curriculum'.
        state.metrics["c2c_curriculum_alpha"] = jnp.float32(0.0)
        state.metrics["c2c_curriculum_r_track"] = state.reward.astype(jnp.float32)
        state.metrics["c2c_curriculum_r_c2c_raw"] = jnp.float32(0.0)

        return state.replace(info=new_info)

    def step(self, state, action):
        state = super().step(state, action)
        if not self._c2c_active:
            return state

        adapter = self._c2c_adapter

        # 1. Roll buffer: drop oldest, append new qpos.
        new_qpos = state.data.qpos                            # (nq,)
        buf = state.info["motion_window"]                     # (w*stride, nq)
        new_buf = jnp.concatenate(
            [buf[1:], new_qpos[None, :]], axis=0,
        )

        # 2. Subsample to mocap-rate spacing and encode.
        # With buffer of length w*stride filled at control rate, the
        # slice [stride-1::stride] gives exactly w entries spaced by
        # mocap_hz (the tokenizer's training rate), ending at the
        # newest entry. See __init__ for the stride derivation.
        stride = self._c2c_buffer_stride
        encoder_window = new_buf[stride - 1::stride]          # (w, nq)
        z = adapter.encode_from_qpos(encoder_window)          # (code_dim,)

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
        # Per-step curriculum bookkeeping (computed for ALL modes; only
        # consumed in 'curriculum' mode).  Always-present fields keep
        # the pytree shape uniform across modes.
        prev_env_step = state.info["c2c_env_step"]
        new_env_step = prev_env_step + jnp.int32(1)
        tau = jnp.float32(new_env_step) / jnp.float32(
            self._c2c_curriculum_target_total_steps
        )
        if self._c2c_reward_mode == "additive":
            new_reward = (
                self._c2c_track_weight * state.reward + self._c2c_alpha * match
            )
            curriculum_alpha = jnp.float32(0.0)
        elif self._c2c_reward_mode == "replace":
            new_reward = (
                self._c2c_alpha * match + self._c2c_replace_alive_bonus
            )
            curriculum_alpha = jnp.float32(0.0)
        else:  # "curriculum" — tanh-scheduled (1-α)·track + α·c2c
            curriculum_alpha = self._c2c_curriculum_alpha_fn(tau).astype(jnp.float32)
            # tracking reward (already weighted by env's own reward
            # terms) at (1-α), raw c2c match (scaled by self._c2c_alpha
            # which is its base magnitude knob) at α.
            new_reward = (
                (1.0 - curriculum_alpha) * state.reward
                + curriculum_alpha * (self._c2c_alpha * match)
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
        new_info["c2c_env_step"] = new_env_step
        scaled_reward = (self._c2c_alpha * match).astype(jnp.float32)
        scaled_distance = d.astype(jnp.float32)
        track_reward = state.reward.astype(jnp.float32)
        new_info["episode_c2c_reward"] = scaled_reward
        new_info["episode_c2c_distance"] = scaled_distance
        new_info["episode_c2c_discrete_match"] = discrete_match
        new_info["episode_c2c_track_reward"] = track_reward
        # Curriculum-mode extras (consumed by the PPO loss for Vendi).
        # Stored every step so the pytree shape is mode-invariant.
        new_info["c2c_z_rollout"] = z.astype(jnp.float32)
        new_info["c2c_target_code"] = c_target.astype(jnp.int32)
        # Per-step reconstructed code (raw argmin index) — see reset()
        # comment. Used by the render pipeline to draw a "what code the
        # rollout's motion encodes to" timeline bar.
        new_info["c2c_reconstructed_code"] = nearest.astype(jnp.int32)

        # Mirror into state.metrics so EpisodeWrapper accumulates and
        # the eval aggregator surfaces as eval/episode_c2c_*. See
        # reset() for the full plumbing comment.
        state.metrics["c2c_reward"] = scaled_reward
        state.metrics["c2c_distance"] = scaled_distance
        state.metrics["c2c_discrete_match"] = discrete_match
        state.metrics["c2c_track_reward"] = track_reward
        state.metrics["c2c_curriculum_alpha"] = curriculum_alpha
        state.metrics["c2c_curriculum_r_track"] = track_reward
        state.metrics["c2c_curriculum_r_c2c_raw"] = scaled_reward

        return state.replace(reward=new_reward, info=new_info)
