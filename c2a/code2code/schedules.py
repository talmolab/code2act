"""Curriculum schedule utilities for the c2c curriculum reward mode.

Schedules are smooth tanh ramps in the normalized training progress
``τ = current_step / target_total_steps``:

    schedule(τ) = v_min + (v_max − v_min) · 0.5 · (1 + tanh((τ − mid) / scale))

So ``schedule(mid) = (v_min + v_max) / 2`` exactly, and the value
saturates near v_max as τ → 1 (sharpness controlled by ``scale``).

This module exposes:

* :func:`tanh_schedule` — generic tanh ramp factory.
* :func:`make_curriculum_schedules` — validates curriculum-config dict
  and returns ``alpha_fn`` for the tracking/c2c reward tradeoff.
  Validation rejects α_max > 0.75 and configs where bootstrap < 1e9
  env steps (see retrospective
  ``scratch/registry/2026-05-30-c2c-reward-is-root-invariant/``).
* :func:`make_finetune_schedules` — same but for finetune mode (no
  bootstrap requirement; linear warmup instead of tanh).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp


# Hard caps enforced at config-validation time.  See the
# c2c-reward-is-root-invariant retrospective and the curriculum plan.
ALPHA_MAX_HARD_CAP = 0.75
MIN_BOOTSTRAP_STEPS = int(1e9)


def tanh_schedule(
    value_min: float,
    value_max: float,
    mid: float,
    scale: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Generic tanh schedule factory.

    Args:
        value_min: value at τ → −∞ (also approximately at τ = 0 for
            reasonable mid/scale).
        value_max: value at τ → +∞ (approximately at τ = 1).
        mid: τ at which the schedule reaches (v_min + v_max) / 2.
        scale: transition sharpness.  Smaller → sharper transition.

    Returns:
        Callable ``f(τ) -> scheduled value`` that broadcasts over τ.
    """
    v_min = float(value_min)
    v_max = float(value_max)
    m = float(mid)
    s = float(scale)

    def schedule_fn(tau: jnp.ndarray) -> jnp.ndarray:
        return v_min + (v_max - v_min) * 0.5 * (1.0 + jnp.tanh((tau - m) / s))

    return schedule_fn


def make_curriculum_schedules(
    cfg: Any,
    num_timesteps: int | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build ``alpha_fn`` from a curriculum_config mapping.

    The config must supply ``alpha_min``, ``alpha_max``, ``tau_mid``,
    ``tau_scale``.

    Validation:

    * ``alpha_max`` cannot exceed :data:`ALPHA_MAX_HARD_CAP` (0.75).
      Higher values would let c2c reward dominate over tracking once the
      schedule saturates, which is the failure mode the retrospective
      documents.
    * If ``num_timesteps`` is supplied, the bootstrap window
      ``num_timesteps · max(τ_mid − τ_scale, 0)`` must be at least
      :data:`MIN_BOOTSTRAP_STEPS` (1e9 env steps) — otherwise the policy
      enters the transition before tracking has converged.

    Args:
        cfg: mapping-like (DictConfig, dict, or attribute-only object)
            with the curriculum keys listed above.
        num_timesteps: total training env-steps target (for bootstrap
            validation).  Skip the check when None.

    Returns:
        ``alpha_fn`` — a callable that takes τ ∈ [0, 1] and returns the
        scheduled tracking/c2c tradeoff weight.

    Raises:
        ValueError: on any validation failure.
    """
    def _get(key, default=None):
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    alpha_min = float(_get("alpha_min", 0.05))
    alpha_max = float(_get("alpha_max", 0.75))
    tau_mid = float(_get("tau_mid", 0.30))
    tau_scale = float(_get("tau_scale", 0.05))

    # ── Hard cap on alpha_max ─────────────────────────────────────────
    if alpha_max > ALPHA_MAX_HARD_CAP + 1e-9:
        raise ValueError(
            f"curriculum_config.alpha_max={alpha_max} exceeds the hard "
            f"cap {ALPHA_MAX_HARD_CAP}.  Raising α_max above 0.75 lets "
            f"the (root-invariant) c2c reward dominate over tracking "
            f"asymptotically, which empirically lets the policy drift "
            f"in world coordinates.  See "
            f"scratch/registry/2026-05-30-c2c-reward-is-root-invariant/."
        )
    if alpha_min < 0.0 or alpha_min > alpha_max:
        raise ValueError(
            f"curriculum_config.alpha_min={alpha_min} must satisfy "
            f"0 ≤ alpha_min ≤ alpha_max={alpha_max}."
        )
    if tau_scale <= 0.0:
        raise ValueError(
            "curriculum_config.tau_scale must be > 0."
        )
    if not (0.0 <= tau_mid <= 1.0):
        raise ValueError(
            "curriculum_config.tau_mid must be in [0, 1]."
        )

    # ── Bootstrap length check ────────────────────────────────────────
    if num_timesteps is not None:
        bootstrap_fraction = max(tau_mid - tau_scale, 0.0)
        bootstrap_steps = num_timesteps * bootstrap_fraction
        if bootstrap_steps < MIN_BOOTSTRAP_STEPS:
            raise ValueError(
                f"curriculum bootstrap window is only "
                f"{bootstrap_steps:.2e} env steps "
                f"({tau_mid=}, {tau_scale=}, "
                f"num_timesteps={num_timesteps}), but tracking needs at "
                f"least {MIN_BOOTSTRAP_STEPS:.2e} steps to converge "
                f"under PPO + tracking reward.  Either increase "
                f"num_timesteps so num_timesteps × (tau_mid − tau_scale) "
                f"≥ {MIN_BOOTSTRAP_STEPS:.0e}, or lower tau_mid / "
                f"tau_scale accordingly."
            )

    return tanh_schedule(alpha_min, alpha_max, tau_mid, tau_scale)


def make_finetune_schedules(
    cfg: Any,
    num_timesteps: int | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build ``alpha_fn`` for the ``finetune`` reward mode.

    Unlike :func:`make_curriculum_schedules`, finetune mode does NOT need
    a bootstrap window — the policy is loaded pretrained from a
    pure-tracking checkpoint. Defaults:

    * α ramps linearly from 0 to its target over the first
      ``warmup_fraction`` of training (default 5%), then holds constant.
    * ``warmup_fraction = 0`` gives a pure constant from τ = 0.

    The α ≤ :data:`ALPHA_MAX_HARD_CAP` cap is still enforced — c2c is
    root-invariant regardless of training origin, so the tracking
    floor (1 − α) ≥ 0.25 stability argument applies.

    Args:
        cfg: mapping-like with keys ``alpha`` (default 0.75),
            ``warmup_fraction`` (default 0.05).
        num_timesteps: unused (kept for signature parity with
            :func:`make_curriculum_schedules`; finetune has no
            bootstrap-length validation to apply).

    Returns:
        ``alpha_fn`` — a callable that takes τ ∈ [0, 1] and returns the
        scheduled tracking/c2c tradeoff weight.

    Raises:
        ValueError: on validation failure (α > cap, α out of range,
        warmup_fraction not in [0, 1)).
    """
    del num_timesteps  # signature parity only

    def _get(key, default=None):
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    alpha = float(_get("alpha", 0.75))
    warmup = float(_get("warmup_fraction", 0.05))

    if alpha > ALPHA_MAX_HARD_CAP + 1e-9:
        raise ValueError(
            f"finetune_config.alpha={alpha} exceeds the hard cap "
            f"{ALPHA_MAX_HARD_CAP}. See "
            f"scratch/registry/2026-05-30-c2c-reward-is-root-invariant/ "
            f"— the c2c reward is root-invariant; tracking weight (1−α) "
            f"must remain ≥ 0.25 to keep the policy from drifting in "
            f"world coordinates."
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            f"finetune_config.alpha={alpha} must be in [0, 1]."
        )
    if not (0.0 <= warmup < 1.0):
        raise ValueError(
            f"finetune_config.warmup_fraction={warmup} must be in [0, 1)."
        )

    if warmup <= 0.0:
        # Pure constant — return a closure that ignores τ.
        def alpha_fn(tau: jnp.ndarray) -> jnp.ndarray:
            return jnp.full_like(jnp.asarray(tau, dtype=jnp.float32), alpha)
    else:
        # Linear ramp 0 → target over first ``warmup`` fraction, then
        # hold at target.  jnp.minimum saturates the ramp at 1.0.
        def alpha_fn(tau: jnp.ndarray) -> jnp.ndarray:
            ramp = jnp.minimum(tau / warmup, 1.0)
            return alpha * ramp

    return alpha_fn
