"""Tests for c2a.code2code.schedules.make_finetune_schedules.

The finetune schedule is simpler than curriculum's tanh ramp:
* warmup_fraction = 0  →  pure constant α for all τ
* warmup_fraction > 0  →  linear ramp 0 → target over first
  ``warmup_fraction`` of training, then hold at target
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from c2a.code2code.schedules import (
    ALPHA_MAX_HARD_CAP,
    make_curriculum_schedules,  # for cap-shared check
    make_finetune_schedules,
)


DEFAULT_CFG = {
    "alpha": 0.75,
    "warmup_fraction": 0.05,
}


# -------------------- warmup_fraction == 0 (pure constants) --------------------


def test_warmup_zero_gives_constant_alpha():
    """warmup_fraction=0 → α(τ) = alpha for all τ ∈ [0, 1]."""
    cfg = dict(DEFAULT_CFG, warmup_fraction=0.0)
    alpha_fn = make_finetune_schedules(cfg, num_timesteps=int(1e9))
    for t in jnp.linspace(0.0, 1.0, 21):
        assert float(alpha_fn(t)) == pytest.approx(DEFAULT_CFG["alpha"], abs=1e-6), (
            f"α({float(t):.3f}) should be {DEFAULT_CFG['alpha']}; got {float(alpha_fn(t)):.6f}"
        )


# -------------------- warmup_fraction > 0 (linear ramp) --------------------


def test_warmup_ramps_to_target():
    """Linear ramp: α(0)=0, α(warmup/2)=alpha/2, α(warmup)=alpha,
    α(τ>warmup)=alpha."""
    cfg = DEFAULT_CFG.copy()
    alpha_fn = make_finetune_schedules(cfg, num_timesteps=int(1e9))
    w = DEFAULT_CFG["warmup_fraction"]
    alpha = DEFAULT_CFG["alpha"]

    assert float(alpha_fn(jnp.array(0.0))) == pytest.approx(0.0, abs=1e-6)
    assert float(alpha_fn(jnp.array(w / 2))) == pytest.approx(alpha / 2, abs=1e-6)
    assert float(alpha_fn(jnp.array(w))) == pytest.approx(alpha, abs=1e-6)
    # τ past warmup: held at target.
    assert float(alpha_fn(jnp.array(w + 0.01))) == pytest.approx(alpha, abs=1e-6)
    assert float(alpha_fn(jnp.array(0.5))) == pytest.approx(alpha, abs=1e-6)
    assert float(alpha_fn(jnp.array(1.0))) == pytest.approx(alpha, abs=1e-6)


def test_warmup_monotone_nondecreasing():
    """Ramp is monotone-nondecreasing on [0, 1]."""
    cfg = DEFAULT_CFG.copy()
    alpha_fn = make_finetune_schedules(cfg, num_timesteps=int(1e9))
    taus = jnp.linspace(0.0, 1.0, 100)
    values = np.asarray([float(alpha_fn(t)) for t in taus])
    assert (np.diff(values) >= -1e-7).all()


# -------------------- validation --------------------


def test_alpha_cap_enforced():
    """alpha > ALPHA_MAX_HARD_CAP raises ValueError (same as curriculum)."""
    bad = dict(DEFAULT_CFG, alpha=0.80)
    with pytest.raises(ValueError, match="alpha"):
        make_finetune_schedules(bad)


def test_alpha_at_cap_allowed():
    """alpha = ALPHA_MAX_HARD_CAP exactly is allowed."""
    cfg = dict(DEFAULT_CFG, alpha=ALPHA_MAX_HARD_CAP)
    make_finetune_schedules(cfg)  # should not raise


def test_alpha_out_of_range_rejected():
    """alpha < 0 or alpha > 1 rejected (modulo cap, which is stricter)."""
    with pytest.raises(ValueError, match="alpha"):
        make_finetune_schedules(dict(DEFAULT_CFG, alpha=-0.01))


def test_invalid_warmup_rejected():
    """warmup_fraction not in [0, 1) rejected."""
    with pytest.raises(ValueError, match="warmup_fraction"):
        make_finetune_schedules(dict(DEFAULT_CFG, warmup_fraction=1.0))
    with pytest.raises(ValueError, match="warmup_fraction"):
        make_finetune_schedules(dict(DEFAULT_CFG, warmup_fraction=-0.01))


def test_no_bootstrap_validation():
    """Finetune has no bootstrap-length check — tiny num_timesteps is allowed.
    Curriculum would reject this because tau_mid*num_timesteps < 1e9."""
    # No raise even with absurdly tiny num_timesteps.
    make_finetune_schedules(DEFAULT_CFG, num_timesteps=10)
    make_finetune_schedules(DEFAULT_CFG, num_timesteps=int(1e6))
    # Contrast: same num_timesteps would fail curriculum.
    with pytest.raises(ValueError, match="bootstrap"):
        make_curriculum_schedules(
            {"alpha_min": 0.05, "alpha_max": 0.75, "tau_mid": 0.30, "tau_scale": 0.05},
            num_timesteps=int(1e6),
        )


def test_num_timesteps_none_ok():
    """num_timesteps=None is fine (no validation depends on it)."""
    make_finetune_schedules(DEFAULT_CFG, num_timesteps=None)
