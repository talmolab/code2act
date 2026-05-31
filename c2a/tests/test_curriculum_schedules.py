"""Tests for c2a.code2code.schedules (curriculum schedule + validation)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from c2a.code2code.schedules import (
    ALPHA_MAX_HARD_CAP,
    MIN_BOOTSTRAP_STEPS,
    make_curriculum_schedules,
    tanh_schedule,
)


# Helpful default config (passes validation).
DEFAULT_CFG = {
    "alpha_min": 0.05,
    "alpha_max": 0.75,
    "tau_mid": 0.30,
    "tau_scale": 0.05,
    "beta_max": 0.10,
    "tau_div_start": 0.55,
    "tau_div_scale": 0.05,
}
DEFAULT_TIMESTEPS = int(5e9)   # ≥ 1e9 bootstrap @ tau_mid=0.3 ⇒ 1.25e9 ✓


# -------------------- generic tanh_schedule --------------------


def test_tanh_at_midpoint():
    """schedule(mid) = (v_min + v_max) / 2."""
    fn = tanh_schedule(value_min=0.1, value_max=0.9, mid=0.5, scale=0.05)
    assert float(fn(jnp.array(0.5))) == pytest.approx(0.5, abs=1e-6)


def test_tanh_endpoints_saturate():
    """schedule(0) ≈ v_min, schedule(1) ≈ v_max for sharp ramps."""
    fn = tanh_schedule(value_min=0.05, value_max=0.75, mid=0.30, scale=0.05)
    assert float(fn(jnp.array(0.0))) == pytest.approx(0.05, abs=0.01)
    assert float(fn(jnp.array(1.0))) == pytest.approx(0.75, abs=0.01)


def test_tanh_monotone_nondecreasing():
    """Schedule is monotone-nondecreasing on [0, 1]."""
    fn = tanh_schedule(0.05, 0.75, 0.30, 0.05)
    taus = jnp.linspace(0.0, 1.0, 50)
    values = np.asarray([float(fn(t)) for t in taus])
    diffs = np.diff(values)
    assert (diffs >= -1e-7).all(), (
        f"schedule must be monotone-nondecreasing; min diff = {diffs.min()}"
    )


def test_tanh_bounded():
    """Schedule values stay in [v_min, v_max]."""
    fn = tanh_schedule(0.05, 0.75, 0.30, 0.05)
    taus = jnp.linspace(-0.1, 1.1, 30)
    for t in taus:
        v = float(fn(t))
        assert 0.05 - 1e-3 <= v <= 0.75 + 1e-3, (
            f"schedule out of range at τ={float(t):.3f}: v={v}"
        )


# -------------------- make_curriculum_schedules: happy path --------------------


def test_make_curriculum_default_cfg():
    """Default config builds valid alpha_fn, beta_fn."""
    alpha_fn, beta_fn = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    assert callable(alpha_fn)
    assert callable(beta_fn)


def test_alpha_endpoint_values():
    """alpha(0) ≈ alpha_min, alpha(1) ≈ alpha_max."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    assert float(alpha_fn(jnp.array(0.0))) == pytest.approx(DEFAULT_CFG["alpha_min"], abs=0.01)
    assert float(alpha_fn(jnp.array(1.0))) == pytest.approx(DEFAULT_CFG["alpha_max"], abs=0.01)


def test_alpha_at_midpoint():
    """alpha(τ_mid) = (alpha_min + alpha_max) / 2."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    expected = (DEFAULT_CFG["alpha_min"] + DEFAULT_CFG["alpha_max"]) / 2.0
    assert float(alpha_fn(jnp.array(DEFAULT_CFG["tau_mid"]))) == pytest.approx(
        expected, abs=1e-5
    )


def test_alpha_monotone_nondecreasing():
    """alpha is monotone-nondecreasing on [0, 1]."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    taus = jnp.linspace(0.0, 1.0, 50)
    values = np.asarray([float(alpha_fn(t)) for t in taus])
    assert (np.diff(values) >= -1e-7).all()


def test_alpha_bounded_within_min_max():
    """alpha(τ) ∈ [alpha_min, alpha_max] for τ ∈ [0, 1]."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    taus = jnp.linspace(0.0, 1.0, 50)
    for t in taus:
        v = float(alpha_fn(t))
        assert DEFAULT_CFG["alpha_min"] - 1e-3 <= v <= DEFAULT_CFG["alpha_max"] + 1e-3


def test_beta_starts_at_zero_ends_at_beta_max():
    """beta(0) ≈ 0, beta(1) ≈ beta_max."""
    _, beta_fn = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    assert float(beta_fn(jnp.array(0.0))) == pytest.approx(0.0, abs=0.01)
    assert float(beta_fn(jnp.array(1.0))) == pytest.approx(DEFAULT_CFG["beta_max"], abs=0.01)


def test_beta_starts_later_than_alpha():
    """At alpha's midpoint, beta should still be near zero (diversity ramps later)."""
    alpha_fn, beta_fn = make_curriculum_schedules(DEFAULT_CFG, DEFAULT_TIMESTEPS)
    at_alpha_mid = float(beta_fn(jnp.array(DEFAULT_CFG["tau_mid"])))
    # tau_mid = 0.30, tau_div_start = 0.55, so at 0.30 we're (0.55-0.30)/0.05
    # = 5 scales before div midpoint → tanh ≈ -1 → beta ≈ 0.
    assert at_alpha_mid < 0.001, (
        f"beta should be near zero at alpha midpoint; got {at_alpha_mid}"
    )


# -------------------- make_curriculum_schedules: validation --------------------


def test_alpha_max_cap_rejected():
    """alpha_max > 0.75 raises ValueError."""
    bad_cfg = dict(DEFAULT_CFG, alpha_max=0.80)
    with pytest.raises(ValueError, match="alpha_max"):
        make_curriculum_schedules(bad_cfg, DEFAULT_TIMESTEPS)


def test_alpha_max_at_cap_allowed():
    """alpha_max = 0.75 (exact cap) is allowed."""
    cfg = dict(DEFAULT_CFG, alpha_max=ALPHA_MAX_HARD_CAP)
    # Should not raise.
    make_curriculum_schedules(cfg, DEFAULT_TIMESTEPS)


def test_alpha_min_invalid_rejected():
    """alpha_min < 0 or alpha_min > alpha_max rejected."""
    with pytest.raises(ValueError, match="alpha_min"):
        make_curriculum_schedules(dict(DEFAULT_CFG, alpha_min=-0.01), DEFAULT_TIMESTEPS)
    with pytest.raises(ValueError, match="alpha_min"):
        make_curriculum_schedules(dict(DEFAULT_CFG, alpha_min=0.80), DEFAULT_TIMESTEPS)


def test_bootstrap_length_check():
    """num_timesteps × (tau_mid − tau_scale) < 1e9 raises ValueError."""
    # tau_mid - tau_scale = 0.25 → need num_timesteps > 4e9 for 1e9 bootstrap
    too_short = int(3e9)   # 3e9 × 0.25 = 7.5e8 < 1e9 → reject
    with pytest.raises(ValueError, match="bootstrap"):
        make_curriculum_schedules(DEFAULT_CFG, too_short)


def test_bootstrap_check_skipped_when_no_num_timesteps():
    """num_timesteps=None skips bootstrap validation."""
    # Should not raise.
    make_curriculum_schedules(DEFAULT_CFG, num_timesteps=None)


def test_invalid_tau_scale_rejected():
    """tau_scale <= 0 rejected."""
    with pytest.raises(ValueError, match="tau_scale"):
        make_curriculum_schedules(dict(DEFAULT_CFG, tau_scale=0.0), DEFAULT_TIMESTEPS)


def test_invalid_tau_mid_rejected():
    """tau_mid outside [0, 1] rejected."""
    with pytest.raises(ValueError, match="tau_mid"):
        make_curriculum_schedules(dict(DEFAULT_CFG, tau_mid=1.5), DEFAULT_TIMESTEPS)


def test_negative_beta_max_rejected():
    """beta_max < 0 rejected."""
    with pytest.raises(ValueError, match="beta_max"):
        make_curriculum_schedules(dict(DEFAULT_CFG, beta_max=-0.1), DEFAULT_TIMESTEPS)
