"""Tests for the curriculum-mode reward composition in moseq_env_wrapper.

These don't instantiate a full MuJoCo env (too heavy for a unit test);
they exercise the composition arithmetic directly against the schedule.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from c2a.code2code.schedules import make_curriculum_schedules


DEFAULT_CFG = {
    "alpha_min": 0.05,
    "alpha_max": 0.75,
    "tau_mid": 0.30,
    "tau_scale": 0.05,
    "beta_max": 0.10,
    "tau_div_start": 0.55,
    "tau_div_scale": 0.05,
}


def _composed(alpha_fn, tau, r_track, r_c2c_raw):
    """Mirror of moseq_env_wrapper.step()'s curriculum branch:
    new_reward = (1 - α) · r_track + α · r_c2c_raw."""
    alpha = float(alpha_fn(jnp.array(tau)))
    return (1.0 - alpha) * r_track + alpha * r_c2c_raw, alpha


def test_curriculum_at_tau_zero_dominated_by_tracking():
    """At τ=0, α ≈ α_min so r_total ≈ r_track."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, int(5e9))
    r_total, alpha = _composed(alpha_fn, tau=0.0, r_track=2.0, r_c2c_raw=1.0)
    # α_min ≈ 0.05, so r_total ≈ 0.95 · 2 + 0.05 · 1 = 1.95
    assert alpha < 0.06
    assert r_total == pytest.approx(0.95 * 2.0 + 0.05 * 1.0, abs=0.05)


def test_curriculum_at_tau_one_capped_at_alpha_max():
    """At τ=1, α saturates at α_max (= 0.75), and r_total uses the floor."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, int(5e9))
    r_total, alpha = _composed(alpha_fn, tau=1.0, r_track=2.0, r_c2c_raw=1.0)
    assert alpha == pytest.approx(0.75, abs=0.01)
    # r_total = 0.25 · 2.0 + 0.75 · 1.0 = 1.25
    assert r_total == pytest.approx(0.25 * 2.0 + 0.75 * 1.0, abs=0.05)


def test_curriculum_at_tau_mid_is_average():
    """At τ = τ_mid, α = (α_min + α_max) / 2."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, int(5e9))
    _, alpha = _composed(
        alpha_fn, tau=DEFAULT_CFG["tau_mid"], r_track=1.0, r_c2c_raw=1.0
    )
    expected = 0.5 * (DEFAULT_CFG["alpha_min"] + DEFAULT_CFG["alpha_max"])
    assert alpha == pytest.approx(expected, abs=1e-5)


def test_curriculum_tracking_floor_always_present():
    """For all τ ∈ [0, 1], (1 - α) ≥ 0.25 — tracking is never fully wiped."""
    alpha_fn, _ = make_curriculum_schedules(DEFAULT_CFG, int(5e9))
    taus = jnp.linspace(0.0, 1.0, 50)
    for t in taus:
        alpha = float(alpha_fn(t))
        track_floor = 1.0 - alpha
        assert track_floor >= 0.24, (
            f"Track floor must be ≥ 0.24 for stability; "
            f"got {track_floor:.4f} at τ={float(t):.3f}"
        )


def test_beta_negligible_during_alpha_transition():
    """Diversity bonus β is near zero through the α transition window."""
    _, beta_fn = make_curriculum_schedules(DEFAULT_CFG, int(5e9))
    # α transition is centered at 0.30 with scale 0.05; β starts at 0.55.
    # During the α transition (say τ ∈ [0.2, 0.4]), β should be < 1e-2.
    transition_taus = jnp.linspace(0.20, 0.40, 10)
    for t in transition_taus:
        b = float(beta_fn(t))
        assert b < 1e-2, (
            f"β should be negligible during α transition; "
            f"got β({float(t):.3f}) = {b:.6f}"
        )


def test_validation_rejects_short_bootstrap():
    """Bootstrap shorter than 1e9 env steps raises ValueError."""
    # Default tau_mid=0.30, tau_scale=0.05 ⇒ bootstrap_fraction = 0.25.
    # 0.25 × num_timesteps ≥ 1e9 ⇒ num_timesteps ≥ 4e9.
    with pytest.raises(ValueError, match="bootstrap"):
        make_curriculum_schedules(DEFAULT_CFG, int(3e9))  # 0.75e9 < 1e9


def test_validation_accepts_5b_default():
    """5B env steps with tau_mid=0.30 gives 1.25B bootstrap — accepted."""
    # Should not raise.
    make_curriculum_schedules(DEFAULT_CFG, int(5e9))
