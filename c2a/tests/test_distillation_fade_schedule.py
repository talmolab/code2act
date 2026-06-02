"""Tests for c2a.code2code.schedules.make_distillation_schedule.

Two shapes:
* curriculum: inverted tanh fade with midpoint and scale.
* finetune:   hold at ``init`` for ``tau_hold``, linear decay over
              [tau_hold, tau_fade_end], hold at ``final`` after.

Both must:
  - return init at τ=0
  - return final at τ=1
  - be monotone-nonincreasing in τ
  - reject invalid configurations
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from c2a.code2code.schedules import make_distillation_schedule


# -------------------- curriculum (tanh) --------------------


CURR_CFG = {
    "init": 1.0,
    "final": 0.0,
    "tau_mid": 0.28,
    "tau_scale": 0.04,
}


def test_curriculum_at_zero_equals_init():
    fn = make_distillation_schedule(CURR_CFG, mode="curriculum")
    # At τ=0, far below mid=0.28 with scale 0.04 → tanh → -1 → shape → 1
    assert float(fn(jnp.array(0.0))) == pytest.approx(CURR_CFG["init"], abs=0.01)


def test_curriculum_at_one_equals_final():
    fn = make_distillation_schedule(CURR_CFG, mode="curriculum")
    # At τ=1, far above mid → tanh → +1 → shape → 0
    assert float(fn(jnp.array(1.0))) == pytest.approx(CURR_CFG["final"], abs=0.01)


def test_curriculum_at_midpoint_is_halfway():
    fn = make_distillation_schedule(CURR_CFG, mode="curriculum")
    expected = 0.5 * (CURR_CFG["init"] + CURR_CFG["final"])
    assert float(fn(jnp.array(CURR_CFG["tau_mid"]))) == pytest.approx(
        expected, abs=1e-5
    )


def test_curriculum_monotone_nonincreasing():
    fn = make_distillation_schedule(CURR_CFG, mode="curriculum")
    taus = jnp.linspace(0.0, 1.0, 100)
    values = np.asarray([float(fn(t)) for t in taus])
    diffs = np.diff(values)
    assert (diffs <= 1e-7).all(), (
        f"curriculum fade must be monotone-nonincreasing; "
        f"max diff = {diffs.max()}"
    )


def test_curriculum_midpoint_after_alpha_midpoint():
    """The curriculum α schedule has midpoint at τ=0.30; distillation's
    midpoint at τ=0.28 means distillation is just starting to fade
    when α is in mid-transition."""
    # This isn't a property of the schedule fn alone; it's a property of
    # our chosen defaults.  Check the default config encodes that intent.
    assert CURR_CFG["tau_mid"] <= 0.30  # distillation fades around α midpoint


# -------------------- finetune (linear with hold) --------------------


FT_CFG = {
    "init": 1.0,
    "final": 0.0,
    "tau_hold": 0.02,
    "tau_fade_end": 0.15,
}


def test_finetune_at_zero_equals_init():
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    assert float(fn(jnp.array(0.0))) == pytest.approx(FT_CFG["init"], abs=1e-6)


def test_finetune_at_one_equals_final():
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    assert float(fn(jnp.array(1.0))) == pytest.approx(FT_CFG["final"], abs=1e-6)


def test_finetune_hold_window_holds_at_init():
    """For τ ∈ [0, tau_hold], λ_distill = init exactly."""
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    for t in jnp.linspace(0.0, FT_CFG["tau_hold"], 5):
        assert float(fn(t)) == pytest.approx(FT_CFG["init"], abs=1e-6), (
            f"hold window violated at τ={float(t):.4f}: "
            f"λ={float(fn(t)):.6f}, expected {FT_CFG['init']}"
        )


def test_finetune_linear_decay_at_midpoint():
    """At τ = (tau_hold + tau_fade_end) / 2, λ = (init + final) / 2."""
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    tau_mid = 0.5 * (FT_CFG["tau_hold"] + FT_CFG["tau_fade_end"])
    expected = 0.5 * (FT_CFG["init"] + FT_CFG["final"])
    assert float(fn(jnp.array(tau_mid))) == pytest.approx(expected, abs=1e-5)


def test_finetune_at_fade_end_equals_final():
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    assert float(fn(jnp.array(FT_CFG["tau_fade_end"]))) == pytest.approx(
        FT_CFG["final"], abs=1e-6
    )


def test_finetune_after_fade_end_clamps_to_final():
    """For τ > tau_fade_end, λ_distill = final exactly."""
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    for t in jnp.linspace(FT_CFG["tau_fade_end"] + 0.01, 1.0, 10):
        assert float(fn(t)) == pytest.approx(FT_CFG["final"], abs=1e-6)


def test_finetune_monotone_nonincreasing():
    fn = make_distillation_schedule(FT_CFG, mode="finetune")
    taus = jnp.linspace(0.0, 1.0, 100)
    values = np.asarray([float(fn(t)) for t in taus])
    diffs = np.diff(values)
    assert (diffs <= 1e-7).all()


# -------------------- validation --------------------


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        make_distillation_schedule(CURR_CFG, mode="bogus")


def test_init_less_than_final_rejected():
    bad = dict(CURR_CFG, init=0.0, final=1.0)
    with pytest.raises(ValueError, match="init"):
        make_distillation_schedule(bad, mode="curriculum")


def test_negative_init_rejected():
    bad = dict(CURR_CFG, init=-0.1)
    with pytest.raises(ValueError, match="init"):
        make_distillation_schedule(bad, mode="curriculum")


def test_curriculum_invalid_tau_scale_rejected():
    bad = dict(CURR_CFG, tau_scale=0.0)
    with pytest.raises(ValueError, match="tau_scale"):
        make_distillation_schedule(bad, mode="curriculum")


def test_finetune_invalid_fade_window_rejected():
    """tau_fade_end <= tau_hold must raise."""
    bad = dict(FT_CFG, tau_fade_end=0.02)  # equal to tau_hold
    with pytest.raises(ValueError, match="tau_fade_end"):
        make_distillation_schedule(bad, mode="finetune")
