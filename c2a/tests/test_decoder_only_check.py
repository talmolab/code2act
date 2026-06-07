"""Contract tests for c2a.experiments.shared.decoder_only.

Uses mock inference fns (no checkpoint) to verify:
  * assert_decoder_only_settings rejects deterministic=False / z_e_scale!=0.
  * verify_encoder_severed passes when the policy ignores task_obs, and raises
    when the policy reads task_obs (encoder leaking into the action).

Run standalone:  python3.12 bash_scripts/_pyrun.py c2a.tests.test_decoder_only_check
Or via pytest:   pytest c2a/tests/test_decoder_only_check.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from c2a.experiments.shared.decoder_only import (
    assert_decoder_only_settings,
    verify_encoder_severed,
)


def _obs(batch: int = 1):
    return {
        "kpms_code": jnp.zeros((batch, 1), jnp.float32),
        "proprioception": jnp.ones((batch, 3), jnp.float32),
        "task_obs": jnp.ones((batch, 5), jnp.float32),
    }


def _hidden(batch: int = 1):
    return [jnp.zeros((batch, 4), jnp.float32)]


def _severed_fn(params, obs, hidden, key):
    """Action ignores task_obs — encoder is severed."""
    action = obs["proprioception"].sum(axis=-1, keepdims=True) + obs["kpms_code"]
    return action, {}, hidden


def _leaking_fn(params, obs, hidden, key):
    """Action depends on task_obs — encoder leaks into the policy."""
    action = obs["task_obs"].sum(axis=-1, keepdims=True)
    new_hidden = [h + obs["task_obs"].sum() for h in hidden]
    return action, {}, new_hidden


def test_assert_rejects_nondeterministic():
    with pytest.raises(ValueError):
        assert_decoder_only_settings(z_e_scale=0.0, deterministic=False)


def test_assert_rejects_nonzero_z_e_scale():
    with pytest.raises(ValueError):
        assert_decoder_only_settings(z_e_scale=1.0, deterministic=True)


def test_assert_accepts_decoder_only():
    assert_decoder_only_settings(z_e_scale=0.0, deterministic=True)  # no raise


def test_verify_passes_when_severed():
    rep = verify_encoder_severed(
        _severed_fn, None, _obs(), _hidden(), jax.random.PRNGKey(0),
    )
    assert rep["severed"] is True
    assert rep["max_abs_diff"] == 0.0


def test_verify_raises_when_leaking():
    with pytest.raises(RuntimeError):
        verify_encoder_severed(
            _leaking_fn, None, _obs(), _hidden(), jax.random.PRNGKey(0),
        )


if __name__ == "__main__":
    test_assert_rejects_nondeterministic()
    test_assert_rejects_nonzero_z_e_scale()
    test_assert_accepts_decoder_only()
    test_verify_passes_when_severed()
    test_verify_raises_when_leaking()
    print("decoder_only checks: ALL PASS")
