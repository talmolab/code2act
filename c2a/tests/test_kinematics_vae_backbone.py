"""Tests for the kinematics-VAE backbone (data + encoder/decoder/predictor
+ loss + window padding).

Mirrors the small focused tests Scott uses on the latent-mimic-handoff
branch so regressions in the c2a port show up immediately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ── frame_features ────────────────────────────────────────────────────────


class _FakeClips:
    """Minimal stand-in for vnl_playground.ReferenceClips that the
    frame_features module's Protocol accepts."""

    def __init__(self, qpos, qvel):
        self.qpos = qpos
        self.qvel = qvel


def test_motion_frame_dim_formula():
    from c2a.codegen.kinematics_vae.backbone.frame_features import MOTION_FRAME_DIM

    # use_qvel=True with 67 joints → 3 + 4 + 6 + 67 + 67 = 147
    assert MOTION_FRAME_DIM(n_joints=67, use_qvel=True) == 147
    # use_qvel=False with 67 joints → 3 + 4 + 67 = 74
    assert MOTION_FRAME_DIM(n_joints=67, use_qvel=False) == 74
    # use_qvel=False with n_active=24 → 3 + 4 + 24 = 31
    assert MOTION_FRAME_DIM(67, use_qvel=False, n_active_joints=24) == 31


def test_extract_motion_frames_no_qvel():
    from c2a.codegen.kinematics_vae.backbone.frame_features import (
        extract_motion_frames,
    )

    n_joints = 5
    n_clips, n_frames = 2, 3
    qpos = np.arange(n_clips * n_frames * (7 + n_joints), dtype=np.float32).reshape(
        n_clips, n_frames, 7 + n_joints,
    )
    qvel = np.zeros((n_clips, n_frames, 6 + n_joints), dtype=np.float32)
    clips = _FakeClips(qpos, qvel)

    out = extract_motion_frames(clips, n_joints=n_joints, use_qvel=False)
    # (p, theta, q) → 3 + 4 + 5 = 12 dims
    assert out.shape == (n_clips, n_frames, 12)
    # p is qpos[..., :3]
    np.testing.assert_array_equal(out[..., :3], qpos[..., :3])
    # theta is qpos[..., 3:7]
    np.testing.assert_array_equal(out[..., 3:7], qpos[..., 3:7])
    # q is qpos[..., 7:]
    np.testing.assert_array_equal(out[..., 7:], qpos[..., 7:])


def test_extract_motion_frames_with_active_joints():
    from c2a.codegen.kinematics_vae.backbone.frame_features import (
        extract_motion_frames,
    )

    n_joints = 5
    qpos = np.arange(2 * 1 * (7 + n_joints), dtype=np.float32).reshape(2, 1, 7 + n_joints)
    qvel = np.zeros((2, 1, 6 + n_joints), dtype=np.float32)
    clips = _FakeClips(qpos, qvel)

    out = extract_motion_frames(
        clips, n_joints=n_joints, use_qvel=False, active_joints=[1, 3],
    )
    # Only joints 1 and 3 selected → feat_dim = 3 + 4 + 2 = 9.
    assert out.shape == (2, 1, 9)
    np.testing.assert_array_equal(out[..., -2], qpos[..., 7 + 1])
    np.testing.assert_array_equal(out[..., -1], qpos[..., 7 + 3])


def test_auto_detect_active_joints_drops_zero_variance():
    from c2a.codegen.kinematics_vae.backbone.frame_features import (
        auto_detect_active_joints,
    )

    n_joints = 5
    qpos = np.tile(
        np.arange(7 + n_joints, dtype=np.float32),
        (3, 4, 1),
    )
    # Make joint 2 vary; others constant.
    qpos[:, :, 7 + 2] += np.arange(qpos.shape[0] * qpos.shape[1]).reshape(qpos.shape[:2])
    clips = _FakeClips(qpos, np.zeros((3, 4, 6 + n_joints), dtype=np.float32))
    active = auto_detect_active_joints(clips, n_joints=n_joints, std_threshold=1e-3)
    assert active.tolist() == [2]


# ── window_dataset ────────────────────────────────────────────────────────


def test_make_windows_shapes_and_alignment():
    from c2a.codegen.kinematics_vae.backbone.window_dataset import make_windows

    n_clips, n_frames, feat = 3, 20, 7
    motion = np.arange(n_clips * n_frames * feat, dtype=np.float32).reshape(
        n_clips, n_frames, feat,
    )
    w, n = 5, 2
    inputs, targets = make_windows(motion, w=w, n=n)

    n_w = n_frames - w - n + 1  # = 14
    assert inputs.shape == (n_clips * n_w, w, feat)
    assert targets.shape == (n_clips * n_w, n, feat)

    # First window of first clip = motion[0, 0:5]
    np.testing.assert_array_equal(inputs[0], motion[0, 0:w])
    # First target of first clip = motion[0, 5:7]
    np.testing.assert_array_equal(targets[0], motion[0, w:w + n])
    # Last window of second clip starts at frame n_frames - w - n.
    np.testing.assert_array_equal(
        inputs[2 * n_w - 1], motion[1, n_frames - w - n:n_frames - n],
    )


def test_make_windows_rejects_too_short_clips():
    from c2a.codegen.kinematics_vae.backbone.window_dataset import make_windows

    short = np.zeros((1, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="too short"):
        make_windows(short, w=10, n=5)


# ── normalizer ────────────────────────────────────────────────────────────


def test_normalizer_round_trip():
    from c2a.codegen.kinematics_vae.backbone.normalizer import fit_normalizer

    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 30, 5)).astype(np.float32) * 3 + 10
    norm = fit_normalizer(x)
    z = norm.apply(jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(z.mean(axis=(0, 1))), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(z.std(axis=(0, 1))), 1.0, atol=1e-2)
    back = norm.invert(z)
    np.testing.assert_allclose(np.asarray(back), x, atol=1e-4)


def test_normalizer_handles_zero_variance_dim():
    from c2a.codegen.kinematics_vae.backbone.normalizer import fit_normalizer

    x = np.zeros((10, 4, 3), dtype=np.float32)
    norm = fit_normalizer(x)
    # std floored to 1e-3 → no division by zero / NaN.
    assert float(jnp.min(norm.std)) >= 1e-3 - 1e-9
    out = norm.apply(jnp.asarray(x))
    assert not jnp.any(jnp.isnan(out))


# ── encoder / decoder / predictor shapes ──────────────────────────────────


def test_encoder_shape():
    from c2a.codegen.kinematics_vae.backbone.encoder import (
        MotionEncoderConv1DSharedSplit,
    )

    enc = MotionEncoderConv1DSharedSplit(branch_latent_dims=(4, 8, 12))
    x = jnp.zeros((3, 10, 31), dtype=jnp.float32)
    params = enc.init(jax.random.PRNGKey(0), x)
    mean, lv = enc.apply(params, x)
    assert mean.shape == (3, 24)
    assert lv.shape == (3, 24)


def test_decoder_predictor_shapes():
    from c2a.codegen.kinematics_vae.backbone.decoder import MotionDecoderSplit
    from c2a.codegen.kinematics_vae.backbone.predictor import MotionPredictorSplit

    branch_idx = ((0, 1, 2), (3, 4, 5, 6, 7), tuple(range(8, 15)))
    full = 15
    z = jnp.zeros((3, 4 + 8 + 12), dtype=jnp.float32)

    dec = MotionDecoderSplit(
        branch_output_indices=branch_idx,
        branch_latent_dims=(4, 8, 12),
        full_feat_dim=full, window_len=10,
    )
    p = dec.init(jax.random.PRNGKey(0), z)
    assert dec.apply(p, z).shape == (3, 10, full)

    pred = MotionPredictorSplit(
        branch_output_indices=branch_idx,
        branch_latent_dims=(4, 8, 12),
        full_feat_dim=full, horizon=5,
    )
    p2 = pred.init(jax.random.PRNGKey(1), z)
    assert pred.apply(p2, z).shape == (3, 5, full)


# ── losses ────────────────────────────────────────────────────────────────


def test_kl_to_unit_gaussian_zero_for_unit():
    from c2a.codegen.kinematics_vae.backbone.losses import kl_to_unit_gaussian

    mean = jnp.zeros((4, 8))
    logvar = jnp.zeros((4, 8))
    assert float(kl_to_unit_gaussian(mean, logvar)) == pytest.approx(0.0, abs=1e-6)


def test_masked_mse_unmasked_equals_mean():
    from c2a.codegen.kinematics_vae.backbone.losses import masked_mse

    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((3, 5)).astype(np.float32))
    b = jnp.zeros_like(a)
    expected = float(jnp.mean(a ** 2))
    assert float(masked_mse(a, b)) == pytest.approx(expected, abs=1e-5)


def test_sigma_cap_penalty_is_zero_under_cap():
    """If logvar is well below 2*log(sigma_max), the hinge is 0."""
    from c2a.codegen.kinematics_vae.backbone.losses import sigma_cap_penalty

    logvar = jnp.full((4, 8), -10.0, dtype=jnp.float32)  # sigma ≈ 0.0067 << 0.1
    pen = sigma_cap_penalty(logvar, sigma_max=0.1)
    assert float(pen) == pytest.approx(0.0, abs=1e-6)


# ── per-frame padding ─────────────────────────────────────────────────────


def test_pad_per_frame_left_right_replication():
    from c2a.codegen.kinematics_vae.backbone.codegen import _pad_per_frame

    n_clips = 2
    w = 10
    horizon = 5
    n_frames = 30
    n_windows = n_frames - w - horizon + 1  # = 16

    # Build distinguishable per-window outputs.
    windowed = np.arange(n_clips * n_windows, dtype=np.float32).reshape(
        n_clips, n_windows
    )[..., None]   # (2, 16, 1)
    padded = _pad_per_frame(
        windowed, n_frames=n_frames, window_len=w, horizon=horizon,
    )
    assert padded.shape == (n_clips, n_frames, 1)

    # Left edge (frames 0..w-2) should be repeats of window 0.
    for t in range(w - 1):
        np.testing.assert_array_equal(padded[:, t], windowed[:, 0])
    # Valid windows occupy frames [w-1, n_frames-horizon-1].
    for w_idx in range(n_windows):
        np.testing.assert_array_equal(padded[:, w - 1 + w_idx], windowed[:, w_idx])
    # Right edge (frames n_frames-horizon..n_frames-1) repeat last window.
    for t in range(n_frames - horizon, n_frames):
        np.testing.assert_array_equal(padded[:, t], windowed[:, -1])
