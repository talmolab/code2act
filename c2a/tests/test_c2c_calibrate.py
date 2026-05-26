"""Tests for c2a.code2code.calibrate.calibrate_temperature."""

import math
from pathlib import Path

import numpy as np
import pytest

_VQVAE_CKPT = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/vqvae/step_0030000"
)
_VQVAE_BEST_CODES = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/vqvae/best_codes.npz"
)
_REF_H5 = Path(
    "/home/jovyan/vast/kaiwen/code2act/data/rodent/rodent_reference_clips.h5"
)


def _require_artifacts():
    if not _VQVAE_CKPT.exists():
        pytest.skip(f"VQ-VAE ckpt not available: {_VQVAE_CKPT}")
    if not _VQVAE_BEST_CODES.exists():
        pytest.skip(f"VQ-VAE best_codes not available: {_VQVAE_BEST_CODES}")
    if not _REF_H5.exists():
        pytest.skip(f"reference clips not available: {_REF_H5}")


def _load_ref_data(clip_length: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Load (n_clips, clip_length, nq) qpos + matching codes."""
    import h5py
    with h5py.File(_REF_H5, "r") as f:
        flat = f["qpos"][:]
    n_total = flat.shape[0] // clip_length
    qpos = flat[: n_total * clip_length].reshape(n_total, clip_length, -1).astype(np.float32)

    data = np.load(_VQVAE_BEST_CODES, allow_pickle=False)
    # all_codes: (n_total, clip_length) aligned to global clip ids in
    # sorted-union order. For the calibration test we need codes whose
    # row index matches `qpos` row index — that's exactly `all_codes`.
    all_codes = data["all_codes"].astype(np.int32)
    return qpos, all_codes


def test_calibrate_returns_sensible_tau():
    """On VQ-VAE codes paired with their own reference clips, the
    median in-clip distance should be small (the rollout IS the
    reference here, so the encoder lands close to its own codebook
    entries). τ at target_r=0.3 should be small but positive."""
    _require_artifacts()
    from c2a.code2code import VQVAEAdapter
    from c2a.code2code.calibrate import calibrate_temperature

    qpos, all_codes = _load_ref_data(clip_length=250)
    adapter = VQVAEAdapter.from_ckpt_dir(
        _VQVAE_CKPT, n_joints=67, use_qvel=False,
        window_len=10, horizon=5,
    )

    # Use a smaller subset for a quick test.
    median_d, tau = calibrate_temperature(
        adapter,
        ref_qpos=qpos[:20],
        target_codes=all_codes[:20],
        distance_form="cosine",
        target_initial_reward=0.3,
        n_clips_sample=20,
        n_windows_per_clip=4,
        seed=0,
    )

    assert median_d >= 0.0
    assert tau > 0.0
    # Sanity: exp(-median_d / tau) should ≈ 0.3 by construction.
    achieved_r = math.exp(-median_d / tau)
    assert abs(achieved_r - 0.3) < 1e-4, (
        f"calibration target was 0.3, got {achieved_r} (median_d={median_d}, "
        f"tau={tau})"
    )


def test_calibrate_target_reward_range():
    """target_initial_reward outside (0, 1) should raise."""
    _require_artifacts()
    from c2a.code2code import VQVAEAdapter
    from c2a.code2code.calibrate import calibrate_temperature

    qpos, all_codes = _load_ref_data(clip_length=250)
    adapter = VQVAEAdapter.from_ckpt_dir(
        _VQVAE_CKPT, n_joints=67, use_qvel=False,
        window_len=10, horizon=5,
    )

    for bad in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError, match="target_initial_reward"):
            calibrate_temperature(
                adapter, qpos[:5], all_codes[:5],
                target_initial_reward=bad,
                n_clips_sample=2, n_windows_per_clip=1,
            )


def test_calibrate_l2_distinct_from_cosine():
    """L2 and cosine return different median distances (sanity that
    the distance_form argument is plumbed through)."""
    _require_artifacts()
    from c2a.code2code import VQVAEAdapter
    from c2a.code2code.calibrate import calibrate_temperature

    qpos, all_codes = _load_ref_data(clip_length=250)
    adapter = VQVAEAdapter.from_ckpt_dir(
        _VQVAE_CKPT, n_joints=67, use_qvel=False,
        window_len=10, horizon=5,
    )

    med_l2, _ = calibrate_temperature(
        adapter, qpos[:10], all_codes[:10],
        distance_form="l2", n_clips_sample=10, n_windows_per_clip=2,
    )
    med_cos, _ = calibrate_temperature(
        adapter, qpos[:10], all_codes[:10],
        distance_form="cosine", n_clips_sample=10, n_windows_per_clip=2,
    )
    # L2 and cosine medians have different units; just confirm they're
    # both positive and not numerically identical.
    assert med_l2 > 0.0
    assert med_cos > 0.0
    assert abs(med_l2 - med_cos) > 1e-6, (
        f"L2 and cosine medians too close: {med_l2}, {med_cos}"
    )
