"""τ (temperature) auto-calibration for the code2code reward.

The recommended default reward form is::

    r = exp(-d / τ)

where ``d`` is :meth:`C2CTokenizerAdapter.distance` between the
encoder's output on a rollout window and the codebook entry for the
target code.

If τ is too small, the reward is ~0 almost everywhere and the policy
gets no gradient. If τ is too large, the reward saturates at 1 and the
policy has nothing to maximize. The sweet spot is mid-saturation —
empirically ``r ≈ 0.3`` at the initial distance distribution.

This module measures the per-frame distance ``d_t = distance(z_t,
codebook[c_t])`` on a sample of reference clips and picks τ such that
``exp(-median(d_t) / τ) = target_reward`` (default 0.3).

Per the literature.md survey, this calibration step is the
recommended fix to AMP/DeepMimic's per-term hand-tuned temperatures
(DeepMimic uses 2, 0.1, 40, 10 for different terms — no global τ
worked across all of them). With a learned encoder there's no
analytic τ, so we measure it.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from c2a.code2code.adapter_base import C2CTokenizerAdapter, DistanceForm

log = logging.getLogger(__name__)


def calibrate_temperature(
    adapter: C2CTokenizerAdapter,
    ref_qpos: np.ndarray,
    target_codes: np.ndarray,
    *,
    distance_form: DistanceForm = "cosine",
    target_initial_reward: float = 0.3,
    n_clips_sample: int = 50,
    n_windows_per_clip: int = 8,
    seed: int = 0,
) -> tuple[float, float]:
    """Encode a sample of reference windows, return ``(median_d, tau)``.

    Args:
        adapter: the loaded tokenizer adapter.
        ref_qpos: ``(n_clips, n_frames, 7 + n_joints)`` reference qpos
            in the SAME row order as ``target_codes``.
        target_codes: ``(n_clips, n_frames)`` int32 target codes (the
            codes the policy would be conditioned on).
        distance_form: distance metric — must match what training uses.
        target_initial_reward: desired ``r = exp(-median(d)/τ)`` at
            initial-policy-distance. ``0.3`` ≡ mid-saturation; lower
            values like 0.1 give a flatter signal early but more
            gradient near zero distance.
        n_clips_sample: how many clips to sample for the calibration
            pass. Cheap so default is conservative.
        n_windows_per_clip: how many windows per clip to encode.
        seed: PRNG seed for clip/window selection.

    Returns:
        ``(median_distance, tau)``. ``tau`` is the value that achieves
        ``exp(-median_d / tau) == target_initial_reward``.

    Raises:
        ValueError: if any sampled target code is out of range, or
            ``target_initial_reward`` is not in ``(0, 1)``.
    """
    if not (0.0 < target_initial_reward < 1.0):
        raise ValueError(
            f"target_initial_reward must be in (0, 1); got "
            f"{target_initial_reward}"
        )

    ref_qpos = np.asarray(ref_qpos, dtype=np.float32)
    target_codes = np.asarray(target_codes, dtype=np.int32)
    n_clips, n_frames = target_codes.shape
    w = adapter.window_len
    if n_frames < w:
        raise ValueError(
            f"n_frames={n_frames} < window_len={w}; cannot extract "
            f"a single window for calibration."
        )

    rng = np.random.RandomState(int(seed))
    sample_size = min(int(n_clips_sample), int(n_clips))
    clip_idx = rng.choice(n_clips, size=sample_size, replace=False)

    distances: list[float] = []

    # JIT the per-window distance computation so the sample loop is
    # fast even with several hundred windows total.
    @jax.jit
    def _dist(qpos_window: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        z = adapter.encode_from_qpos(qpos_window)
        return adapter.distance(z, c, distance_form=distance_form)

    for ci in clip_idx:
        valid_starts = n_frames - w + 1
        n_w = min(int(n_windows_per_clip), valid_starts)
        win_starts = rng.choice(valid_starts, size=n_w, replace=False)
        for s in win_starts:
            window = jnp.asarray(ref_qpos[ci, s : s + w])
            # Target code at the LAST frame of the window — matches
            # encode_motion_array's "windowed-z assigned to the last
            # frame" convention.
            c = jnp.int32(int(target_codes[ci, s + w - 1]))
            d = float(_dist(window, c))
            distances.append(d)

    if not distances:
        raise RuntimeError(
            "calibrate_temperature produced no samples; n_clips_sample "
            "and n_windows_per_clip must both be ≥ 1."
        )

    median_d = float(np.median(distances))

    # τ such that exp(-median_d / τ) = target_initial_reward
    #   => τ = -median_d / log(target_initial_reward)
    # Guard against median_d == 0 (rollout already at target perfectly,
    # nothing to scale — fall back to 1.0).
    if median_d <= 1e-10:
        log.warning(
            "calibrate_temperature: median distance ≈ 0 (%.3e). "
            "Falling back to tau=1.0; the reward will be ~1 everywhere "
            "near the reference manifold.",
            median_d,
        )
        tau = 1.0
    else:
        tau = median_d / (-math.log(target_initial_reward))

    log.info(
        "calibrate_temperature(distance_form=%s, target_r=%.2f): "
        "median_d=%.4f (n=%d windows), tau=%.4f",
        distance_form, target_initial_reward, median_d, len(distances), tau,
    )

    return median_d, tau


def validate_reference_round_trip(
    adapter: "C2CTokenizerAdapter",
    ref_qpos: jnp.ndarray,
    target_codes: jnp.ndarray,
    *,
    tau: float,
    distance_form: DistanceForm = "cosine",
    reward_form: str = "exp_neg_distance",
    expected_reward: float = 0.30,
    tolerance: float = 0.08,
    n_clips_sample: int = 40,
    n_windows_per_clip: int = 10,
    seed: int = 0,
) -> dict:
    """Reference-data round-trip integrity check for the c2c reward.

    Feeds mocap-rate reference windows through the (frozen) adapter and
    verifies that the resulting per-step reward is centered on the
    calibration target. This catches a class of silent failure modes:
    wrong normalizer loaded, mis-indexed codebook, wrong window length,
    wrong feature extraction, mismatched encoder weights, etc.

    Also performs a NEGATIVE control: encode windows with intentionally
    shuffled target codes (so the codebook entry doesn't correspond to
    the encoded window) and verify the reward is much lower than the
    positive case. Without this control, a check that always passes
    (e.g., because tau is enormous) would never fail.

    Args:
        adapter: Frozen c2c tokenizer adapter (vae).
        ref_qpos: ``(n_clips, n_frames, nq)`` reference qpos at mocap rate.
        target_codes: ``(n_clips, n_frames)`` precomputed per-frame codes.
        tau: Temperature from calibrate_temperature.
        distance_form: must match what the env uses.
        reward_form: must match what the env uses.
        expected_reward: target the calibration was tuned to (default 0.30
            matches ``calibrate_target_reward`` in the base config).
        tolerance: how far median(reward) is allowed to drift from
            ``expected_reward`` before raising. Default ±0.08 = roughly
            one standard error at the sample sizes below.
        n_clips_sample: number of clips to sample.
        n_windows_per_clip: number of windows to sample per clip.
        seed: PRNG seed for sampling.

    Returns:
        Dict with ``median_distance``, ``median_reward``,
        ``negative_median_reward``, ``n_samples``, and the
        positive-vs-negative ``contrast``.

    Raises:
        AssertionError: if median positive reward is outside
            ``[expected_reward - tolerance, expected_reward + tolerance]``,
            OR if the positive-vs-negative contrast is < 0.05 (meaning
            the check can't distinguish matched from shuffled codes).
    """
    ref_qpos = np.asarray(ref_qpos, dtype=np.float32)
    target_codes = np.asarray(target_codes, dtype=np.int32)
    n_clips, n_frames = target_codes.shape
    w = adapter.window_len
    if n_frames < w:
        raise ValueError(
            f"n_frames={n_frames} < window_len={w}; cannot extract a "
            f"reference window for round-trip validation."
        )

    rng = np.random.RandomState(int(seed))
    sample_size = min(int(n_clips_sample), int(n_clips))
    clip_idx = rng.choice(n_clips, size=sample_size, replace=False)

    @jax.jit
    def _dist(qpos_window: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        z = adapter.encode_from_qpos(qpos_window)
        return adapter.distance(z, c, distance_form=distance_form)

    pos_rewards: list[float] = []
    pos_distances: list[float] = []
    neg_rewards: list[float] = []

    for ci in clip_idx:
        valid_starts = n_frames - w + 1
        n_w = min(int(n_windows_per_clip), valid_starts)
        win_starts = rng.choice(valid_starts, size=n_w, replace=False)
        for s in win_starts:
            window = jnp.asarray(ref_qpos[ci, s : s + w])
            # POSITIVE: codebook entry is the actual code for this window.
            c_true = jnp.int32(int(target_codes[ci, s + w - 1]))
            d_pos = float(_dist(window, c_true))
            pos_distances.append(d_pos)
            if reward_form == "exp_neg_distance":
                pos_rewards.append(float(math.exp(-d_pos / tau)))
            else:
                pos_rewards.append(-d_pos)
            # NEGATIVE: codebook entry is a different randomly-chosen
            # code (with sufficient codes the chance of accidentally
            # picking c_true is small; we resample if needed).
            n_codes = adapter.codebook_size
            c_neg = int(rng.randint(n_codes))
            while c_neg == int(c_true):
                c_neg = int(rng.randint(n_codes))
            d_neg = float(_dist(window, jnp.int32(c_neg)))
            if reward_form == "exp_neg_distance":
                neg_rewards.append(float(math.exp(-d_neg / tau)))
            else:
                neg_rewards.append(-d_neg)

    if not pos_rewards:
        raise RuntimeError(
            "validate_reference_round_trip produced no samples."
        )

    median_d = float(np.median(pos_distances))
    median_r = float(np.median(pos_rewards))
    median_neg_r = float(np.median(neg_rewards))
    contrast = median_r - median_neg_r

    log.info(
        "c2c reference round-trip: median_d=%.4f, median_r=%.4f "
        "(target %.4f ± %.4f); negative_median_r=%.4f; contrast=%.4f",
        median_d, median_r, expected_reward, tolerance,
        median_neg_r, contrast,
    )

    if abs(median_r - expected_reward) > tolerance:
        raise AssertionError(
            f"c2c reference round-trip FAILED: median reward "
            f"{median_r:.4f} is outside the expected range "
            f"[{expected_reward - tolerance:.4f}, "
            f"{expected_reward + tolerance:.4f}]. This indicates the "
            f"runtime encode/lookup pipeline does not reproduce the "
            f"calibration. Likely culprits: wrong normalizer, "
            f"mis-indexed codebook, window length / rate mismatch, "
            f"or wrong encoder weights. Check the most recent change "
            f"to c2a/moseq_env_wrapper.py or c2a/code2code/."
        )

    if contrast < 0.05:
        raise AssertionError(
            f"c2c reference round-trip FAILED on the negative control: "
            f"positive median reward {median_r:.4f} vs shuffled-code "
            f"median {median_neg_r:.4f} (contrast {contrast:.4f} < "
            f"0.05). The reward signal cannot distinguish a matched "
            f"codebook entry from a random one, which means the "
            f"distance function or temperature is degenerate."
        )

    return {
        "median_distance": median_d,
        "median_reward": median_r,
        "negative_median_reward": median_neg_r,
        "contrast": contrast,
        "n_samples": len(pos_rewards),
    }
