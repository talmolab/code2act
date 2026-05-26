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
