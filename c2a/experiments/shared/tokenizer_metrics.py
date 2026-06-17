"""Per-tokenizer diagnostic metrics computed from per-frame code arrays.

Operates on a single ``(n_clips, n_frames)`` int32 array — no model
state, no checkpoint loading — so it's tokenizer-agnostic. KPMS, VAE,
VQ-VAE, PS-VQ-VAE all reduce to the same array shape after their
respective codegen passes.

Four metrics mirror the four panels in the user's reference image:

1. **Median duration** — distribution of run lengths (consecutive
   frames with the same code, across all clips and all codes).
2. **Frequency distribution** — per-code total frame count.
3. **Active syllables** — count of codes whose frame frequency exceeds
   a usage threshold (configurable; default any non-zero usage).
4. **Instances per syllable** — per-code count of distinct runs
   ("bouts"). Differs from #2 when codes have variable dwell time.

The framerate is needed only to convert frame counts to ms; the metric
math itself is unit-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenizerMetrics:
    """Summary of one ``(n_clips, n_frames)`` code array.

    Attributes:
        num_codes: Codebook size ``K`` (max code id + 1).
        n_clips: Number of clips in the source array.
        n_frames_total: Total frames (n_clips × n_frames).
        frame_frequency: ``(K,)`` int — frame count per code.
        bout_instances: ``(K,)`` int — count of distinct runs per code.
        bout_lengths: ``(num_bouts,)`` int — length (in frames) of every
            bout, flattened across clips. The mean / median / std of
            these IS the duration distribution.
        active_any: count of codes with frame frequency ≥ 1.
        active_at_1pct: count of codes whose frame frequency ≥ 1% of
            total frames.
        active_at_0p1pct: count of codes whose frame frequency ≥ 0.1%
            of total frames.
        mean_duration_frames: mean of bout_lengths.
        median_duration_frames: median of bout_lengths.
        std_duration_frames: std of bout_lengths.
        mean_duration_ms: ``mean_duration_frames * 1000 / fps``.
        median_duration_ms: same.
        n_bouts: total number of bouts (sum of bout_instances).
        instances_per_code_min: min over codes whose frame_frequency > 0.
            ``0`` if no codes are used.
        instances_per_code_median: median over codes whose
            frame_frequency > 0.
        instances_per_code_max: max over codes whose frame_frequency > 0.
    """

    num_codes: int
    n_clips: int
    n_frames_total: int

    frame_frequency: np.ndarray
    bout_instances: np.ndarray
    bout_lengths: np.ndarray

    active_any: int
    active_at_1pct: int
    active_at_0p1pct: int

    mean_duration_frames: float
    median_duration_frames: float
    std_duration_frames: float
    mean_duration_ms: float
    median_duration_ms: float

    n_bouts: int
    instances_per_code_min: int
    instances_per_code_median: float
    instances_per_code_max: int

    def to_table_row(self, family: str, config: str) -> dict:
        """One row for the active-syllables / instances tables."""
        return {
            "family": family,
            "config": config,
            "K": int(self.num_codes),
            "n_frames_total": int(self.n_frames_total),
            "active_any": int(self.active_any),
            "active_at_1pct": int(self.active_at_1pct),
            "active_at_0p1pct": int(self.active_at_0p1pct),
            "usage_ratio": float(self.active_any / max(self.num_codes, 1)),
            "n_bouts_total": int(self.n_bouts),
            "instances_per_code_min": int(self.instances_per_code_min),
            "instances_per_code_median": float(self.instances_per_code_median),
            "instances_per_code_max": int(self.instances_per_code_max),
            "mean_duration_frames": float(self.mean_duration_frames),
            "median_duration_frames": float(self.median_duration_frames),
            "std_duration_frames": float(self.std_duration_frames),
            "mean_duration_ms": float(self.mean_duration_ms),
            "median_duration_ms": float(self.median_duration_ms),
        }


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def compute_tokenizer_metrics(
    codes: np.ndarray,
    *,
    num_codes: int | None = None,
    fps: float = 50.0,
) -> TokenizerMetrics:
    """Compute the diagnostic-metric bundle for one tokenizer config.

    Args:
        codes: ``(n_clips, n_frames)`` int array. Negative values are
            treated as "no code" (KPMS sentinel) and excluded from
            all counts; they do NOT split bouts (i.e. consecutive
            frames with the same negative value are not bouts).
        num_codes: Override the codebook size ``K``. Defaults to
            ``max(codes) + 1`` (1-based) using non-negative values only.
            Specify explicitly when the codebook can have unused codes
            at the top end (e.g. KPMS K=30 but only 28 actually used).
        fps: Frames per second. Used only to convert durations to ms.

    Returns:
        :class:`TokenizerMetrics`.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(
            f"codes must be 2-D (n_clips, n_frames); got shape {codes.shape}"
        )
    if codes.size == 0:
        raise ValueError("codes is empty; cannot compute metrics.")

    if num_codes is None:
        nonneg = codes[codes >= 0]
        if nonneg.size == 0:
            raise ValueError(
                "codes has no non-negative entries; specify num_codes."
            )
        num_codes = int(nonneg.max() + 1)
    K = int(num_codes)

    # ------------------------------------------------------------------
    # Per-code frame frequency (#2 in the reference image)
    # ------------------------------------------------------------------
    flat = codes.ravel()
    valid_mask = flat >= 0
    frame_frequency = np.bincount(
        flat[valid_mask].astype(np.int64), minlength=K,
    )[:K].astype(np.int64)
    n_frames_total = int(valid_mask.sum())

    # ------------------------------------------------------------------
    # Per-code bout counts + bout-length list (#1 and #4 in the image)
    #
    # Bout = maximal run of consecutive frames with the same non-negative
    # code within one clip. We iterate clip-by-clip; cheaper than
    # vectorizing across the (n_clips, n_frames) boundary which would
    # need a per-clip break in the same-code mask.
    # ------------------------------------------------------------------
    bout_instances = np.zeros(K, dtype=np.int64)
    bout_lengths_list: list[int] = []

    for clip in codes:
        clip = np.asarray(clip, dtype=np.int64)
        if clip.size == 0:
            continue
        # Run-length encoding via change-point detection.
        # change[i] is True where clip[i] != clip[i-1]; clip[0] is always
        # a start.
        change = np.empty(clip.shape, dtype=bool)
        change[0] = True
        change[1:] = clip[1:] != clip[:-1]
        starts = np.where(change)[0]
        # Run values at each start; runs of -1 are skipped below.
        run_values = clip[starts]
        # Run ends = next start (or len(clip)); lengths = diffs.
        ends = np.empty(starts.shape, dtype=np.int64)
        ends[:-1] = starts[1:]
        ends[-1] = clip.shape[0]
        run_lengths = ends - starts

        for v, ln in zip(run_values, run_lengths):
            if v < 0:
                continue  # KPMS sentinel — not a real syllable
            if v >= K:
                continue  # Out of codebook; shouldn't happen
            bout_instances[v] += 1
            bout_lengths_list.append(int(ln))

    bout_lengths = np.asarray(bout_lengths_list, dtype=np.int64)
    n_bouts = int(bout_lengths.shape[0])

    # ------------------------------------------------------------------
    # Active-syllable counts (#3 in the image) — multiple thresholds
    # ------------------------------------------------------------------
    active_any = int((frame_frequency >= 1).sum())
    thresh_1pct = max(int(n_frames_total * 0.01), 1)
    thresh_0p1pct = max(int(n_frames_total * 0.001), 1)
    active_at_1pct = int((frame_frequency >= thresh_1pct).sum())
    active_at_0p1pct = int((frame_frequency >= thresh_0p1pct).sum())

    # ------------------------------------------------------------------
    # Duration stats
    # ------------------------------------------------------------------
    if n_bouts > 0:
        mean_d = float(np.mean(bout_lengths))
        median_d = float(np.median(bout_lengths))
        std_d = float(np.std(bout_lengths))
    else:
        mean_d = 0.0
        median_d = 0.0
        std_d = 0.0
    ms_per_frame = 1000.0 / float(fps)

    # ------------------------------------------------------------------
    # Instances-per-code summary (#4 in the image, as a table)
    # ------------------------------------------------------------------
    used_mask = frame_frequency > 0
    if used_mask.any():
        inst_used = bout_instances[used_mask]
        inst_min = int(inst_used.min())
        inst_median = float(np.median(inst_used))
        inst_max = int(inst_used.max())
    else:
        inst_min, inst_median, inst_max = 0, 0.0, 0

    return TokenizerMetrics(
        num_codes=K,
        n_clips=int(codes.shape[0]),
        n_frames_total=n_frames_total,
        frame_frequency=frame_frequency,
        bout_instances=bout_instances,
        bout_lengths=bout_lengths,
        active_any=active_any,
        active_at_1pct=active_at_1pct,
        active_at_0p1pct=active_at_0p1pct,
        mean_duration_frames=mean_d,
        median_duration_frames=median_d,
        std_duration_frames=std_d,
        mean_duration_ms=mean_d * ms_per_frame,
        median_duration_ms=median_d * ms_per_frame,
        n_bouts=n_bouts,
        instances_per_code_min=inst_min,
        instances_per_code_median=inst_median,
        instances_per_code_max=inst_max,
    )
