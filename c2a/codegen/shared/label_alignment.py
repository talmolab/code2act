"""Load and align human behavior labels with KPMS syllable outputs.

Schema (as of 2026-05-06)
-------------------------
The CSV at ``data/rodent/rodent_labels.csv`` has **two parallel annotation
tracks** that together define a 4 × 3 = 12-cell label space:

* ``behavior``  ∈ {Immobile, Turn, Walk, Rear}
* ``direction`` ∈ {Straight, Left, Right}

Each row covers a contiguous frame range ``[Start_Frame, End_Frame)`` in
**CSV frames at 30 fps** (500 frames per 16.67-second clip).  The two
tracks run concurrently — at any given frame the rodent is labeled with
both a behavior *and* a direction.  Combining them yields 12 fine labels,
but only ~7 cells are populated in practice (Immobile is always Straight,
Turn is never Straight, Walk is overwhelmingly Straight).  The loader
exposes all three views (behavior / direction / combined) and the caller
chooses the granularity.

Subsample factor (CSV → H5 frames)
----------------------------------
The CSV is annotated against the original 30 fps videos (500 frames / clip).
The mocap H5 (``rodent_reference_clips.h5``) stores 250 frames per clip —
half the rate.  CSV frame ``F`` therefore maps to H5 frame ``F // 2`` by
default (``csv_to_h5_subsample=2``).  Set ``csv_to_h5_subsample=1`` if a
future CSV is regenerated at the H5 rate.

Behavior labels are sourced from the ``TopoMimic-annotation`` repository:
https://github.com/KevinBian107/TopoMimic-annotation

Public API
----------
* :func:`load_label_frames` — returns a :class:`LabelArrays` NamedTuple
  with ``behavior`` (4-class), ``direction`` (3-class), and ``combined``
  (12-class) per-frame integer arrays + clip indices + per-clip counts.
* :func:`load_behavior_frames` — backward-compatible thin wrapper that
  returns just the behavior-track view as ``(frame_labels, clip_indices,
  n_labeled_per_clip)``.  Used by older callers (``c2a.codegen.run_sweep``).

Sentinel
--------
``UNLABELED = -1`` — frames not covered by any track-row are masked.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

log = logging.getLogger(__name__)


# Canonical 4-behavior vocabulary, alphabetically ordered.
BEHAVIORS: tuple[str, ...] = ("Immobile", "Rear", "Turn", "Walk")

# Canonical 3-direction vocabulary, alphabetically ordered.
DIRECTIONS: tuple[str, ...] = ("Left", "Right", "Straight")

# 12 combined labels in row-major order: behavior × direction.
COMBINED_LABELS: tuple[str, ...] = tuple(
    f"{b}_{d}" for b in BEHAVIORS for d in DIRECTIONS
)

BEHAVIOR_TO_INT: dict[str, int] = {b: i for i, b in enumerate(BEHAVIORS)}
DIRECTION_TO_INT: dict[str, int] = {d: i for i, d in enumerate(DIRECTIONS)}
COMBINED_TO_INT: dict[str, int] = {c: i for i, c in enumerate(COMBINED_LABELS)}

# 5-class scheme used by ``c2a.codegen.code_sources.SOURCE_GT_DIRECTIONAL_TURN``
# and by ``c2a.experiments.run_labeled_hidden_dynamics`` when configured
# for a 5-class breakdown. Direction is collapsed for non-Turn behaviors:
#
#   Immobile (any direction) -> 0
#   Rear     (any direction) -> 1
#   Walk     (any direction) -> 2
#   Turn + Left              -> 3
#   Turn + Right             -> 4
#   Turn + Straight          -> UNLABELED  (not present in the current CSV)
#
# Defined here (not in code_sources.py) so analysis scripts can import
# it without pulling the GT-codes loader / training-only machinery.
DIRECTIONAL_TURN_LABELS: tuple[str, ...] = (
    "Immobile",
    "Rear",
    "Walk",
    "Turn_Left",
    "Turn_Right",
)
DIRECTIONAL_TURN_TO_INT: dict[str, int] = {
    n: i for i, n in enumerate(DIRECTIONAL_TURN_LABELS)
}

UNLABELED: int = -1

# Track names recognized in the CSV's ``Track`` column.
_TRACK_BEHAVIOR = "behavior"
_TRACK_DIRECTION = "direction"

_CLIP_RE = re.compile(r"clip_(\d+)\.mp4")


class LabelArrays(NamedTuple):
    """Per-clip frame labels at three granularities.

    Attributes:
        behavior: ``(N_clips, n_frames)`` int32, values in
            ``{-1, 0, 1, 2, 3}`` corresponding to UNLABELED + ``BEHAVIORS``.
        direction: ``(N_clips, n_frames)`` int32, values in
            ``{-1, 0, 1, 2}`` corresponding to UNLABELED + ``DIRECTIONS``.
        combined: ``(N_clips, n_frames)`` int32, values in
            ``{-1, 0..11}``.  Computed as ``behavior * 3 + direction``
            iff both tracks label the frame, else UNLABELED.  Indices
            line up with :data:`COMBINED_LABELS`.
        clip_indices: ``(N_clips,)`` int — clip ids included.
        n_labeled_per_clip: ``(N_clips,)`` int — frames with a *combined*
            label present.  (A frame can be behavior-labeled but
            direction-unlabeled if the CSV is malformed; in practice the
            current CSV labels every frame in both tracks.)
    """

    behavior: np.ndarray
    direction: np.ndarray
    combined: np.ndarray
    clip_indices: np.ndarray
    n_labeled_per_clip: np.ndarray


def directional_turn_array(arrays: LabelArrays) -> np.ndarray:
    """Project ``(behavior, direction)`` onto the 5-class directional-turn enum.

    Class mapping (same as :data:`DIRECTIONAL_TURN_LABELS`):

        Immobile (any direction)  -> 0
        Rear     (any direction)  -> 1
        Walk     (any direction)  -> 2
        Turn + Left               -> 3
        Turn + Right              -> 4
        Turn + Straight           -> UNLABELED  (absent in current CSV)
        Turn + UNLABELED          -> UNLABELED  (cannot disambiguate L/R)
        UNLABELED behavior        -> UNLABELED

    Only the ``Turn`` class actually depends on the ``direction`` track —
    ``Walk`` / ``Rear`` / ``Immobile`` are direction-independent and pass
    through whenever ``behavior`` is labeled, even if ``direction`` is
    UNLABELED at that frame. (Direction annotations in the CSV are
    sparser than behavior annotations, especially at early-clip frames;
    suppressing direction-independent classes whenever direction is
    missing would needlessly collapse most early-clip Walk/Rear/Immobile
    frames to UNLABELED.)

    Args:
        arrays: ``LabelArrays`` from :func:`load_label_frames`.

    Returns:
        ``(N_clips, n_frames)`` int32 with values in
        ``{-1, 0, 1, 2, 3, 4}``. Indices align with
        :data:`DIRECTIONAL_TURN_LABELS`.
    """
    behavior = arrays.behavior
    direction = arrays.direction
    out = np.full(behavior.shape, UNLABELED, dtype=np.int32)

    # Direction-independent classes — only need behavior to be labeled.
    for name in ("Immobile", "Rear", "Walk"):
        mask = behavior == BEHAVIOR_TO_INT[name]
        out[mask] = DIRECTIONAL_TURN_TO_INT[name]

    # Turn — needs direction to disambiguate Left vs Right.
    # Turn + Straight, and Turn + UNLABELED, both stay UNLABELED.
    turn = behavior == BEHAVIOR_TO_INT["Turn"]
    out[turn & (direction == DIRECTION_TO_INT["Left"])] = (
        DIRECTIONAL_TURN_TO_INT["Turn_Left"]
    )
    out[turn & (direction == DIRECTION_TO_INT["Right"])] = (
        DIRECTIONAL_TURN_TO_INT["Turn_Right"]
    )
    return out


def _parse_clip_id(video_field: str, row_idx: int) -> int:
    m = _CLIP_RE.search(video_field)
    if m is None:
        raise ValueError(
            f"Could not parse clip number from Video={video_field!r} "
            f"(row {row_idx})"
        )
    return int(m.group(1))


def _accumulate_intervals(
    rows: list[dict[str, str]],
    track_name: str,
    label_to_int: dict[str, int],
    clip_to_pos: dict[int, int],
    n_frames_h5: int,
    subsample: int,
    arr: np.ndarray,
) -> dict[str, int]:
    """Project track-rows of one track type onto ``arr[N_clips, n_frames_h5]``.

    Returns row-accounting counters for sanity-check assertions.
    """
    counters = {
        "wrong_track": 0,
        "out_of_range_clip": 0,
        "beyond_nframes": 0,
        "empty_interval": 0,
        "applied": 0,
    }

    for row_idx, row in enumerate(rows, start=1):
        if row["Track"].strip() != track_name:
            counters["wrong_track"] += 1
            continue

        label = row["Label"].strip()
        if label not in label_to_int:
            raise KeyError(
                f"Unknown {track_name} label {label!r} at row {row_idx}. "
                f"Known: {sorted(label_to_int)}"
            )

        clip_id = _parse_clip_id(row["Video"], row_idx)
        if clip_id not in clip_to_pos:
            counters["out_of_range_clip"] += 1
            continue

        # CSV frames at 30 fps → H5 frames at 30/subsample fps.
        s_csv = int(row["Start_Frame"])
        e_csv = int(row["End_Frame"])
        s = s_csv // subsample
        e = e_csv // subsample

        if s >= n_frames_h5:
            counters["beyond_nframes"] += 1
            continue
        e = min(e, n_frames_h5)
        if e <= s:
            counters["empty_interval"] += 1
            continue

        pos = clip_to_pos[clip_id]
        arr[pos, s:e] = label_to_int[label]
        counters["applied"] += 1

    return counters


def load_label_frames(
    csv_path: str | Path,
    clip_range: tuple[int, int] = (0, 841),
    n_frames: int = 250,
    csv_to_h5_subsample: int = 2,
) -> LabelArrays:
    """Load behavior/direction/combined frame labels for a clip range.

    Args:
        csv_path: Path to the rodent labels CSV (two-track schema).
        clip_range: ``(low, high)`` inclusive clip-index range to include.
            Defaults to the full corpus ``(0, 841)``.
        n_frames: H5 frames per clip (default 250 — matches
            ``rodent_reference_clips.h5``).
        csv_to_h5_subsample: CSV frame number → H5 frame number divisor.
            Default 2 (CSV at 30 fps, H5 at 15 fps for our data).

    Returns:
        A :class:`LabelArrays` with ``behavior`` / ``direction`` /
        ``combined`` arrays of shape ``(N_clips, n_frames)`` plus
        bookkeeping.

    Raises:
        FileNotFoundError: csv_path does not exist.
        KeyError: required CSV column missing or unknown label string.
        ValueError: malformed Video field, or ``clip_range`` inverted.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    low, high = int(clip_range[0]), int(clip_range[1])
    if low > high:
        raise ValueError(f"clip_range low > high: {clip_range}")
    if csv_to_h5_subsample < 1:
        raise ValueError(f"csv_to_h5_subsample must be >= 1, got {csv_to_h5_subsample}")

    clip_indices = np.arange(low, high + 1, dtype=np.int32)
    n_clips = int(clip_indices.size)
    clip_to_pos = {int(c): i for i, c in enumerate(clip_indices)}

    behavior = np.full((n_clips, n_frames), UNLABELED, dtype=np.int32)
    direction = np.full((n_clips, n_frames), UNLABELED, dtype=np.int32)

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Video", "Track", "Label", "Start_Frame", "End_Frame"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"Labels CSV at {path} missing required columns: {sorted(missing)}. "
                f"This module expects the two-track schema (Track ∈ "
                f"{{behavior, direction}}). For the legacy single-Behavior CSV, "
                f"check out an older revision of c2a/kpms/label_alignment.py."
            )
        rows = list(reader)

    n_total_rows = len(rows)

    cb = _accumulate_intervals(
        rows, _TRACK_BEHAVIOR, BEHAVIOR_TO_INT,
        clip_to_pos, n_frames, csv_to_h5_subsample, behavior,
    )
    cd = _accumulate_intervals(
        rows, _TRACK_DIRECTION, DIRECTION_TO_INT,
        clip_to_pos, n_frames, csv_to_h5_subsample, direction,
    )

    # Sanity check: every CSV row should belong to exactly one of the two tracks
    # (i.e. wrong_track[behavior] + applied[behavior] + ... == n_total_rows for
    # each track's pass; equivalently, the sum across all "non-skipped" rows
    # plus wrong_track must reconcile.).  We don't strict-assert here because
    # exotic Track values would just be silently skipped — log instead.
    seen_track = (n_total_rows - cb["wrong_track"]) + (n_total_rows - cd["wrong_track"])
    if seen_track != n_total_rows:
        log.warning(
            "Track-row accounting mismatch: %d rows seen as behavior or direction, "
            "%d total CSV rows.", seen_track, n_total_rows,
        )

    # Combined label: per-frame, encode (behavior_idx * 3 + direction_idx).
    # Mask cells where either track is UNLABELED.
    n_dir = len(DIRECTIONS)
    both_labeled = (behavior != UNLABELED) & (direction != UNLABELED)
    combined = np.full((n_clips, n_frames), UNLABELED, dtype=np.int32)
    combined[both_labeled] = (
        behavior[both_labeled] * n_dir + direction[both_labeled]
    )

    n_labeled_per_clip = (combined != UNLABELED).sum(axis=1).astype(np.int32)

    log.info(
        "load_label_frames: %d CSV rows; behavior applied=%d (drops: track=%d, "
        "clip_oor=%d, beyond=%d, empty=%d); direction applied=%d (drops: track=%d, "
        "clip_oor=%d, beyond=%d, empty=%d). %d/%d clips fully covered (combined "
        "label on every H5 frame); median combined-labeled frames/clip = %d.",
        n_total_rows,
        cb["applied"], cb["wrong_track"], cb["out_of_range_clip"],
        cb["beyond_nframes"], cb["empty_interval"],
        cd["applied"], cd["wrong_track"], cd["out_of_range_clip"],
        cd["beyond_nframes"], cd["empty_interval"],
        int((n_labeled_per_clip == n_frames).sum()), n_clips,
        int(np.median(n_labeled_per_clip)),
    )

    return LabelArrays(
        behavior=behavior,
        direction=direction,
        combined=combined,
        clip_indices=clip_indices,
        n_labeled_per_clip=n_labeled_per_clip,
    )


def load_behavior_frames(
    csv_path: str | Path,
    clip_range: tuple[int, int] = (0, 841),
    n_frames: int = 250,
    csv_to_h5_subsample: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compat: return only the behavior-track view.

    Returns ``(behavior_frames, clip_indices, n_labeled_per_clip)``
    matching the old API.  The 4-class behavior vocabulary is now
    ``BEHAVIORS = (Immobile, Rear, Turn, Walk)`` — the previous
    6-class vocabulary (Grooming/Immobility/Orienting/Supported Rear/
    Unsupported Rear/Walking) is gone with the new CSV schema.
    Existing callers depending on those exact strings must migrate.
    """
    arr = load_label_frames(
        csv_path, clip_range=clip_range, n_frames=n_frames,
        csv_to_h5_subsample=csv_to_h5_subsample,
    )
    n_per_clip = (arr.behavior != UNLABELED).sum(axis=1).astype(np.int32)
    return arr.behavior, arr.clip_indices, n_per_clip
