"""Load and align human behavior labels with KPMS syllable outputs.

Reads ``data/rodent/rodent_labels.csv`` (columns: ``Video, Behavior,
Start_Time, End_Time, Start_Frame, End_Frame``) and produces a per-clip
int-encoded frame array suitable for comparison against KPMS ``labels_list``.

Behaviors ``Left Turn`` and ``Right Turn`` are dropped entirely — frames
covered only by those intervals remain ``UNLABELED`` (sentinel ``-1``) and
are masked out when the coverage metrics are computed.

Frame alignment: CSV ``Start_Frame`` / ``End_Frame`` are consumed at face
value and clipped to ``[0, n_frames)``.  Intervals starting at or beyond
``n_frames`` are discarded.  Reported ``n_labeled_frames`` lets the caller
sanity-check what was usable.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Canonical 6-behavior vocabulary in alphabetical order (also the order that
# emerged from sort() of the unique labels in the CSV after dropping turns).
BEHAVIORS: tuple[str, ...] = (
    "Grooming",
    "Immobility",
    "Orienting",
    "Supported Rear",
    "Unsupported Rear",
    "Walking",
)
BEHAVIOR_TO_INT: dict[str, int] = {b: i for i, b in enumerate(BEHAVIORS)}
DROPPED_BEHAVIORS: frozenset[str] = frozenset({"Left Turn", "Right Turn"})
UNLABELED: int = -1  # sentinel for the `ignore` mask in NMI / purity.

_CLIP_RE = re.compile(r"clip_(\d+)\.mp4")


def load_behavior_frames(
    csv_path: str | Path,
    clip_range: tuple[int, int] = (642, 741),
    n_frames: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load behavior labels and project onto per-clip frame arrays.

    Args:
        csv_path: Path to the rodent labels CSV.
        clip_range: ``(low, high)`` inclusive clip-index range to include.
        n_frames: Number of frames per clip in the target KPMS space.

    Returns:
        ``(frame_labels, clip_indices, n_labeled_per_clip)`` where
        ``frame_labels`` has shape ``[N_clips_in_range, n_frames]`` int32,
        ``clip_indices`` is the sorted clip-index array of length
        ``N_clips_in_range``, and ``n_labeled_per_clip`` reports how many
        frames in each clip received a non-``UNLABELED`` label.

    Raises:
        KeyError: If a behavior string is not in ``BEHAVIOR_TO_INT`` and
            also not in ``DROPPED_BEHAVIORS``.
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {path}")

    low, high = int(clip_range[0]), int(clip_range[1])
    if low > high:
        raise ValueError(f"clip_range low > high: {clip_range}")

    clip_indices = np.arange(low, high + 1, dtype=np.int32)
    n_clips = int(clip_indices.size)
    clip_to_pos = {int(c): i for i, c in enumerate(clip_indices)}

    frame_labels = np.full((n_clips, n_frames), UNLABELED, dtype=np.int32)

    n_rows_total = 0
    n_rows_dropped_turn = 0
    n_rows_out_of_range = 0
    n_rows_beyond_nframes = 0
    n_rows_empty_interval = 0
    n_rows_applied = 0

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Video", "Behavior", "Start_Frame", "End_Frame"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"Labels CSV at {path} missing required columns: {sorted(missing)}"
            )

        for row in reader:
            n_rows_total += 1
            behavior = row["Behavior"].strip()

            if behavior in DROPPED_BEHAVIORS:
                n_rows_dropped_turn += 1
                continue
            if behavior not in BEHAVIOR_TO_INT:
                raise KeyError(
                    f"Unknown behavior {behavior!r} in {path} (row "
                    f"{n_rows_total}). Known: {sorted(BEHAVIOR_TO_INT)}, "
                    f"dropped: {sorted(DROPPED_BEHAVIORS)}"
                )

            m = _CLIP_RE.search(row["Video"])
            if m is None:
                raise ValueError(
                    f"Could not parse clip number from Video={row['Video']!r} "
                    f"(row {n_rows_total})"
                )
            clip_idx = int(m.group(1))
            if clip_idx not in clip_to_pos:
                n_rows_out_of_range += 1
                continue

            start = int(row["Start_Frame"])
            end = int(row["End_Frame"])
            if start >= n_frames:
                n_rows_beyond_nframes += 1
                continue
            end_clipped = min(end, n_frames)
            if end_clipped <= start:
                # Empty interval after clipping (e.g., End_Frame <= Start_Frame,
                # or a placeholder ``Start_Frame=End_Frame=0`` row).
                n_rows_empty_interval += 1
                continue

            pos = clip_to_pos[clip_idx]
            frame_labels[pos, start:end_clipped] = BEHAVIOR_TO_INT[behavior]
            n_rows_applied += 1

    n_labeled_per_clip = (frame_labels != UNLABELED).sum(axis=1).astype(np.int32)

    accounted = (
        n_rows_dropped_turn
        + n_rows_out_of_range
        + n_rows_beyond_nframes
        + n_rows_empty_interval
        + n_rows_applied
    )
    assert accounted == n_rows_total, (
        f"Row accounting mismatch: {accounted} != {n_rows_total}"
    )

    log.info(
        "load_behavior_frames: %d rows (dropped %d turns, %d out-of-range, "
        "%d beyond frame %d, %d empty intervals, applied %d). "
        "%d/%d clips have ≥1 labeled frame; median labeled frames/clip = %d.",
        n_rows_total,
        n_rows_dropped_turn,
        n_rows_out_of_range,
        n_rows_beyond_nframes,
        n_frames,
        n_rows_empty_interval,
        n_rows_applied,
        int((n_labeled_per_clip > 0).sum()),
        n_clips,
        int(np.median(n_labeled_per_clip)),
    )

    return frame_labels, clip_indices, n_labeled_per_clip
