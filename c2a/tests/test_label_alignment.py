"""Tests for ``c2a.codegen.shared.label_alignment`` — two-track schema.

The CSV at ``data/rodent/rodent_labels.csv`` has two parallel tracks
(``behavior`` and ``direction``); the loader returns three views (4-class
behavior, 3-class direction, 12-class combined) plus bookkeeping. CSV
frames are at 30 fps, H5 frames at 15 fps — default subsample factor 2.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from c2a.codegen.shared.label_alignment import (
    BEHAVIOR_TO_INT,
    BEHAVIORS,
    COMBINED_LABELS,
    COMBINED_TO_INT,
    DIRECTION_TO_INT,
    DIRECTIONS,
    UNLABELED,
    load_behavior_frames,
    load_label_frames,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


_FIELDS = ["Video", "Track", "Label", "Start_Time", "End_Time", "Start_Frame", "End_Frame"]


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "labels.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in _FIELDS})
    return p


def _b(video: str, label: str, s: int, e: int) -> dict:
    return {"Video": video, "Track": "behavior", "Label": label,
            "Start_Frame": s, "End_Frame": e}


def _d(video: str, label: str, s: int, e: int) -> dict:
    return {"Video": video, "Track": "direction", "Label": label,
            "Start_Frame": s, "End_Frame": e}


# ── Vocabulary invariants ────────────────────────────────────────────────────


def test_vocabulary_sizes():
    assert len(BEHAVIORS) == 4
    assert len(DIRECTIONS) == 3
    assert len(COMBINED_LABELS) == 12
    assert set(BEHAVIORS) == {"Immobile", "Rear", "Turn", "Walk"}
    assert set(DIRECTIONS) == {"Left", "Right", "Straight"}


def test_combined_index_encoding():
    """combined index = behavior * 3 + direction."""
    for b_idx, b in enumerate(BEHAVIORS):
        for d_idx, d in enumerate(DIRECTIONS):
            combined_str = f"{b}_{d}"
            assert COMBINED_TO_INT[combined_str] == b_idx * 3 + d_idx


# ── Subsample mapping (CSV → H5) ─────────────────────────────────────────────


def test_default_subsample_2_maps_csv_500_to_h5_250(tmp_path):
    """A CSV interval [0, 500) at 30 fps → H5 [0, 250) at 15 fps."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Immobile", 0, 500),
        _d("clip_0.mp4", "Straight", 0, 500),
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)
    assert arr.behavior.shape == (1, 250)
    assert (arr.behavior[0] == BEHAVIOR_TO_INT["Immobile"]).all()
    assert (arr.direction[0] == DIRECTION_TO_INT["Straight"]).all()
    expected = BEHAVIOR_TO_INT["Immobile"] * 3 + DIRECTION_TO_INT["Straight"]
    assert (arr.combined[0] == expected).all()


def test_subsample_1_passthrough(tmp_path):
    """If a future CSV is at H5 rate, subsample=1 should pass frames through."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Walk", 50, 100),
        _d("clip_0.mp4", "Straight", 50, 100),
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 0), n_frames=250,
                            csv_to_h5_subsample=1)
    assert (arr.behavior[0, 50:100] == BEHAVIOR_TO_INT["Walk"]).all()
    assert (arr.behavior[0, :50] == UNLABELED).all()
    assert (arr.behavior[0, 100:] == UNLABELED).all()


# ── Track parsing ────────────────────────────────────────────────────────────


def test_two_tracks_combine_into_combined_label(tmp_path):
    csv_path = _write_csv(tmp_path, [
        _b("clip_5.mp4", "Walk", 0, 200),
        _d("clip_5.mp4", "Left", 0, 100),
        _d("clip_5.mp4", "Right", 100, 200),
    ])
    arr = load_label_frames(csv_path, clip_range=(5, 5), n_frames=250)
    walk_left = COMBINED_TO_INT["Walk_Left"]
    walk_right = COMBINED_TO_INT["Walk_Right"]
    # CSV [0, 100) → H5 [0, 50); CSV [100, 200) → H5 [50, 100).
    assert (arr.combined[0, 0:50] == walk_left).all()
    assert (arr.combined[0, 50:100] == walk_right).all()
    assert (arr.combined[0, 100:] == UNLABELED).all()


def test_combined_unlabeled_when_only_one_track_present(tmp_path):
    """If a frame is behavior-labeled but direction-unlabeled (or vice
    versa), the combined view masks it as UNLABELED."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Immobile", 0, 500),
        # No direction row → direction-unlabeled everywhere.
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)
    assert (arr.behavior[0] == BEHAVIOR_TO_INT["Immobile"]).all()
    assert (arr.direction[0] == UNLABELED).all()
    assert (arr.combined[0] == UNLABELED).all()
    assert int(arr.n_labeled_per_clip[0]) == 0


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_interval_spanning_h5_boundary_is_clipped(tmp_path):
    """CSV [400, 600) at sub=2 → H5 [200, 300); clipped to [200, 250)."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_3.mp4", "Rear", 400, 600),
        _d("clip_3.mp4", "Straight", 400, 600),
    ])
    arr = load_label_frames(csv_path, clip_range=(3, 3), n_frames=250)
    expected_combined = COMBINED_TO_INT["Rear_Straight"]
    assert (arr.combined[0, 200:250] == expected_combined).all()
    assert (arr.combined[0, :200] == UNLABELED).all()


def test_interval_entirely_beyond_h5_is_dropped(tmp_path):
    """CSV [600, 700) at sub=2 → H5 [300, 350); all beyond n_frames=250."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Walk", 600, 700),
        _d("clip_0.mp4", "Right", 600, 700),
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)
    assert (arr.behavior == UNLABELED).all()
    assert (arr.combined == UNLABELED).all()


def test_empty_interval_is_dropped(tmp_path):
    """Start_Frame == End_Frame produces a no-op."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Walk", 50, 50),
        _d("clip_0.mp4", "Straight", 50, 50),
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)
    assert (arr.behavior == UNLABELED).all()
    assert (arr.direction == UNLABELED).all()


def test_inverted_interval_is_dropped(tmp_path):
    """End < Start produces a no-op (after CSV→H5 mapping)."""
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Walk", 100, 50),
        _d("clip_0.mp4", "Straight", 100, 50),
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)
    assert (arr.behavior == UNLABELED).all()


def test_out_of_range_clips_are_ignored(tmp_path):
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4",  "Walk",     0, 100),
        _b("clip_500.mp4", "Walk",    0, 100),
        _d("clip_0.mp4",  "Straight", 0, 100),
        _d("clip_500.mp4", "Straight", 0, 100),
    ])
    arr = load_label_frames(csv_path, clip_range=(0, 99), n_frames=250)
    assert arr.behavior.shape == (100, 250)
    assert int(arr.clip_indices[0]) == 0 and int(arr.clip_indices[-1]) == 99
    assert (arr.behavior[0, :50] == BEHAVIOR_TO_INT["Walk"]).all()


# ── Errors ───────────────────────────────────────────────────────────────────


def test_unknown_behavior_raises(tmp_path):
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Flapping", 0, 10),
    ])
    with pytest.raises(KeyError, match="Flapping"):
        load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)


def test_unknown_direction_raises(tmp_path):
    csv_path = _write_csv(tmp_path, [
        _d("clip_0.mp4", "Sideways", 0, 10),
    ])
    with pytest.raises(KeyError, match="Sideways"):
        load_label_frames(csv_path, clip_range=(0, 0), n_frames=250)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_label_frames(tmp_path / "nope.csv")


def test_missing_track_column_raises(tmp_path):
    """Old single-Behavior CSV (no Track column) must fail loudly with a
    migration hint."""
    p = tmp_path / "old.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["Video", "Behavior", "Start_Frame", "End_Frame"],
        )
        w.writeheader()
        w.writerow({
            "Video": "clip_642.mp4", "Behavior": "Walking",
            "Start_Frame": 0, "End_Frame": 50,
        })
    with pytest.raises(KeyError, match="Track"):
        load_label_frames(p)


def test_inverted_clip_range_raises(tmp_path):
    csv_path = _write_csv(tmp_path, [_b("clip_0.mp4", "Walk", 0, 10),
                                     _d("clip_0.mp4", "Straight", 0, 10)])
    with pytest.raises(ValueError, match="clip_range"):
        load_label_frames(csv_path, clip_range=(10, 5), n_frames=250)


def test_invalid_subsample_raises(tmp_path):
    csv_path = _write_csv(tmp_path, [_b("clip_0.mp4", "Walk", 0, 10),
                                     _d("clip_0.mp4", "Straight", 0, 10)])
    with pytest.raises(ValueError, match="subsample"):
        load_label_frames(csv_path, csv_to_h5_subsample=0)


# ── Backward-compat wrapper ──────────────────────────────────────────────────


def test_load_behavior_frames_returns_legacy_tuple(tmp_path):
    csv_path = _write_csv(tmp_path, [
        _b("clip_0.mp4", "Rear", 0, 100),
        _d("clip_0.mp4", "Straight", 0, 100),
    ])
    fl, idx, n_lab = load_behavior_frames(csv_path, clip_range=(0, 0), n_frames=250)
    assert fl.shape == (1, 250)
    assert idx.tolist() == [0]
    assert (fl[0, :50] == BEHAVIOR_TO_INT["Rear"]).all()
    assert (fl[0, 50:] == UNLABELED).all()
    assert n_lab.tolist() == [50]


# ── Real CSV smoke test ──────────────────────────────────────────────────────


def test_real_csv_structural_sanity():
    """Sanity-check against the real CSV in the repo (if it exists).

    This guards against a malformed file shape silently propagating.
    Exact counts may shift if labels are re-annotated; we test only
    that the loader runs end-to-end and returns plausible shapes.
    """
    repo_csv = Path(__file__).resolve().parents[2] / "data" / "rodent" / "rodent_labels.csv"
    if not repo_csv.exists():
        pytest.skip(f"Real labels CSV not present at {repo_csv}")
    arr = load_label_frames(repo_csv, clip_range=(0, 841), n_frames=250)
    assert arr.behavior.shape == (842, 250)
    assert arr.direction.shape == (842, 250)
    assert arr.combined.shape == (842, 250)
    # The current CSV labels every frame in both tracks → combined fully covered.
    assert int((arr.n_labeled_per_clip == 250).sum()) >= 800, (
        f"Only {(arr.n_labeled_per_clip == 250).sum()} clips fully labeled"
    )
    # Every of the 4 behaviors must appear somewhere.
    present_behaviors = set(int(v) for v in np.unique(arr.behavior) if v != UNLABELED)
    assert present_behaviors == set(range(len(BEHAVIORS))), (
        f"missing some behaviors: present={present_behaviors}"
    )
