"""Tests for c2a.kpms.label_alignment."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from c2a.kpms.label_alignment import (
    BEHAVIOR_TO_INT,
    DROPPED_BEHAVIORS,
    UNLABELED,
    load_behavior_frames,
)


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "labels.csv"
    fieldnames = ["Video", "Behavior", "Start_Time", "End_Time", "Start_Frame", "End_Frame"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            filled = {k: row.get(k, "") for k in fieldnames}
            w.writerow(filled)
    return p


def test_basic_interval_maps_to_frame_slice(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_642.mp4", "Behavior": "Grooming", "Start_Frame": 0,  "End_Frame": 50},
        {"Video": "clip_642.mp4", "Behavior": "Walking",  "Start_Frame": 50, "End_Frame": 100},
    ])

    fl, idx, n_lab = load_behavior_frames(csv_path, clip_range=(642, 642), n_frames=250)

    assert fl.shape == (1, 250)
    assert idx.tolist() == [642]
    assert (fl[0, 0:50] == BEHAVIOR_TO_INT["Grooming"]).all()
    assert (fl[0, 50:100] == BEHAVIOR_TO_INT["Walking"]).all()
    assert (fl[0, 100:] == UNLABELED).all()
    assert n_lab.tolist() == [100]


def test_interval_spanning_nframes_boundary_is_clipped(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_700.mp4", "Behavior": "Immobility", "Start_Frame": 240, "End_Frame": 260},
    ])
    fl, _, n_lab = load_behavior_frames(csv_path, clip_range=(700, 700), n_frames=250)
    assert (fl[0, 240:250] == BEHAVIOR_TO_INT["Immobility"]).all()
    assert (fl[0, :240] == UNLABELED).all()
    assert n_lab.tolist() == [10]  # 250 - 240


def test_interval_entirely_beyond_nframes_is_dropped(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_700.mp4", "Behavior": "Immobility", "Start_Frame": 300, "End_Frame": 400},
    ])
    fl, _, n_lab = load_behavior_frames(csv_path, clip_range=(700, 700), n_frames=250)
    assert (fl == UNLABELED).all()
    assert n_lab.tolist() == [0]


def test_turns_are_dropped(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_642.mp4", "Behavior": "Left Turn",  "Start_Frame": 10, "End_Frame": 30},
        {"Video": "clip_642.mp4", "Behavior": "Right Turn", "Start_Frame": 30, "End_Frame": 50},
        {"Video": "clip_642.mp4", "Behavior": "Walking",    "Start_Frame": 50, "End_Frame": 80},
    ])
    fl, _, n_lab = load_behavior_frames(csv_path, clip_range=(642, 642), n_frames=250)
    # Turn frames should remain UNLABELED.
    assert (fl[0, 10:50] == UNLABELED).all()
    assert (fl[0, 50:80] == BEHAVIOR_TO_INT["Walking"]).all()
    # Turn behaviors must not have been smuggled into BEHAVIOR_TO_INT.
    for t in DROPPED_BEHAVIORS:
        assert t not in BEHAVIOR_TO_INT
    assert n_lab.tolist() == [30]


def test_out_of_range_clips_are_ignored(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_500.mp4", "Behavior": "Walking", "Start_Frame": 0, "End_Frame": 100},
        {"Video": "clip_900.mp4", "Behavior": "Walking", "Start_Frame": 0, "End_Frame": 100},
        {"Video": "clip_700.mp4", "Behavior": "Walking", "Start_Frame": 0, "End_Frame": 100},
    ])
    fl, idx, _ = load_behavior_frames(csv_path, clip_range=(642, 741), n_frames=250)
    assert idx.tolist() == list(range(642, 742))
    pos_700 = np.where(idx == 700)[0][0]
    assert (fl[pos_700, :100] == BEHAVIOR_TO_INT["Walking"]).all()
    # 500 and 900 are out of range — those clips are not in idx.


def test_unknown_behavior_raises(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_642.mp4", "Behavior": "Flapping", "Start_Frame": 0, "End_Frame": 10},
    ])
    with pytest.raises(KeyError, match="Flapping"):
        load_behavior_frames(csv_path, clip_range=(642, 642), n_frames=250)


def test_empty_interval_is_dropped_and_accounted(tmp_path):
    """Rows with Start_Frame == End_Frame (real CSV has these) must not write
    anything, and row-accounting must still balance (no silent drops)."""
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_700.mp4", "Behavior": "Immobility", "Start_Frame": 0, "End_Frame": 0},
        {"Video": "clip_700.mp4", "Behavior": "Walking", "Start_Frame": 0, "End_Frame": 50},
    ])
    fl, _, n_lab = load_behavior_frames(csv_path, clip_range=(700, 700), n_frames=250)
    assert (fl[0, :50] == BEHAVIOR_TO_INT["Walking"]).all()
    assert (fl[0, 50:] == UNLABELED).all()
    assert n_lab.tolist() == [50]


def test_inverted_interval_is_dropped(tmp_path):
    csv_path = _write_csv(tmp_path, [
        {"Video": "clip_700.mp4", "Behavior": "Walking", "Start_Frame": 100, "End_Frame": 50},
    ])
    fl, _, _ = load_behavior_frames(csv_path, clip_range=(700, 700), n_frames=250)
    assert (fl == UNLABELED).all()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_behavior_frames(tmp_path / "does_not_exist.csv")


def test_missing_column_raises(tmp_path):
    # Write a CSV missing the Start_Frame column.
    p = tmp_path / "bad.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Video", "Behavior", "End_Frame"])
        w.writeheader()
        w.writerow({"Video": "clip_642.mp4", "Behavior": "Walking", "End_Frame": 50})
    with pytest.raises(KeyError, match="Start_Frame"):
        load_behavior_frames(p, clip_range=(642, 642), n_frames=250)


def test_real_csv_structural_sanity():
    """Sanity-check against the real CSV in the repo (if it exists).

    This is not a unit test of exact content (the CSV may change), but it
    guards against a malformed file shape silently propagating.
    """
    repo_csv = Path(__file__).resolve().parents[2] / "data" / "rodent" / "rodent_labels.csv"
    if not repo_csv.exists():
        pytest.skip(f"Real labels CSV not present at {repo_csv}")
    fl, idx, n_lab = load_behavior_frames(repo_csv, clip_range=(642, 741), n_frames=250)
    assert fl.shape == (100, 250)
    assert idx[0] == 642 and idx[-1] == 741
    # Real data has labels for nearly every clip in range; most clips are near-fully labeled.
    assert int((n_lab > 0).sum()) >= 95, f"Only {(n_lab > 0).sum()} clips labeled"
    assert float(n_lab.mean()) > 200, f"Mean labeled frames too low: {n_lab.mean()}"
