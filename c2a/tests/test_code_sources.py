"""Tests for ``c2a.codegen.code_sources.load_codes``.

The loader is the single source-of-truth for behavioral codes feeding
the MoSeq decoder. These tests exercise the contract every downstream
consumer depends on:

  * shape / dtype / value-range invariants
  * source dispatch
  * provenance enforcement for ``kpms_merged``
  * train/test index alignment for ``kpms_raw`` (caller's order must
    match the npz's stored order — silent misalignment would feed
    misaligned codes to the policy)
  * UNLABELED handling for ``gt_directional_turn`` (raises, no silent
    sentinel substitution)
  * the 5-class mapping itself
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from c2a.codegen.code_sources import (
    GT_DIRECTIONAL_TURN_NUM_CODES,
    SOURCE_GT_DIRECTIONAL_TURN,
    SOURCE_KPMS_MERGED,
    SOURCE_KPMS_RAW,
    SOURCE_RANDOM,
    SOURCE_SHIMOSEQ,
    load_codes,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_cfg(**kpms_overrides) -> OmegaConf:
    """Minimal cfg with the keys load_codes reads."""
    cfg = OmegaConf.create(
        {
            "kpms_config": {
                "code_source": SOURCE_KPMS_RAW,
                "codes_path": None,
                "labels_csv_path": None,
                "csv_to_h5_subsample": 2,
                **kpms_overrides,
            },
            "network_config": {"num_codes": 7},
            "env_config": {"clip_length": 10},
        }
    )
    return cfg


def _save_kpms_npz(
    path: Path,
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    is_merged: bool = False,
    num_states: int | None = None,
    extra: dict | None = None,
) -> Path:
    payload = {
        "train_codes": train_codes.astype(np.int32),
        "test_codes": test_codes.astype(np.int32),
        "train_indices": train_indices.astype(np.int64),
        "test_indices": test_indices.astype(np.int64),
        "kappa": np.float64(123.0),
        "model_type": np.array("arhmm"),
        "mean_duration": np.float64(6.0),
    }
    if num_states is not None:
        payload["num_states"] = np.int32(num_states)
    if is_merged:
        payload["is_merged"] = np.bool_(True)
        payload["original_num_states"] = np.int32(50)
        payload["new_num_states"] = np.int32(num_states or 7)
    if extra:
        payload.update(extra)
    np.savez(path, **payload)
    return path


# ── kpms_raw / kpms_merged (the .npz path) ───────────────────────────────────


def test_kpms_raw_loads_and_validates_shape(tmp_path):
    n_train, n_test, T = 3, 2, 10
    train_codes = np.random.RandomState(0).randint(0, 5, (n_train, T))
    test_codes = np.random.RandomState(1).randint(0, 5, (n_test, T))
    train_idx = np.array([0, 5, 9])
    test_idx = np.array([3, 8])
    p = _save_kpms_npz(
        tmp_path / "codes.npz", train_codes, test_codes, train_idx, test_idx,
        num_states=5,
    )

    cfg = _make_cfg(code_source=SOURCE_KPMS_RAW, codes_path=str(p))
    payload = load_codes(cfg, train_idx.tolist(), test_idx.tolist(), n_frames=T)
    assert payload.code_source == SOURCE_KPMS_RAW
    assert payload.num_codes == 5
    assert payload.train_codes.shape == (n_train, T)
    assert payload.test_codes.shape == (n_test, T)
    assert payload.train_codes.dtype == np.int32
    assert payload.test_codes.dtype == np.int32
    assert int(payload.train_codes.min()) >= 0
    assert int(payload.train_codes.max()) < payload.num_codes
    np.testing.assert_array_equal(payload.train_codes, train_codes)
    np.testing.assert_array_equal(payload.test_codes, test_codes)


def test_kpms_raw_warns_on_unexpected_merged_flag(tmp_path, caplog):
    n_train, n_test, T = 2, 1, 4
    train_codes = np.zeros((n_train, T), np.int32)
    test_codes = np.zeros((n_test, T), np.int32)
    p = _save_kpms_npz(
        tmp_path / "codes.npz", train_codes, test_codes,
        np.array([0, 1]), np.array([2]),
        is_merged=True, num_states=1,
    )
    cfg = _make_cfg(code_source=SOURCE_KPMS_RAW, codes_path=str(p))
    with caplog.at_level("WARNING"):
        payload = load_codes(cfg, [0, 1], [2], n_frames=T)
    assert payload.num_codes == 1
    # Loader respects the user's requested source; the fact that the
    # underlying file is merged surfaces only via provenance, not via
    # code_source — silently re-categorizing the run would break WandB
    # filters.
    assert payload.code_source == SOURCE_KPMS_RAW
    assert payload.provenance["is_merged"] is True
    assert any("is_merged=True" in m for m in caplog.messages)


def test_kpms_merged_requires_provenance_flag(tmp_path):
    """A raw .npz must NOT pass when code_source=kpms_merged."""
    n_train, n_test, T = 2, 1, 4
    train_codes = np.zeros((n_train, T), np.int32)
    test_codes = np.zeros((n_test, T), np.int32)
    p = _save_kpms_npz(
        tmp_path / "raw.npz", train_codes, test_codes,
        np.array([0, 1]), np.array([2]),
        is_merged=False, num_states=1,
    )
    cfg = _make_cfg(code_source=SOURCE_KPMS_MERGED, codes_path=str(p))
    with pytest.raises(RuntimeError, match="is_merged=True"):
        load_codes(cfg, [0, 1], [2], n_frames=T)


def test_kpms_merged_accepts_with_flag(tmp_path):
    n_train, n_test, T = 2, 1, 4
    train_codes = np.zeros((n_train, T), np.int32)
    test_codes = np.zeros((n_test, T), np.int32)
    p = _save_kpms_npz(
        tmp_path / "merged.npz", train_codes, test_codes,
        np.array([0, 1]), np.array([2]),
        is_merged=True, num_states=1,
    )
    cfg = _make_cfg(code_source=SOURCE_KPMS_MERGED, codes_path=str(p))
    payload = load_codes(cfg, [0, 1], [2], n_frames=T)
    assert payload.code_source == SOURCE_KPMS_MERGED
    assert payload.provenance["is_merged"] is True


def test_kpms_indices_misalignment_raises(tmp_path):
    """If the npz was generated against a different split, raise."""
    n_train, n_test, T = 2, 1, 4
    train_codes = np.zeros((n_train, T), np.int32)
    test_codes = np.zeros((n_test, T), np.int32)
    p = _save_kpms_npz(
        tmp_path / "codes.npz", train_codes, test_codes,
        train_indices=np.array([100, 200]),  # stored
        test_indices=np.array([300]),
        num_states=1,
    )
    cfg = _make_cfg(code_source=SOURCE_KPMS_RAW, codes_path=str(p))
    with pytest.raises(RuntimeError, match="train_indices"):
        # caller passes different indices -> raise
        load_codes(cfg, [0, 1], [2], n_frames=T)


def test_kpms_data_max_exceeds_declared_K_raises(tmp_path):
    n_train, n_test, T = 2, 1, 4
    # codes contain value 4 but num_states=2 — must raise
    train_codes = np.array([[4, 0, 0, 0], [0, 0, 0, 0]])
    test_codes = np.zeros((n_test, T), np.int32)
    p = _save_kpms_npz(
        tmp_path / "bad.npz", train_codes, test_codes,
        np.array([0, 1]), np.array([2]),
        num_states=2,
    )
    cfg = _make_cfg(code_source=SOURCE_KPMS_RAW, codes_path=str(p))
    with pytest.raises(RuntimeError, match="declares num_states"):
        load_codes(cfg, [0, 1], [2], n_frames=T)


# ── random fallback ──────────────────────────────────────────────────────────


def test_random_when_codes_path_null():
    cfg = _make_cfg(code_source=SOURCE_KPMS_RAW, codes_path=None)
    cfg.network_config.num_codes = 7
    payload = load_codes(cfg, [0, 1, 2], [3, 4], n_frames=8)
    assert payload.code_source == SOURCE_RANDOM
    assert payload.num_codes == 7
    assert payload.train_codes.shape == (3, 8)
    assert payload.test_codes.shape == (2, 8)
    assert int(payload.train_codes.min()) >= 0
    assert int(payload.train_codes.max()) < 7


# ── shimoseq (same npz schema; differentiated only by requested_source) ─────


def test_shimoseq_loads_via_kpms_npz_schema(tmp_path):
    """shimoseq codes share the npz schema with kpms; load_codes routes
    by code_source but uses the same reader. payload.code_source must
    echo the user's request, not the file's model_type."""
    n_train, n_test, T = 3, 2, 10
    train_codes = np.random.RandomState(0).randint(0, 5, (n_train, T))
    test_codes = np.random.RandomState(1).randint(0, 5, (n_test, T))
    train_idx = np.array([0, 5, 9])
    test_idx = np.array([3, 8])
    p = _save_kpms_npz(
        tmp_path / "shimoseq_codes.npz",
        train_codes, test_codes, train_idx, test_idx,
        num_states=5,
    )

    cfg = _make_cfg(code_source=SOURCE_SHIMOSEQ, codes_path=str(p))
    payload = load_codes(cfg, train_idx.tolist(), test_idx.tolist(), n_frames=T)
    assert payload.code_source == SOURCE_SHIMOSEQ
    assert payload.num_codes == 5
    assert payload.train_codes.shape == (n_train, T)
    np.testing.assert_array_equal(payload.train_codes, train_codes)


def test_shimoseq_null_path_falls_back_to_random():
    """``code_source=shimoseq`` with ``codes_path=null`` should follow the
    same smoke-test fallback as kpms_raw."""
    cfg = _make_cfg(code_source=SOURCE_SHIMOSEQ, codes_path=None)
    cfg.network_config.num_codes = 7
    payload = load_codes(cfg, [0, 1, 2], [3, 4], n_frames=8)
    # Random fallback overrides the requested source — that's the existing
    # contract (SOURCE_RANDOM is the truth, kpms_config.code_source is
    # the "requested" intent).
    assert payload.code_source == SOURCE_RANDOM
    assert payload.num_codes == 7


# ── kinematics_vae family (npz schema shared with kpms/shimoseq) ────────────


def test_kinematics_vae_loads_via_npz_schema(tmp_path):
    """kinematics_vae / vqvae / ps_vqvae all share the codes-npz schema
    and load through the same _load_kpms_npz path; only the WandB-
    provenance ``code_source`` differs."""
    from c2a.codegen.code_sources import SOURCE_KINEMATICS_VAE, SOURCE_VQVAE, SOURCE_PS_VQVAE
    n_train, n_test, T = 3, 2, 10
    train_codes = np.random.RandomState(0).randint(0, 8, (n_train, T))
    test_codes = np.random.RandomState(1).randint(0, 8, (n_test, T))
    train_idx = np.array([0, 5, 9])
    test_idx = np.array([3, 8])

    for src in (SOURCE_KINEMATICS_VAE, SOURCE_VQVAE, SOURCE_PS_VQVAE):
        p = _save_kpms_npz(
            tmp_path / f"codes_{src}.npz",
            train_codes, test_codes, train_idx, test_idx,
            num_states=8,
        )
        cfg = _make_cfg(code_source=src, codes_path=str(p))
        payload = load_codes(cfg, train_idx.tolist(), test_idx.tolist(), n_frames=T)
        assert payload.code_source == src
        assert payload.num_codes == 8


def test_kinematics_vae_null_path_raises_not_random():
    """For kinematics_vae family sources, codes_path=null must FAIL —
    random fallback would feed decoder codes unrelated to any real
    tokenizer's codebook size, silently corrupting training. The
    pre-PR2 fallback for kpms_raw / kpms_merged / shimoseq is unchanged."""
    from c2a.codegen.code_sources import SOURCE_KINEMATICS_VAE, SOURCE_VQVAE, SOURCE_PS_VQVAE
    for src in (SOURCE_KINEMATICS_VAE, SOURCE_VQVAE, SOURCE_PS_VQVAE):
        cfg = _make_cfg(code_source=src, codes_path=None)
        with pytest.raises(ValueError, match="codes_path to be set"):
            load_codes(cfg, [0, 1], [2], n_frames=8)


# ── gt_directional_turn ──────────────────────────────────────────────────────


_FIELDS = [
    "Video", "Track", "Label", "Start_Time", "End_Time",
    "Start_Frame", "End_Frame",
]


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def _row(clip_id: int, track: str, label: str, s_csv: int, e_csv: int) -> dict:
    return {
        "Video": f"clip_{clip_id}.mp4",
        "Track": track,
        "Label": label,
        "Start_Time": str(s_csv / 30.0),
        "End_Time": str(e_csv / 30.0),
        "Start_Frame": str(s_csv),
        "End_Frame": str(e_csv),
    }


def test_gt_directional_turn_basic_mapping(tmp_path):
    """Cover the 5-class mapping with a hand-crafted CSV.

    csv_to_h5_subsample=2; n_frames=4; so a CSV interval [0,8) maps to
    H5 frames [0,4).
    """
    rows = []
    # clip 0: Immobile/Straight for [0,4) Turn/Left for [4,6) Walk/Straight [6,8)
    rows += [
        _row(0, "behavior", "Immobile", 0, 4),
        _row(0, "behavior", "Turn", 4, 6),
        _row(0, "behavior", "Walk", 6, 8),
        _row(0, "direction", "Straight", 0, 4),
        _row(0, "direction", "Left", 4, 6),
        _row(0, "direction", "Straight", 6, 8),
    ]
    # clip 1: Rear/Right for [0,4) Turn/Right [4,8)
    rows += [
        _row(1, "behavior", "Rear", 0, 4),
        _row(1, "behavior", "Turn", 4, 8),
        _row(1, "direction", "Right", 0, 4),
        _row(1, "direction", "Right", 4, 8),
    ]
    csv_path = _write_csv(tmp_path / "labels.csv", rows)

    cfg = _make_cfg(
        code_source=SOURCE_GT_DIRECTIONAL_TURN,
        labels_csv_path=str(csv_path),
        csv_to_h5_subsample=2,
    )
    cfg.env_config.clip_length = 4
    payload = load_codes(cfg, [0], [1], n_frames=4)
    assert payload.code_source == SOURCE_GT_DIRECTIONAL_TURN
    assert payload.num_codes == GT_DIRECTIONAL_TURN_NUM_CODES == 5
    # train clip 0: H5 frames 0,1 = Immobile->0; frame 2 = Turn+Left -> 3; frame 3 = Walk -> 2
    np.testing.assert_array_equal(payload.train_codes[0], [0, 0, 3, 2])
    # test clip 1: H5 frames 0,1 = Rear -> 1; frames 2,3 = Turn+Right -> 4
    np.testing.assert_array_equal(payload.test_codes[0], [1, 1, 4, 4])
    assert payload.train_codes.dtype == np.int32


def test_gt_directional_turn_raises_on_unlabeled(tmp_path):
    """A clip with a gap in either track must trigger a hard error."""
    rows = [
        # clip 0: behavior covers [0,4) only (CSV-frames [0,8))  — H5 frames 4..9 (in 10) unlabeled
        _row(0, "behavior", "Immobile", 0, 8),
        _row(0, "direction", "Straight", 0, 8),
        # but n_frames=10 in env, so frames 4..9 of H5 are uncovered
    ]
    csv_path = _write_csv(tmp_path / "labels.csv", rows)
    cfg = _make_cfg(
        code_source=SOURCE_GT_DIRECTIONAL_TURN,
        labels_csv_path=str(csv_path),
        csv_to_h5_subsample=2,
    )
    cfg.env_config.clip_length = 10
    with pytest.raises(RuntimeError, match="UNLABELED"):
        load_codes(cfg, [0], [], n_frames=10)


def test_gt_directional_turn_raises_on_turn_straight(tmp_path):
    """Turn+Straight has no class in the 5-class mapping; must raise."""
    rows = [
        _row(0, "behavior", "Turn", 0, 8),
        _row(0, "direction", "Straight", 0, 8),
    ]
    csv_path = _write_csv(tmp_path / "labels.csv", rows)
    cfg = _make_cfg(
        code_source=SOURCE_GT_DIRECTIONAL_TURN,
        labels_csv_path=str(csv_path),
        csv_to_h5_subsample=2,
    )
    cfg.env_config.clip_length = 4
    with pytest.raises(RuntimeError, match="Turn\\+Straight"):
        load_codes(cfg, [0], [], n_frames=4)


def test_gt_directional_turn_missing_csv_raises():
    cfg = _make_cfg(
        code_source=SOURCE_GT_DIRECTIONAL_TURN,
        labels_csv_path="/nonexistent/path/labels.csv",
    )
    cfg.env_config.clip_length = 4
    with pytest.raises(FileNotFoundError):
        load_codes(cfg, [0], [], n_frames=4)


# ── source-name dispatch / unknown source ───────────────────────────────────


def test_unknown_source_raises():
    cfg = _make_cfg(code_source="not_a_source", codes_path="/dev/null")
    with pytest.raises(ValueError, match="Unknown.*code_source"):
        load_codes(cfg, [0], [1], n_frames=4)
