"""Tests for ``c2a.codegen.shared.npz_schema.write_codes_npz``.

This is the single-writer chokepoint every tokenizer's codegen routes
through. Any silent regression here would break downstream loading via
``c2a.codegen.code_sources.load_codes``. Tests verify:

  * shape / dtype / range validation BEFORE write (no corrupt files left
    on disk)
  * reserved-field collision detection
  * round-trip through the reader (``_load_kpms_npz``) — the contract
    that downstream depends on
  * is_merged provenance pathway
"""

from __future__ import annotations

import numpy as np
import pytest

from c2a.codegen.code_sources import _load_kpms_npz
from c2a.codegen.shared.npz_schema import RESERVED, write_codes_npz


def _basic_inputs(n_train=3, n_test=2, T=10, K=5):
    rng = np.random.RandomState(0)
    return {
        "train_codes": rng.randint(0, K, (n_train, T)).astype(np.int32),
        "test_codes": rng.randint(0, K, (n_test, T)).astype(np.int32),
        "train_indices": np.arange(n_train, dtype=np.int64),
        "test_indices": np.arange(n_train, n_train + n_test, dtype=np.int64),
        "num_states": K,
        "model_type": "tokenizer_x",
    }


def test_writer_round_trips_through_loader(tmp_path):
    """Written file must pass the loader's validation chokepoint."""
    inputs = _basic_inputs()
    out = tmp_path / "x.npz"
    written = write_codes_npz(out, **inputs, kappa=0.5, mean_duration=3.0)
    assert written.exists()

    payload = _load_kpms_npz(
        str(written),
        inputs["train_indices"], inputs["test_indices"],
        n_frames=inputs["train_codes"].shape[1],
        require_merged=False,
        requested_source="tokenizer_x",
    )
    np.testing.assert_array_equal(payload.train_codes, inputs["train_codes"])
    np.testing.assert_array_equal(payload.test_codes, inputs["test_codes"])
    assert payload.num_codes == inputs["num_states"]
    assert payload.provenance["kpms_kappa"] == 0.5
    assert payload.provenance["kpms_mean_duration"] == 3.0
    assert payload.provenance["kpms_model_type"] == "tokenizer_x"


def test_writer_rejects_out_of_range_codes(tmp_path):
    """Writing OOB codes must raise BEFORE the file lands on disk."""
    inputs = _basic_inputs()
    inputs["train_codes"][0, 0] = inputs["num_states"] + 1
    out = tmp_path / "oob.npz"
    with pytest.raises(ValueError, match="train_codes range"):
        write_codes_npz(out, **inputs)
    assert not out.exists(), "writer left a corrupt file on disk on validation failure"


def test_writer_rejects_negative_codes(tmp_path):
    inputs = _basic_inputs()
    inputs["test_codes"][0, 0] = -1
    out = tmp_path / "neg.npz"
    with pytest.raises(ValueError, match="test_codes range"):
        write_codes_npz(out, **inputs)
    assert not out.exists()


def test_writer_rejects_index_mismatch(tmp_path):
    inputs = _basic_inputs()
    inputs["train_indices"] = np.arange(inputs["train_codes"].shape[0] + 1)
    with pytest.raises(ValueError, match="len\\(train_indices\\)"):
        write_codes_npz(tmp_path / "mismatch.npz", **inputs)


def test_writer_rejects_reserved_collision(tmp_path):
    inputs = _basic_inputs()
    with pytest.raises(KeyError, match="reserved"):
        write_codes_npz(tmp_path / "collide.npz", **inputs, extra={"num_states": 99})


def test_writer_rejects_extra_at_each_reserved_name(tmp_path):
    """All reserved names must be guarded — not just the well-known ones."""
    for name in RESERVED:
        inputs = _basic_inputs()
        with pytest.raises(KeyError):
            write_codes_npz(tmp_path / f"r_{name}.npz", **inputs, extra={name: 1})


def test_writer_writes_is_merged_provenance(tmp_path):
    inputs = _basic_inputs()
    out = tmp_path / "merged.npz"
    written = write_codes_npz(
        out, **inputs,
        is_merged=True, original_num_states=50, new_num_states=inputs["num_states"],
    )
    # Reader must see is_merged=True and accept require_merged=True
    payload = _load_kpms_npz(
        str(written),
        inputs["train_indices"], inputs["test_indices"],
        n_frames=inputs["train_codes"].shape[1],
        require_merged=True,
        requested_source="kpms_merged",
    )
    assert payload.provenance["is_merged"] is True
    assert payload.provenance["original_num_states"] == 50


def test_writer_is_merged_requires_bookkeeping_fields(tmp_path):
    inputs = _basic_inputs()
    with pytest.raises(ValueError, match="original_num_states"):
        write_codes_npz(tmp_path / "x.npz", **inputs, is_merged=True)


def test_writer_all_codes_shape_validated(tmp_path):
    inputs = _basic_inputs()
    bad_all = np.zeros(
        (inputs["train_codes"].shape[0] + 1,
         inputs["train_codes"].shape[1]),
        dtype=np.int32,
    )
    with pytest.raises(ValueError, match="all_codes"):
        write_codes_npz(tmp_path / "x.npz", **inputs, all_codes=bad_all)


def test_writer_extra_fields_persist(tmp_path):
    """Tokenizer-specific provenance keys must survive the round-trip."""
    inputs = _basic_inputs()
    out = tmp_path / "extras.npz"
    write_codes_npz(out, **inputs, extra={"commit_beta": 0.25, "seed": 42})

    # Read raw to confirm the extras landed
    raw = dict(np.load(str(out), allow_pickle=False))
    assert float(raw["commit_beta"]) == 0.25
    assert int(raw["seed"]) == 42
