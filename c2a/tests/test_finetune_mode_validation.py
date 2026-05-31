"""Tests for finetune-mode startup validation logic.

Covers:
  * the default checkpoint at ``model_checkpoints/260513_062522_599062/``
    exists and has the expected internal layout
  * ``_warn_if_ckpt_had_c2c_enabled`` correctly identifies whether a
    saved Hydra config has c2c on or off, and emits a warning only when
    c2c was on
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from c2a.paths import get_repo_root


DEFAULT_CKPT = Path(
    "/home/jovyan/vast/kaiwen/code2act/model_checkpoints/260513_062522_599062"
)


def _has_default_ckpt() -> bool:
    return DEFAULT_CKPT.exists() and any(
        DEFAULT_CKPT.glob("MoSeqPPONetwork_*")
    )


# -------------------- default checkpoint presence --------------------


@pytest.mark.skipif(
    not _has_default_ckpt(),
    reason=f"default finetune ckpt not present at {DEFAULT_CKPT}",
)
def test_default_checkpoint_exists():
    """The user-specified default ckpt exists and contains training-state subdirs."""
    assert DEFAULT_CKPT.exists()
    subdirs = list(DEFAULT_CKPT.glob("MoSeqPPONetwork_*"))
    assert len(subdirs) >= 1, (
        f"expected MoSeqPPONetwork_* subdirs under {DEFAULT_CKPT}; found 0"
    )
    # Sanity: the ckpt should have many saved steps (it's a long training run).
    assert len(subdirs) >= 50, (
        f"default ckpt should have ≥ 50 saved steps; found {len(subdirs)}"
    )


# -------------------- _warn_if_ckpt_had_c2c_enabled --------------------


def _make_synthetic_hydra_config(tmp_path: Path, run_id: str, *, c2c_mode: str | None) -> Path:
    """Build a fake outputs/2026-XX-XX/HH-MM-SS/.hydra/config.yaml that
    references the given run_id and has c2c.reward_mode set as specified."""
    out_dir = tmp_path / "outputs" / "2026-05-13" / "12-34-56" / ".hydra"
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = f"""# Synthetic test config
train_setup:
  run_name: synthetic_test_{run_id}
"""
    if c2c_mode is not None:
        yaml_text += f"""code2code:
  reward_mode: {c2c_mode}
"""
    config_yaml = out_dir / "config.yaml"
    config_yaml.write_text(yaml_text)

    # Also write a fake run_state-style marker so the run_id matches.
    # The helper searches by substring, so anything referencing run_id works.
    marker = config_yaml.parent.parent / "run.log"
    marker.write_text(f"Run ID: {run_id}\n")
    return out_dir.parent.parent  # the timestamp dir


def test_warn_when_loading_c2c_ckpt(monkeypatch, tmp_path, caplog):
    """A ckpt whose saved Hydra config has c2c on should trigger a WARNING."""
    from c2a import train_moseq_decoder as tmd

    run_id = "fake_ckpt_c2c_on"
    fake_ckpt = tmp_path / "model_checkpoints" / run_id
    fake_ckpt.mkdir(parents=True)
    (fake_ckpt / "MoSeqPPONetwork_0").mkdir()
    _make_synthetic_hydra_config(tmp_path, run_id, c2c_mode="replace")

    monkeypatch.setattr(tmd, "get_repo_root", lambda: tmp_path)

    with caplog.at_level(logging.WARNING):
        tmd._warn_if_ckpt_had_c2c_enabled(fake_ckpt)

    assert any(
        "was trained with code2code.reward_mode='replace'" in rec.message
        or "was trained with code2code.reward_mode=\"replace\"" in rec.message
        or "reward_mode='replace'" in rec.message
        for rec in caplog.records
    ), f"expected warning about c2c='replace'; got {[r.message for r in caplog.records]}"


def test_no_warn_for_pure_tracking_ckpt(monkeypatch, tmp_path, caplog):
    """A ckpt whose saved Hydra config has c2c off (or absent) should NOT warn."""
    from c2a import train_moseq_decoder as tmd

    run_id = "fake_ckpt_c2c_off"
    fake_ckpt = tmp_path / "model_checkpoints" / run_id
    fake_ckpt.mkdir(parents=True)
    (fake_ckpt / "MoSeqPPONetwork_0").mkdir()
    _make_synthetic_hydra_config(tmp_path, run_id, c2c_mode="off")

    monkeypatch.setattr(tmd, "get_repo_root", lambda: tmp_path)

    with caplog.at_level(logging.WARNING):
        tmd._warn_if_ckpt_had_c2c_enabled(fake_ckpt)

    # No WARNING-level records should mention c2c-on misconfiguration.
    warning_msgs = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    ]
    bad = [m for m in warning_msgs if "reward_mode=" in m and "off" not in m]
    assert bad == [], f"unexpected warnings: {bad}"


def test_no_warn_for_pre_c2c_ckpt_no_code2code_block(monkeypatch, tmp_path, caplog):
    """A ckpt whose saved config predates c2c entirely (no code2code block)
    is treated as pure-tracking — no warning."""
    from c2a import train_moseq_decoder as tmd

    run_id = "fake_ckpt_pre_c2c"
    fake_ckpt = tmp_path / "model_checkpoints" / run_id
    fake_ckpt.mkdir(parents=True)
    (fake_ckpt / "MoSeqPPONetwork_0").mkdir()
    _make_synthetic_hydra_config(tmp_path, run_id, c2c_mode=None)

    monkeypatch.setattr(tmd, "get_repo_root", lambda: tmp_path)

    with caplog.at_level(logging.WARNING):
        tmd._warn_if_ckpt_had_c2c_enabled(fake_ckpt)

    warning_msgs = [
        r.message for r in caplog.records if r.levelno >= logging.WARNING
    ]
    bad = [m for m in warning_msgs if "reward_mode=" in m]
    assert bad == [], f"unexpected warnings for pre-c2c ckpt: {bad}"


def test_missing_saved_config_warns_but_doesnt_raise(monkeypatch, tmp_path, caplog):
    """If no saved Hydra config is findable, we warn but don't crash."""
    from c2a import train_moseq_decoder as tmd

    run_id = "fake_ckpt_no_config"
    fake_ckpt = tmp_path / "model_checkpoints" / run_id
    fake_ckpt.mkdir(parents=True)
    (fake_ckpt / "MoSeqPPONetwork_0").mkdir()
    # No outputs/ tree at all — search should come up empty.

    monkeypatch.setattr(tmd, "get_repo_root", lambda: tmp_path)

    with caplog.at_level(logging.WARNING):
        tmd._warn_if_ckpt_had_c2c_enabled(fake_ckpt)

    # Should not raise.  Should emit a "could not locate" warning.
    assert any(
        "could not locate" in rec.message.lower()
        or "outputs/ dir not found" in rec.message
        for rec in caplog.records
    )
