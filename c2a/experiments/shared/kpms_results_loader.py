"""Read per-frame syllable assignments from a KPMS ``results.h5``.

Each KPMS sweep config produces ``<project_dir>/<model_name>/results.h5``
containing one top-level group per recording (named ``recording_NNNN``)
with a ``syllable`` dataset of shape ``(n_frames,)`` int64.

For diagnostic metrics (frame_frequency, bout instances, durations),
the order of recordings does not matter — we just stack them into a
single ``(n_clips, n_frames)`` array. Alignment to a specific balanced
split is NOT done here (use ``c2a.codegen.code_sources.load_codes`` for
that).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger(__name__)


def load_kpms_syllables(
    results_h5_path: str | Path,
) -> np.ndarray:
    """Load all recordings as ``(n_clips, n_frames)`` int32.

    Args:
        results_h5_path: Path to KPMS ``results.h5``.

    Returns:
        ``(n_clips, n_frames)`` int32 syllable assignments. Recordings
        are sorted by group-name lexicographic order (matches numeric
        order for the ``recording_NNNN`` pattern KPMS uses).

    Raises:
        FileNotFoundError: results.h5 missing.
        RuntimeError: clips have inconsistent ``n_frames``.
    """
    p = Path(results_h5_path)
    if not p.exists():
        raise FileNotFoundError(f"KPMS results.h5 not found: {p}")

    with h5py.File(p, "r") as f:
        names = sorted(f.keys())
        if not names:
            raise RuntimeError(f"KPMS results.h5 at {p} has no recordings.")
        syllables = []
        n_frames = None
        for name in names:
            grp = f[name]
            if "syllable" not in grp:
                raise RuntimeError(
                    f"{p}::{name} missing 'syllable' dataset."
                )
            s = np.asarray(grp["syllable"][:], dtype=np.int32)
            if s.ndim != 1:
                raise RuntimeError(
                    f"{p}::{name}/syllable expected 1-D, got shape {s.shape}"
                )
            if n_frames is None:
                n_frames = s.shape[0]
            elif s.shape[0] != n_frames:
                raise RuntimeError(
                    f"{p}::{name}/syllable has n_frames={s.shape[0]} "
                    f"but earlier clips had n_frames={n_frames}. "
                    f"All clips must have matching length."
                )
            syllables.append(s)

    arr = np.stack(syllables, axis=0)
    log.info(
        "Loaded KPMS syllables from %s: %d recordings × %d frames, "
        "max code = %d",
        p, arr.shape[0], arr.shape[1], int(arr.max()),
    )
    return arr


def kpms_project_dir_for_setting(
    sweep_root: str | Path,
    setting: str,
    seed: int,
) -> Path:
    """Resolve the per-(setting, seed) KPMS project dir.

    KPMS sweep stores configs at::

        <sweep_root>/<setting>/seed<N>/keypoint_arhmm_states<S>_seed<N>/

    where ``setting`` is e.g. ``"s30_k8e+01_l10_arhmm"`` and ``S`` is the
    num_states. The model name embeds both.
    """
    sweep_root = Path(sweep_root)
    setting_dir = sweep_root / setting / f"seed{int(seed)}"
    if not setting_dir.exists():
        raise FileNotFoundError(
            f"KPMS seed dir not found: {setting_dir}"
        )
    # Find the inner model dir: keypoint_*_states<S>_seed<seed>
    model_dirs = sorted(
        d for d in setting_dir.iterdir()
        if d.is_dir() and d.name.startswith("keypoint_")
    )
    if not model_dirs:
        raise FileNotFoundError(
            f"No keypoint_* model dir under {setting_dir}"
        )
    if len(model_dirs) > 1:
        log.warning(
            "Multiple keypoint_* model dirs under %s; picking %s",
            setting_dir, model_dirs[0].name,
        )
    return model_dirs[0]
