"""Load KPMS code sequences and batch them for shMoSeq.

shMoSeq operates on per-frame syllable IDs as input.  In the c2a
pipeline these come from :func:`c2a.codegen.kpms.sweep_impl.run_sweep` →
:mod:`c2a.codegen.kpms.generate_codes`, which writes ``best_codes.npz``
(typically at ``c2a/outputs/kpms_sweep/best_codes.npz``).

The npz schema (set by :mod:`c2a.codegen.kpms.generate_codes`) is::

    all_codes: int (N_clips, T) — KPMS syllable per mocap frame
    train_codes / test_codes: convenience views
    train_indices / test_indices: balanced-split clip ids
    kappa, num_states, model_type, mean_duration: provenance metadata

We accept any of those formats; the only required key is ``all_codes``.

The ``state_moseq.batch`` API expects a Python dict mapping clip-id
strings to 1D integer arrays — which is what this module produces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def load_kpms_codes(
    codes_path: str | Path,
    clip_subset: list[int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load KPMS codes from a ``best_codes.npz`` and return a
    state_moseq-ready dict.

    Args:
        codes_path: path to ``best_codes.npz``.
        clip_subset: optional list of clip indices to keep.  Default:
            keep all clips present in ``all_codes``.

    Returns:
        ``(syllables_dict, provenance)`` where:

        * ``syllables_dict`` maps ``f"clip_{i}"`` (string id; what
          ``state_moseq.batch`` wants) to a 1D ``int32`` array of length
          ``T`` containing the per-frame KPMS syllable ids.
        * ``provenance`` contains the npz metadata (kappa, num_states,
          mean_duration, …) so the sweep can log what it ingested.
    """
    p = Path(codes_path)
    if not p.exists():
        raise FileNotFoundError(f"KPMS codes not found at {p}")

    with np.load(p, allow_pickle=False) as data:
        if "all_codes" not in data.files:
            raise KeyError(
                f"npz at {p} lacks 'all_codes'; got keys {sorted(data.files)}. "
                f"This loader expects the schema written by "
                f"c2a.codegen.kpms.generate_codes."
            )
        all_codes = np.asarray(data["all_codes"], dtype=np.int32)
        provenance = {}
        for k in (
            "kappa", "num_states", "model_type", "mean_duration",
            "train_indices", "test_indices",
        ):
            if k in data.files:
                provenance[k] = data[k]

    n_clips, T = all_codes.shape
    if clip_subset is None:
        clip_subset = list(range(n_clips))

    syllables_dict: dict[str, np.ndarray] = {}
    for ci in clip_subset:
        if ci < 0 or ci >= n_clips:
            raise IndexError(
                f"clip {ci} out of range [0, {n_clips}) for codes_path={p}"
            )
        syllables_dict[f"clip_{ci}"] = all_codes[ci].copy()

    log.info(
        f"Loaded KPMS codes: {p}; {len(syllables_dict)} clips × T={T} frames; "
        f"num_codes={int(all_codes.max()) + 1}; "
        f"provenance keys = {sorted(provenance.keys())}"
    )
    return syllables_dict, provenance


def batch_for_shimoseq(
    syllables_dict: dict[str, np.ndarray],
    seg_length: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Wrap ``state_moseq.batch`` so the sweep can call it without
    importing ``state_moseq`` directly.

    Returns:
        ``(seqs, mask, metadata)``:

        * ``seqs`` shape ``[B, seg_length]`` int — chunked syllable ids.
        * ``mask`` shape ``[B, seg_length]`` bool — valid timepoint mask.
        * ``metadata`` opaque tuple used by ``state_moseq.unbatch`` to
          reconstruct per-clip arrays after fitting.
    """
    import state_moseq as sm

    seqs, mask, metadata = sm.batch(syllables_dict, seg_length=seg_length)
    return np.asarray(seqs), np.asarray(mask), metadata
