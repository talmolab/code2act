"""Data loading + batching for AR code-sequence training.

Wraps :func:`c2a.codegen.code_sources.load_codes` so AR training inherits the
exact same code-source dispatch (kpms_raw / kpms_merged /
gt_directional_turn / random) used by the main decoder training. The
clip-index alignment requirement is preserved: the npz's stored
train/test indices must match the balanced split JSON.

This module is the only place in ``ar_codegen/`` that reads from the
balanced-split JSON or constructs a synthetic ``cfg`` for
``load_codes`` — keeps the model classes ignorant of how codes arrived.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from c2a.codegen.code_sources import CodesPayload, load_codes
from c2a.paths import get_repo_root

log = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Materialized training inputs for an AR fit.

    Attributes:
        payload: Original :class:`CodesPayload` (carries provenance).
        train_codes: ``[n_train, T]`` int32.
        val_codes:   ``[n_val,   T]`` int32 (test split repurposed as val).
        num_codes:   Codebook size.
        n_frames:    Sequence length T.
    """

    payload: CodesPayload
    train_codes: np.ndarray
    val_codes: np.ndarray
    num_codes: int
    n_frames: int


def load_training_data(cfg: DictConfig) -> TrainingData:
    """Load codes for AR training from a Hydra config.

    Expected config keys (under ``cfg.data``):

    * ``code_source``: ``kpms_raw`` | ``kpms_merged`` | ``gt_directional_turn``.
    * ``codes_path`` (KPMS sources only): path to ``best_codes.npz``.
    * ``labels_csv_path`` (GT source only): path to TopoMimic CSV.
    * ``balanced_split_path``: path to the JSON used by the decoder
      training, so the same train/test partition is used here.
    * ``n_frames``: clip length in frames (e.g. ``250``).
    * ``num_codes_for_random``: only consulted if ``codes_path`` is null
      (random fallback for smoke tests).
    """
    data_cfg = cfg.data

    # Balanced split — required so AR training and decoder training see
    # the same partition. If a user wants a different split they can point
    # this at a different JSON; null falls back to a seeded random split.
    split_path = data_cfg.get("balanced_split_path", None)
    if split_path is None:
        raise ValueError(
            "data.balanced_split_path is required for AR training so the "
            "same partition is used as in decoder training. "
            "Default: data/rodent/rodent_balanced_splits.json."
        )
    sp = Path(split_path)
    if not sp.is_absolute():
        sp = get_repo_root() / sp
    if not sp.exists():
        raise FileNotFoundError(
            f"data.balanced_split_path={split_path} does not resolve "
            f"(tried {sp})."
        )
    with open(sp) as f:
        splits = json.load(f)
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)

    n_frames = int(data_cfg.n_frames)

    # Build a synthetic cfg that load_codes() understands. load_codes
    # reads kpms_config.{code_source, codes_path, labels_csv_path,
    # csv_to_h5_subsample} and network_config.num_codes (only for the
    # random fallback).
    synthetic = OmegaConf.create(
        {
            "kpms_config": {
                "code_source": str(data_cfg.code_source),
                "codes_path": data_cfg.get("codes_path", None),
                "labels_csv_path": data_cfg.get("labels_csv_path", None),
                "csv_to_h5_subsample": int(data_cfg.get("csv_to_h5_subsample", 2)),
            },
            "network_config": {
                "num_codes": int(data_cfg.get("num_codes_for_random", 32)),
            },
        }
    )

    payload = load_codes(synthetic, train_indices, test_indices, n_frames)

    log.info(
        "AR training data: code_source=%s, num_codes=%d, n_train=%d, "
        "n_val=%d, T=%d",
        payload.code_source, payload.num_codes,
        payload.train_codes.shape[0], payload.test_codes.shape[0], n_frames,
    )

    return TrainingData(
        payload=payload,
        train_codes=payload.train_codes,
        val_codes=payload.test_codes,
        num_codes=payload.num_codes,
        n_frames=n_frames,
    )


def make_batches(
    codes: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> list[np.ndarray]:
    """Slice ``[N, T]`` into per-epoch ``[B, T]`` mini-batches.

    Returns a list of int32 arrays. The last batch is dropped if it
    would be smaller than ``batch_size`` (cleaner JIT shapes than
    runtime-variable batch dims) — *except* when ``N < batch_size``,
    in which case we emit one short batch rather than silently
    training on nothing.
    """
    n = codes.shape[0]
    if n == 0:
        return []
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    if n < batch_size:
        # Don't silently no-op when the dataset is smaller than a batch.
        # Single short batch keeps training functional; JIT will
        # recompile but for a one-shot tiny dataset that's fine.
        return [codes[idx].astype(np.int32, copy=False)]
    batches = []
    for start in range(0, n - batch_size + 1, batch_size):
        sl = idx[start : start + batch_size]
        batches.append(codes[sl].astype(np.int32, copy=False))
    return batches
