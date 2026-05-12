"""Fit-then-sample test for HMMGenerator on tiny synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from c2a.ar_codegen.models.hmm import HMMGenerator


def _synthetic_codes(n_clips: int, T: int, num_codes: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    # Two-state alternating syllable structure — easier to fit than uniform random.
    seqs = np.zeros((n_clips, T), dtype=np.int32)
    for i in range(n_clips):
        state = 0
        for t in range(T):
            seqs[i, t] = rng.randint(0, num_codes // 2) if state == 0 else \
                         rng.randint(num_codes // 2, num_codes)
            if rng.rand() < 0.05:
                state = 1 - state
    return seqs


def test_fit_then_sample_in_range():
    num_codes = 4
    train_codes = _synthetic_codes(8, 25, num_codes, seed=0)
    val_codes = _synthetic_codes(2, 25, num_codes, seed=1)
    cfg = OmegaConf.create({
        "num_states_sweep": [2, 3],
        "num_em_iters": 5,
        "train_ratio": 0.7,
        "seed": 0,
    })
    gen = HMMGenerator(num_codes=num_codes)
    gen.fit(train_codes, val_codes, cfg, seed=0, wandb_run=None)
    seqs = gen.sample(n_seq=4, length=20, seed=42)
    assert seqs.shape == (4, 20)
    assert seqs.dtype == np.int32
    assert (seqs >= 0).all() and (seqs < num_codes).all()
    assert gen.meta is not None
    assert gen.meta.model_type == "hmm"
    assert gen.meta.num_codes == num_codes
