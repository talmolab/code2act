"""Cross-model contract test: save → load → resample round-trip.

Verifies that every generator obeys the ``CodeGenerator`` interface:
the same RNG produces the same samples before and after a save/load
cycle. Catches silent serialization bugs (missing leaves, wrong shapes,
forgotten state).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from c2a.ar_codegen.models.base import CodeGenerator


def _train_data(num_codes: int = 4, n_clips: int = 6, T: int = 20):
    rng = np.random.RandomState(0)
    return (
        rng.randint(0, num_codes, size=(n_clips, T)).astype(np.int32),
        rng.randint(0, num_codes, size=(2, T)).astype(np.int32),
    )


@pytest.mark.parametrize("model_name", ["gpt2", "mamba", "s5", "hmm"])
def test_save_load_roundtrip(model_name):
    num_codes = 4
    train_codes, val_codes = _train_data(num_codes=num_codes, T=20)

    if model_name == "gpt2":
        from c2a.ar_codegen.models.gpt2 import GPT2Generator, TransformerConfig
        gen = GPT2Generator(
            num_codes,
            config=TransformerConfig(
                vocab_size=num_codes + 1, emb_dim=16, num_heads=2, num_layers=1,
                qkv_dim=16, mlp_dim=32, max_len=32, dropout_rate=0.0,
            ),
        )
        cfg = OmegaConf.create({
            "batch_size": 2, "num_epochs": 1, "learning_rate": 1e-3,
            "weight_decay": 0.0, "log_every": 1, "seed": 0,
        })
    elif model_name == "mamba":
        from c2a.ar_codegen.models.mamba import MambaGenerator, MambaConfig
        gen = MambaGenerator(
            num_codes,
            config=MambaConfig(
                vocab_size=num_codes + 1, d_model=16, n_layers=1, d_state=4,
                d_conv=3, expand=2, max_len=32, dropout_rate=0.0,
            ),
        )
        cfg = OmegaConf.create({
            "batch_size": 2, "num_epochs": 1, "learning_rate": 1e-3,
            "weight_decay": 0.0, "log_every": 1, "seed": 0,
        })
    elif model_name == "s5":
        from c2a.ar_codegen.models.s5 import S5Generator, S5Config
        gen = S5Generator(
            num_codes,
            config=S5Config(
                vocab_size=num_codes + 1, H=16, P=8, n_layers=1,
                max_len=32, dropout_rate=0.0, bidirectional=False,
            ),
        )
        cfg = OmegaConf.create({
            "batch_size": 2, "num_epochs": 1, "learning_rate": 1e-3,
            "weight_decay": 0.0, "log_every": 1, "seed": 0,
        })
    else:  # hmm
        from c2a.ar_codegen.models.hmm import HMMGenerator
        gen = HMMGenerator(num_codes)
        cfg = OmegaConf.create({
            "num_states_sweep": [2], "num_em_iters": 3,
            "train_ratio": 0.7, "seed": 0,
        })

    gen.fit(train_codes, val_codes, cfg, seed=0, wandb_run=None)

    # Sample with the *generator*, save, load, sample again with the same
    # seed — should match.
    seed = 123
    s1 = gen.sample(n_seq=3, length=12, seed=seed)
    assert s1.shape == (3, 12) and s1.dtype == np.int32
    assert (s1 >= 0).all() and (s1 < num_codes).all()

    with tempfile.TemporaryDirectory() as tmp:
        gen.save(tmp)
        loaded = CodeGenerator.from_checkpoint(Path(tmp))
        assert loaded.MODEL_TYPE == model_name
        assert loaded.num_codes == num_codes

        s2 = loaded.sample(n_seq=3, length=12, seed=seed)
        np.testing.assert_array_equal(s1, s2)
