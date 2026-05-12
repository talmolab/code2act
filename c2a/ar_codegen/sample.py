"""Hydra entrypoint for sampling from a trained AR code-sequence generator.

Usage:

    python -m c2a.ar_codegen.sample \\
        checkpoint=outputs/ar_codegen/gpt2/260512_140000_user \\
        n=100 length=250 seed=42 \\
        out=outputs/ar_codegen/samples/gpt2_kpms_raw.npz

If ``out`` is omitted, samples land at
``<checkpoint>/samples_n{n}_T{length}_s{seed}.npz``.

The output ``.npz`` schema matches the existing pattern used by
``c2a/codegen/`` so downstream scripts (``run_code_generation.py``)
can consume it interchangeably with HMM/ARHMM sampled outputs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import numpy as np
from omegaconf import DictConfig

from c2a.ar_codegen.models.base import CodeGenerator

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="sample")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    ckpt = Path(str(cfg.checkpoint))
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint={ckpt} does not exist.")

    gen = CodeGenerator.from_checkpoint(ckpt)
    log.info(
        "Loaded %s generator (num_codes=%d, n_frames=%d, code_source=%s).",
        type(gen).__name__, gen.num_codes,
        gen.meta.n_frames_per_clip if gen.meta else -1,
        gen.meta.code_source if gen.meta else "?",
    )

    n = int(cfg.n)
    length = int(cfg.length)
    seed = int(cfg.seed)
    temperature = float(cfg.temperature)

    seqs = gen.sample(n_seq=n, length=length, seed=seed, temperature=temperature)
    assert seqs.shape == (n, length)
    assert seqs.dtype == np.int32

    out_path = cfg.get("out", None)
    if out_path is None:
        out_path = ckpt / f"samples_n{n}_T{length}_s{seed}.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        sequences=seqs,
        model_type=gen.meta.model_type if gen.meta else "unknown",
        num_codes=int(gen.num_codes),
        code_source=gen.meta.code_source if gen.meta else "unknown",
        checkpoint_path=str(ckpt),
        temperature=float(temperature),
        seed=int(seed),
        length=int(length),
        n=int(n),
    )
    log.info(
        "Wrote %d sequences (T=%d) to %s. Value range: [%d, %d].",
        n, length, out_path, int(seqs.min()), int(seqs.max()),
    )


if __name__ == "__main__":
    main()
