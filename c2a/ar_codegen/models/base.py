"""Abstract base class for code-sequence generators.

All four generators (GPT-2 XS, Mamba, S5, HMM) implement this interface so
``c2a/experiments/run_code_generation.py`` can treat them uniformly:

    gen = CodeGenerator.from_checkpoint(path)
    seqs = gen.sample(n_seq=100, length=250, seed=42)

A generator's saved directory contains:

* ``meta.json`` — model_type, num_codes, code_source, config snapshot.
* Model-specific files (params .msgpack for neural models, ``hmm.npz`` for
  the HMM). The ``model_type`` field in ``meta.json`` is the dispatch key.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


META_FILENAME = "meta.json"


@dataclass
class GeneratorMeta:
    """Provenance metadata persisted alongside each checkpoint."""

    model_type: str  # "gpt2" | "mamba" | "s5" | "hmm"
    num_codes: int
    code_source: str  # echoes CodesPayload.code_source
    n_frames_per_clip: int  # context window the model was trained at
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_type": self.model_type,
                "num_codes": int(self.num_codes),
                "code_source": str(self.code_source),
                "n_frames_per_clip": int(self.n_frames_per_clip),
                "config_snapshot": self.config_snapshot,
                "provenance": self.provenance,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, s: str) -> "GeneratorMeta":
        d = json.loads(s)
        return cls(
            model_type=str(d["model_type"]),
            num_codes=int(d["num_codes"]),
            code_source=str(d["code_source"]),
            n_frames_per_clip=int(d["n_frames_per_clip"]),
            config_snapshot=dict(d.get("config_snapshot", {})),
            provenance=dict(d.get("provenance", {})),
        )


class CodeGenerator(abc.ABC):
    """Unified train/save/load/sample interface for code-sequence priors.

    Subclasses implement: ``fit``, ``sample``, ``_save_model_files``,
    ``_load_model_files``. The :meth:`save` / :meth:`load` /
    :meth:`from_checkpoint` plumbing — writing ``meta.json`` and
    dispatching on ``model_type`` — lives here.

    Attributes:
        num_codes: Vocabulary size of the underlying code source.
        meta: Provenance metadata (populated at fit/load time).
    """

    MODEL_TYPE: str = "abstract"  # overridden by each subclass

    def __init__(self, num_codes: int):
        if num_codes <= 0:
            raise ValueError(f"num_codes must be positive, got {num_codes}")
        self.num_codes = int(num_codes)
        self.meta: GeneratorMeta | None = None

    # ------------------------------------------------------------------
    # Core abstract API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        train_codes: np.ndarray,
        val_codes: np.ndarray,
        cfg: Any,
        seed: int = 0,
        wandb_run: Any = None,
    ) -> None:
        """Fit the model on per-frame code sequences.

        Args:
            train_codes: ``[n_train, T]`` int array in ``[0, num_codes)``.
            val_codes: ``[n_val, T]`` validation set (may be empty).
            cfg: A model-specific configuration object (Hydra DictConfig
                or OmegaConf node). Each subclass documents the keys it
                reads.
            seed: PRNG seed.
            wandb_run: An optional wandb Run object. When provided,
                training loss / perplexity / sample diagnostics are
                logged via ``wandb.log`` as the fit progresses.
        """

    @abc.abstractmethod
    def sample(
        self,
        n_seq: int,
        length: int,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Sample ``n_seq`` code sequences of the given length.

        Returns:
            ``np.int32`` array of shape ``[n_seq, length]`` with values
            in ``[0, num_codes)``.
        """

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the model to ``path`` (a directory).

        Writes ``meta.json`` and delegates the actual params to
        :meth:`_save_model_files`.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.meta is None:
            # The fit() implementation should have populated this. We
            # don't silently fabricate one — that would lose code-source
            # provenance.
            raise RuntimeError(
                f"{type(self).__name__}.save() called before .meta was "
                "populated by .fit(). Subclass must set self.meta inside fit()."
            )
        (path / META_FILENAME).write_text(self.meta.to_json())
        self._save_model_files(path)

    @abc.abstractmethod
    def _save_model_files(self, path: Path) -> None:
        """Write model-specific files into ``path``. ``meta.json`` already done."""

    @classmethod
    @abc.abstractmethod
    def _load_model_files(cls, path: Path, meta: GeneratorMeta) -> "CodeGenerator":
        """Reconstruct an instance from ``path`` given pre-loaded meta."""

    @classmethod
    def load(cls, path: str | Path) -> "CodeGenerator":
        """Load a checkpoint of *this exact subclass* from ``path``.

        Use :meth:`from_checkpoint` if you don't statically know which
        subclass produced the directory.
        """
        path = Path(path)
        meta = GeneratorMeta.from_json((path / META_FILENAME).read_text())
        if meta.model_type != cls.MODEL_TYPE:
            raise ValueError(
                f"{cls.__name__}.load expected meta.model_type="
                f"{cls.MODEL_TYPE!r} but checkpoint has "
                f"{meta.model_type!r}. Use CodeGenerator.from_checkpoint "
                "to dispatch automatically."
            )
        gen = cls._load_model_files(path, meta)
        gen.meta = meta
        return gen

    @staticmethod
    def from_checkpoint(path: str | Path) -> "CodeGenerator":
        """Dispatch on ``meta.model_type`` to load the right subclass.

        This is the entry point ``run_code_generation.py`` uses — callers
        don't need to know which architecture is on disk.
        """
        path = Path(path)
        meta = GeneratorMeta.from_json((path / META_FILENAME).read_text())
        # Local imports avoid pulling Flax/JAX into this base module unless
        # actually needed.
        if meta.model_type == "gpt2":
            from c2a.ar_codegen.models.gpt2 import GPT2Generator
            return GPT2Generator.load(path)
        if meta.model_type == "mamba":
            from c2a.ar_codegen.models.mamba import MambaGenerator
            return MambaGenerator.load(path)
        if meta.model_type == "s5":
            from c2a.ar_codegen.models.s5 import S5Generator
            return S5Generator.load(path)
        if meta.model_type == "hmm":
            from c2a.ar_codegen.models.hmm import HMMGenerator
            return HMMGenerator.load(path)
        raise ValueError(
            f"Unknown model_type={meta.model_type!r} in {path / META_FILENAME}. "
            "Expected one of: gpt2, mamba, s5, hmm."
        )
