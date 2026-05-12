"""Autoregressive code-sequence generators for Code2Act.

This subpackage hosts model classes that fit a sequence prior over per-frame
KPMS code sequences (or any compatible code source from
:mod:`c2a.code_sources`) and emit new code sequences for downstream rollout
through the trained MoSeq decoder policy.

Four generators live here, sharing the unified
:class:`c2a.ar_codegen.models.base.CodeGenerator` interface
(``fit`` / ``sample`` / ``save`` / ``load``):

* :class:`c2a.ar_codegen.models.gpt2.GPT2Generator`  — small decoder-only
  transformer (adapted from ``flax/examples/lm1b``).
* :class:`c2a.ar_codegen.models.mamba.MambaGenerator` — selective SSM
  (ported from ``ekellbuch/deep_ssm/mixers/mamba_simple.py``).
* :class:`c2a.ar_codegen.models.s5.S5Generator` — Linderman lab S5
  (vendored from ``lindermanlab/S5``).
* :class:`c2a.ar_codegen.models.hmm.HMMGenerator` — dynamax
  ``CategoricalHMM`` (moved here from ``experiments/run_code_generation``).

The quick-sample baselines (empirical transition matrix, ARHMM-level-1,
ARHMM-level-2, controls) remain inline in
``c2a/experiments/run_code_generation.py`` because they require no
training or checkpointing.
"""
