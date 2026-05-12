"""Hierarchical state-MoSeq (shMoSeq) layer on top of KPMS syllables.

Implements `Weinreb, Kannan, ..., Linderman (Neuron, 2026) — "Spontaneous
behavior is a succession of self-directed tasks" <https://www.cell.com/
neuron/fulltext/S0896-6273(25)00894-3>`_.

The model fits a hierarchical hidden Markov model on top of a low-level
syllable sequence (e.g., the Keypoint-MoSeq output in
``c2a.outputs.kpms_sweep.best_codes.npz``) to discover higher-level
"task" states.  Same convention as :mod:`c2a.codegen.kpms`:

* :mod:`c2a.codegen.shimoseq.config` — :class:`ShimoseqHyperparams`
* :mod:`c2a.codegen.shimoseq.fit_shimoseq` — single-fit wrapper around
  ``state_moseq.hhmm_efficient``
* :mod:`c2a.codegen.shimoseq.syllable_loader` — load ``best_codes.npz`` and
  batch via ``state_moseq.batch``
* :mod:`c2a.codegen.shimoseq.sweep_impl` — full sweep orchestration with
  paper-faithful selection (lagged-MI → trans_kappa filter → Rand-score
  for n_states).  Invoked by ``c2a.codegen.run_sweep`` when the YAML has
  ``model: shimoseq``.
* :mod:`c2a.codegen.shimoseq.render_topk_bands` — band-plot of top-K shMoSeq
  state sequences over the manual-label ground truth.
"""
