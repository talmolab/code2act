"""Environment compatibility shims for the transfer package.

The venv pins ``orbax-checkpoint`` (0.11.24) against ``jax`` 0.5.3, where
``jax.monitoring.record_scalar`` no longer exists; orbax calls it during
checkpoint restore. The decoder/encoder training scripts and the code-grid
experiments already monkey-patch a no-op (see ``c2a/train_moseq_decoder.py`` and
``c2a/experiments/code_grid/run_code_grid_rollouts.py``). Any transfer module
that restores an Orbax checkpoint must apply the same shim **before** the first
orbax restore, so we centralize it here and import it from
``c2a.transfer.__init__``.
"""

import jax

if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None  # type: ignore[attr-defined]
