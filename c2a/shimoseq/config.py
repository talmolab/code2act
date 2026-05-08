"""shMoSeq hyperparameter configuration.

Mirror of :mod:`c2a.kpms.config`.  The hyperparameters are those exposed
by ``state_moseq.hhmm_efficient`` per the
`tutorial <https://state-moseq.readthedocs.io/en/latest/example_data_tutorial.html>`_.
"""

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ShimoseqHyperparams:
    """Hyperparameters for a single shMoSeq fit.

    The full ``hypparams`` dict consumed by
    ``state_moseq.hhmm_efficient.{initialize_params, resample_states,
    resample_params}`` requires the following keys:

    * ``n_syllables`` — input vocabulary size; **derived from the data**
      at fit time (we pass it through to the dict, not stored here).
    * ``n_states`` — upper bound on hierarchical states.
    * ``trans_beta`` / ``trans_kappa`` — Dirichlet base / stickiness for
      the transition prior on ``z``.
    * ``emission_base_sigma`` / ``emission_biases_sigma`` — emission
      Gaussian-prior std on the base + per-state biases.
    * ``emission_gd_iters`` / ``emission_gd_lr`` — gradient-descent
      hyperparameters for the Laplace-approximation step on emission
      params during Gibbs sampling.

    Defaults match the example notebook
    (``state-moseq/docs/source/example_data_tutorial.ipynb``).
    """

    # Swept dimensions
    n_states: int = 10
    trans_kappa: float = 6e5

    # Held fixed across cells (also overridable from YAML)
    trans_beta: float = 1.0
    emission_base_sigma: float = 1.0
    emission_biases_sigma: float = 1.0
    emission_gd_iters: int = 1000
    emission_gd_lr: float = 5e-3

    # Outer training-loop knobs
    num_iters: int = 100
    seg_length: int = 256


def hyperparams_to_dict(hp: ShimoseqHyperparams) -> dict[str, Any]:
    """Convert hyperparams dataclass to a plain dict."""
    return asdict(hp)
