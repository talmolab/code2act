"""Sweep dispatcher — routes to the per-model implementation by config.

The user picks the model by setting the ``model`` field in the YAML
config:

* ``model: kpms`` (default) → runs
  :func:`c2a.codegen.kpms.sweep_impl.run_sweep`, which fits Keypoint-MoSeq models
  across a (κ × num_states × seed) grid and selects per Weinreb 2024
  (median state duration target → EML rank).

The lazy import is deliberate — the KPMS module sets ``jax_enable_x64``
at import time, and we only want that side effect when a sweep actually
runs.

Usage::

    # KPMS (sweeps κ, num_states, seeds)
    python -m c2a.codegen.run_sweep --config c2a/configs/kpms_sweep.yaml
"""

import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf

from c2a.paths import get_c2a_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_config(config_path: str | None) -> dict:
    """Resolve a YAML config to a plain dict (with ``${oc.env:...}`` expanded).

    Default config: ``c2a/configs/kpms_sweep.yaml`` (preserves the old
    pre-dispatcher behavior, where ``python -m c2a.codegen.run_sweep`` ran a
    KPMS sweep).
    """
    if config_path is None:
        config_path = str(get_c2a_root() / "configs" / "kpms_sweep.yaml")
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep dispatcher (chosen by YAML's `model` field)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a sweep YAML")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    model = str(cfg.get("model", "kpms")).lower()

    if model == "kpms":
        from c2a.codegen.kpms.sweep_impl import run_sweep
        log.info("=== Dispatcher: routing to c2a.codegen.kpms.sweep_impl.run_sweep ===")
    else:
        raise ValueError(
            f"Unknown model {model!r}. Set `model: kpms` "
            f"in the YAML config (file: {args.config})."
        )

    return run_sweep(cfg)


if __name__ == "__main__":
    main()
