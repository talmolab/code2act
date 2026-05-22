"""Vendored TopoMimic-annotation auto-labeler.

Upstream: https://github.com/KevinBian107/TopoMimic-annotation
Vendored on 2026-05-17 from commit at scratch/2026-05-16-topomimic-labeler-integration/repo/.

We keep the two upstream modules (``auto_labeler``, ``auto_direction``)
verbatim except for the relative import in ``auto_direction.py``. New
numpy-friendly wrappers live in ``api.py`` so callers can label rollout
qpos arrays directly without writing CSVs.
"""

from .api import (
    label_rollout_body,
    DEFAULT_LABELER_FPS,
    resample_qpos,
)
from .auto_labeler import load_config, DEFAULT_CONFIG

__all__ = [
    "label_rollout_body",
    "DEFAULT_LABELER_FPS",
    "resample_qpos",
    "load_config",
    "DEFAULT_CONFIG",
]
