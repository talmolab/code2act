"""Single source of truth for code2act filesystem paths.

`c2a` is pip-installable. When the package is installed in editable mode
(``uv pip install -e .`` from repo root), ``Path(__file__).resolve().parent``
is ``code2act/c2a/`` and ``.parent.parent`` is the repo root. Non-editable
installs land the package in ``site-packages/`` — in that case the repo root
cannot be inferred from ``__file__`` and must be supplied via the
``C2A_ROOT`` env var.

All code that needs absolute paths (data, checkpoints, XML asset) should go
through ``get_repo_root()``. Never recompute ``parent.parent`` arithmetic
inline.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["get_repo_root", "get_c2a_root", "get_asset_path"]


def get_c2a_root() -> Path:
    """Return the path to the installed c2a package directory."""
    return Path(__file__).resolve().parent


def get_repo_root() -> Path:
    """Return the code2act repository root.

    Resolution order:
      1. ``C2A_ROOT`` env var if set.
      2. ``Path(__file__).resolve().parent.parent`` — correct for editable
         installs.
    """
    env = os.environ.get("C2A_ROOT")
    if env:
        return Path(env).resolve()
    return get_c2a_root().parent


def get_asset_path(name: str) -> Path:
    """Return the absolute path to a bundled asset in ``c2a/assets/``."""
    return get_c2a_root() / "assets" / name
