"""Canonical definition of the 9-label MLB (Manually Labeled Behaviors) codebook.

MLB is one of the three Code2Act *steering strategies* (alongside ``kpms`` and
``kvae``). Unlike the data-discovered tokenizers, its codes are specified a
priori from a manual behavior lexicon: the four manually annotated behaviors
(Immobile, Rear, Turn, Walk) crossed with a Left / Right / Straight *direction*
that is derived kinematically from the reference body's net heading change over
each behavior bout (see :mod:`c2a.codegen.mlb_segmenter`).

Direction is asymmetric by construction, matching what the data supports:

* ``Walk`` and ``Rear`` take all three directions (Left, Right, Straight).
* ``Turn`` is never Straight (a turn is a heading change), so it takes only
  Left / Right.
* ``Immobile`` carries no direction (a single code).

This yields exactly **9** flat codes (no n-grams, no sentinel — every reference
frame is manually behavior-labeled). This module is the single source of truth
for the code order, human-readable names, the color palette used across every
figure, and the 3x3 grid layout used by the per-category atlas (Fig. 3c).
"""
from __future__ import annotations

import numpy as np

from c2a.codegen.shared.label_alignment import (
    BEHAVIOR_TO_INT,
    DIRECTION_TO_INT,
    UNLABELED,
)

# ---------------------------------------------------------------------------
# Canonical code order (code id == index). Behavior-major, direction-minor.
# ---------------------------------------------------------------------------
MLB_LABELS: tuple[str, ...] = (
    "Walk_Left",       # 0
    "Walk_Right",      # 1
    "Walk_Straight",   # 2
    "Rear_Left",       # 3
    "Rear_Right",      # 4
    "Rear_Straight",   # 5
    "Turn_Left",       # 6
    "Turn_Right",      # 7
    "Immobile",        # 8
)
MLB_NUM_CODES: int = len(MLB_LABELS)
MLB_LABEL_TO_INT: dict[str, int] = {n: i for i, n in enumerate(MLB_LABELS)}

# Short display names for plot tick labels / overlays.
MLB_SHORT: dict[str, str] = {
    "Walk_Left": "Walk L", "Walk_Right": "Walk R", "Walk_Straight": "Walk S",
    "Rear_Left": "Rear L", "Rear_Right": "Rear R", "Rear_Straight": "Rear S",
    "Turn_Left": "Turn L", "Turn_Right": "Turn R", "Immobile": "Immobile",
}

# ---------------------------------------------------------------------------
# Color palette: behavior = hue family, direction = shade (Left dark -> Right
# light). Stable across every figure so a category reads the same everywhere.
# ---------------------------------------------------------------------------
MLB_COLORS: dict[str, list[float]] = {
    "Walk_Left":      [0.10, 0.25, 0.65],
    "Walk_Right":     [0.55, 0.70, 0.95],
    "Walk_Straight":  [0.30, 0.50, 0.85],
    "Rear_Left":      [0.10, 0.45, 0.20],
    "Rear_Right":     [0.55, 0.80, 0.55],
    "Rear_Straight":  [0.25, 0.65, 0.35],
    "Turn_Left":      [0.80, 0.40, 0.10],
    "Turn_Right":     [0.95, 0.70, 0.35],
    "Immobile":       [0.55, 0.55, 0.55],
}

# 3x3 grid layout (rows = behavior, cols = Left / center / Right) used by the
# per-category atlas. Immobile sits in the center of the Turn row.
MLB_GRID: tuple[tuple[str, ...], ...] = (
    ("Walk_Left",  "Walk_Straight", "Walk_Right"),
    ("Rear_Left",  "Rear_Straight", "Rear_Right"),
    ("Turn_Left",  "Immobile",      "Turn_Right"),
)


def mlb_color_array() -> np.ndarray:
    """Return an ``(9, 3)`` uint8 RGB array indexed by code id (overlay use)."""
    out = np.zeros((MLB_NUM_CODES, 3), dtype=np.uint8)
    for i, name in enumerate(MLB_LABELS):
        out[i] = (np.clip(MLB_COLORS[name], 0, 1) * 255).astype(np.uint8)
    return out


def mlb_code(behavior_name: str, direction_idx: int) -> int:
    """Map a (behavior, kinematic-direction) pair to an MLB code id.

    Args:
        behavior_name: one of ``Immobile`` / ``Rear`` / ``Turn`` / ``Walk``.
        direction_idx: a :data:`DIRECTION_TO_INT` value (Left=0, Right=1,
            Straight=2). Ignored for ``Immobile``; ``Turn`` must be Left/Right.

    Returns:
        The MLB code id in ``0..8``.

    Raises:
        ValueError: unknown behavior, or ``Turn`` with a Straight direction.
    """
    if behavior_name == "Immobile":
        return MLB_LABEL_TO_INT["Immobile"]
    if direction_idx == DIRECTION_TO_INT["Left"]:
        d = "Left"
    elif direction_idx == DIRECTION_TO_INT["Right"]:
        d = "Right"
    elif direction_idx == DIRECTION_TO_INT["Straight"]:
        d = "Straight"
    else:
        raise ValueError(f"bad direction_idx {direction_idx}")
    if behavior_name == "Turn" and d == "Straight":
        raise ValueError("Turn cannot be Straight")
    if behavior_name not in ("Walk", "Rear", "Turn"):
        raise ValueError(f"unknown behavior {behavior_name!r}")
    return MLB_LABEL_TO_INT[f"{behavior_name}_{d}"]
