"""Nature-style matplotlib configuration and plotting helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from c2a.codegen.shared.mlb_labels import MLB_COLORS, MLB_LABELS, MLB_SHORT

# ---------------------------------------------------------------------------
# Colorblind-safe palette (Wong, Nature Methods 2011)
# ---------------------------------------------------------------------------
NATURE_COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "vermillion": "#E69F00",
    "sky_blue": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#999999",
}

BEHAVIOR_COLORS = {
    "groom": "#0072B2",
    "walk": "#D55E00",
    "rear": "#009E73",
}

CONDITION_COLORS = {
    "correct": "#009E73",
    "shuffled_step": "#D55E00",
    "shuffled_trajectory": "#0072B2",
}

MODE_COLORS = {
    "code2act": "#0072B2",
    "mimic_mjx": "#D55E00",
}

MODE_LABELS = {
    "code2act": "Code2Act",
    "mimic_mjx": "Mimic-MJX (oracle)",
}


def set_nature_style() -> None:
    """Apply Nature-style matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
        }
    )


def get_trajectory_colors(k: int) -> list[list[float]]:
    """Return *k* qualitative RGBA colours suitable for ghost-body overlay."""
    cmap = plt.cm.get_cmap("tab10")
    return [list(cmap(i % 10)) for i in range(k)]


# ---------------------------------------------------------------------------
# 9-class (D3) canonical behavior palette + legend, shared across every figure
# and all three steering strategies. Behavior = hue family, direction = shade.
# Source of truth for names/colors is ``c2a.codegen.shared.mlb_labels``.
# ---------------------------------------------------------------------------
NINE_CLASS_COLORS: dict[str, str] = {
    name: mpl.colors.to_hex(MLB_COLORS[name]) for name in MLB_LABELS
}
NINE_CLASS_COLORS["unlabeled"] = "#cccccc"


def nine_class_color(label: str) -> str:
    """Hex color for a 9-class label (``"unlabeled"``/unknown -> light gray)."""
    return NINE_CLASS_COLORS.get(label, "#cccccc")


def nine_class_rgb(label: str) -> tuple[float, float, float]:
    """RGB float tuple for a 9-class label (for in-image overlays)."""
    return mpl.colors.to_rgb(nine_class_color(label))


def nine_class_legend_handles(
    *, short: bool = True, include_unlabeled: bool = False
) -> tuple[list, list[str]]:
    """Return ``(handles, labels)`` for a 9-entry behavior legend.

    Usage in a figure script::

        from c2a.experiments.shared.plotting import nine_class_legend_handles
        handles, labels = nine_class_legend_handles()
        fig.legend(handles, labels, ncol=3, loc="lower center")

    Args:
        short: use the compact ``"Walk L"`` display names instead of the
            canonical ``"Walk_Left"`` strings.
        include_unlabeled: append a gray ``"Unlabeled"`` swatch.
    """
    from matplotlib.patches import Patch

    handles: list = []
    labels: list[str] = []
    for name in MLB_LABELS:
        handles.append(
            Patch(facecolor=NINE_CLASS_COLORS[name], edgecolor="none")
        )
        labels.append(MLB_SHORT[name] if short else name)
    if include_unlabeled:
        handles.append(Patch(facecolor=NINE_CLASS_COLORS["unlabeled"], edgecolor="none"))
        labels.append("Unlabeled")
    return handles, labels


def get_code_colormap(num_codes: int) -> np.ndarray:
    """Return ``[num_codes, 3]`` uint8 RGB array for code timeline bars.

    Uses Nature-friendly Tableau palettes (``tab10`` / ``tab20`` /
    ``tab20`` + ``tab20b`` for up to 40 codes) instead of garish HSV. For
    >40 codes the palette cycles modulo 40 (acceptable visual collisions).
    """
    if num_codes <= 10:
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, 10))[:num_codes, :3]
    elif num_codes <= 20:
        colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, 20))[:num_codes, :3]
    else:
        a = plt.cm.get_cmap("tab20")(np.linspace(0, 1, 20))[:, :3]
        b = plt.cm.get_cmap("tab20b")(np.linspace(0, 1, 20))[:, :3]
        pool = np.concatenate([a, b], axis=0)  # 40 colors
        idx = np.arange(num_codes) % pool.shape[0]
        colors = pool[idx]
    return (colors * 255).astype(np.uint8)
