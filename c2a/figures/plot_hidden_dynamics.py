"""RNN hidden state dynamics visualization (3D PCA).

Creates Nature-style figures with N panels (one per label):
  - Scatter: point clouds, one label per panel, shared PCA axes
  - Trajectory: lines colored by time, one label per panel, shared PCA axes
  - Combined-panel versions of both

PCA is fitted on ALL labels jointly so the coordinate system is shared.

Auto-detects two npz formats:
  - Legacy (run_hidden_dynamics.py output): hidden_walk / hidden_immobility / hidden_rear
  - Labeled (run_labeled_hidden_dynamics.py): hidden_<combined-label> like hidden_Walk_Straight

Usage::

    # Legacy 3-class (default file: figures/data/hidden_dynamics.npz)
    python -m c2a.figures.plot_hidden_dynamics

    # Labeled 7-class
    python -m c2a.figures.plot_hidden_dynamics \\
        --npz c2a/figures/data/hidden_dynamics_labeled.npz \\
        --output_suffix labeled
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "hidden_dynamics"

# ── Legacy aliases (used as defaults / pretty-names when keys match) ────────
LEGACY_BEHAVIORS = ["walk", "immobility", "rear"]
LEGACY_BEHAVIOR_LABELS = {
    "walk": "Walking Code",
    "immobility": "Immobility Code",
    "rear": "Rearing Code",
}
LEGACY_BEHAVIOR_COLORS = {
    "walk": "#D55E00",        # orange (Wong)
    "immobility": "#0072B2",  # blue (Wong)
    "rear": "#009E73",        # green (Wong)
}

# Backward-compat aliases — earlier code in this file references the
# old global names.  Populated dynamically in main() once the labels are
# discovered, but seeded with legacy values here so unit tests still pass.
BEHAVIORS = list(LEGACY_BEHAVIORS)
BEHAVIOR_LABELS = dict(LEGACY_BEHAVIOR_LABELS)
BEHAVIOR_COLORS = dict(LEGACY_BEHAVIOR_COLORS)


def _discover_labels(
    npz, level: str | None = None, label_subset: list[str] | None = None,
) -> list[str]:
    """Return the list of label strings present in ``npz`` as ``hidden_*`` keys.

    Mirrors the discovery in ``plot_hidden_dsa.py`` / ``plot_hidden_rslds.py``:

    * Legacy: keys ``hidden_walk``, ``hidden_immobility``, ``hidden_rear``
      (no ``__``) → labels are the suffixes.
    * Labeled flat: keys ``hidden_Walk_Straight``, ``hidden_Turn_Left`` (no
      ``__`` either, but multi-word) → also legacy-style.
    * Labeled nested: keys ``hidden_<level>__<label>`` (older design,
      now obsolete but still supported) → restrict to ``level``.
    """
    hidden_keys = [k for k in npz.keys() if k.startswith("hidden_")]
    if not hidden_keys:
        raise KeyError("npz has no hidden_<label> keys; nothing to plot.")
    is_nested = any("__" in k for k in hidden_keys)
    out: list[str] = []
    if is_nested:
        eff = level or "behavior"
        prefix = f"hidden_{eff}__"
        for k in hidden_keys:
            if k.startswith(prefix):
                out.append(k[len(prefix):])
    else:
        for k in hidden_keys:
            out.append(k[len("hidden_"):])
    if label_subset is not None:
        keep = set(label_subset)
        out = [l for l in out if l in keep]
    if not out:
        raise ValueError(
            f"After filtering (level={level}, subset={label_subset}) no labels "
            f"matched. Available hidden_* keys: {hidden_keys}"
        )
    # Stable order: legacy three first if present, else alphabetic.
    if not is_nested and any(b in out for b in LEGACY_BEHAVIORS):
        order = [b for b in LEGACY_BEHAVIORS if b in out]
        out = order + sorted(l for l in out if l not in order)
    else:
        out = sorted(out) if is_nested else out
    return out


# Categorical palette to fall back on for unknown labels (Wong + Okabe-Ito,
# colorblind-safe).
_FALLBACK_PALETTE = (
    "#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7",
    "#56B4E9", "#F0E442", "#999999", "#6A3D9A", "#B15928",
    "#1F78B4", "#33A02C",
)


def _label_colors(labels: list[str]) -> dict[str, str]:
    """Map each label to a consistent color.

    Legacy labels keep their canonical Wong colors; new labels cycle
    through ``_FALLBACK_PALETTE`` in a deterministic order.
    """
    out: dict[str, str] = {}
    palette_idx = 0
    for lab in labels:
        if lab in LEGACY_BEHAVIOR_COLORS:
            out[lab] = LEGACY_BEHAVIOR_COLORS[lab]
        else:
            out[lab] = _FALLBACK_PALETTE[palette_idx % len(_FALLBACK_PALETTE)]
            palette_idx += 1
    return out


def _label_display(labels: list[str]) -> dict[str, str]:
    """Pretty label-string lookup. Legacy gets canonical names; combined
    labels get underscores replaced with space."""
    out: dict[str, str] = {}
    for lab in labels:
        if lab in LEGACY_BEHAVIOR_LABELS:
            out[lab] = LEGACY_BEHAVIOR_LABELS[lab]
        else:
            out[lab] = lab.replace("_", " ")
    return out

# View angle
VIEW_ELEV = 20
VIEW_AZIM = 140


# ── Style ────────────────────────────────────────────────────────────────────


def _setup_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_3d_ax(
    ax: plt.Axes,
    var_explained: np.ndarray,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
    show_ylabel: bool = True,
) -> None:
    """Apply clean Nature-style formatting to a 3D axes."""
    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1%})", labelpad=0, fontsize=6)
    if show_ylabel:
        ax.set_ylabel(f"PC 2 ({var_explained[1]:.1%})", labelpad=0, fontsize=6)
    else:
        ax.set_ylabel("")
    ax.set_zlabel(f"PC 3 ({var_explained[2]:.1%})", labelpad=0, fontsize=6)

    ax.tick_params(axis="both", which="major", pad=0, labelsize=4.5)
    ax.tick_params(axis="z", pad=0)

    # Shared limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # Transparent panes with subtle edges
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#d5d5d5")
        pane.set_alpha(0.6)

    # Subtle grid
    ax.grid(True, linewidth=0.2, alpha=0.4, color="#cccccc")

    # Thin axis lines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(0.4)
        axis.line.set_color("#666666")

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def _add_rounded_border(fig: plt.Figure) -> mpatches.FancyBboxPatch:
    fig.patch.set_facecolor("white")
    rect = mpatches.FancyBboxPatch(
        (0.005, 0.005),
        0.99,
        0.99,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#cccccc",
        linewidth=0.6,
        zorder=-1,
    )
    fig.patches.append(rect)
    return rect


def _compute_shared_limits(
    emb_3d: np.ndarray, pad_frac: float = 0.05,
) -> tuple[tuple, tuple, tuple]:
    """Compute shared axis limits across all data."""
    lims = []
    for i in range(3):
        lo, hi = emb_3d[:, i].min(), emb_3d[:, i].max()
        d = (hi - lo) * pad_frac
        lims.append((lo - d, hi + d))
    return tuple(lims)


# ── Data loading ─────────────────────────────────────────────────────────────


def load_hidden_data(
    npz_path: Path | None = None,
    level: str | None = None,
    label_subset: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str], str | None]:
    """Load hidden dynamics NPZ — auto-detects legacy vs. labeled format.

    Returns:
        hiddens: ``{label: ndarray [K, T, hidden_dim]}``
        codes:   ``{label: ndarray [K, T]}``
        labels:  ordered list of label strings
        level_used: ``None`` for legacy / flat-labeled; level prefix for nested
    """
    if npz_path is None:
        npz_path = DATA_DIR / "hidden_dynamics.npz"
        if not npz_path.exists():
            alt = DATA_DIR / "hidden_dynamics_labeled.npz"
            if alt.exists():
                npz_path = alt
                print(f"NOTE: legacy npz absent; using labeled npz {alt.name}")
    raw = np.load(npz_path, allow_pickle=True)

    is_nested = any("__" in k for k in raw.keys() if k.startswith("hidden_"))
    eff_level = (level or "behavior") if is_nested else None

    labels = _discover_labels(raw, level=eff_level, label_subset=label_subset)

    hiddens: dict[str, np.ndarray] = {}
    codes: dict[str, np.ndarray] = {}
    for lab in labels:
        h_key = f"hidden_{eff_level}__{lab}" if eff_level is not None else f"hidden_{lab}"
        c_key = f"codes_{eff_level}__{lab}" if eff_level is not None else f"codes_{lab}"
        if h_key in raw:
            hiddens[lab] = np.array(raw[h_key])
        if c_key in raw:
            codes[lab] = np.array(raw[c_key])
    return hiddens, codes, labels, eff_level


def find_single_code_clips(
    codes: dict[str, np.ndarray],
) -> dict[str, list[int]]:
    """Find clips with exactly 1 unique code (stationary dynamics).

    Returns:
        ``{behavior: [clip_indices]}``
    """
    result = {}
    for beh, code_arr in codes.items():
        indices = []
        for ki in range(code_arr.shape[0]):
            if len(np.unique(code_arr[ki])) == 1:
                indices.append(ki)
        result[beh] = indices
    return result


# ── Embedding ────────────────────────────────────────────────────────────────


def fit_pca_3d(
    all_points: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 3-component PCA. Returns ``([N, 3], variance_explained[3])``."""
    reducer = PCA(n_components=3, random_state=seed)
    result = reducer.fit_transform(all_points)
    var = reducer.explained_variance_ratio_
    print(
        f"  PCA variance explained: "
        f"PC1={var[0]:.1%}, PC2={var[1]:.1%}, PC3={var[2]:.1%} "
        f"(total={sum(var):.1%})"
    )
    return result, var


# ── Behavior-specific colormaps ──────────────────────────────────────────────


def _behavior_cmap(hex_color: str) -> mcolors.LinearSegmentedColormap:
    """Create a light-tint -> full-color sequential colormap."""
    rgb = mcolors.to_rgb(hex_color)
    light = tuple(0.80 + 0.20 * c for c in rgb)
    return mcolors.LinearSegmentedColormap.from_list("beh", [light, rgb])


BEHAVIOR_CMAPS = {beh: _behavior_cmap(c) for beh, c in BEHAVIOR_COLORS.items()}


# ── 3-panel scatter ──────────────────────────────────────────────────────────


def plot_scatter_3d(
    emb_3d: np.ndarray,
    beh_slices: dict[str, tuple[int, int]],
    var_explained: np.ndarray,
    show_behaviors: list[str] | None = None,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """N-panel scatter, one per behavior, shared PCA axes."""
    if show_behaviors is None:
        show_behaviors = BEHAVIORS
    show_behaviors = [b for b in show_behaviors if b in beh_slices]
    n_panels = len(show_behaviors)

    fig = plt.figure(figsize=(3.0 * n_panels, 3.2))
    xlim, ylim, zlim = _compute_shared_limits(emb_3d)

    for pi, beh in enumerate(show_behaviors):
        ax = fig.add_subplot(1, n_panels, pi + 1, projection="3d")
        s, e = beh_slices[beh]
        pts = emb_3d[s:e]

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=BEHAVIOR_COLORS[beh],
            s=6,
            alpha=0.20,
            edgecolors="none",
            rasterized=True,
            depthshade=True,
        )

        ax.set_title(
            BEHAVIOR_LABELS[beh],
            fontsize=8, fontweight="bold", pad=-2,
            color=BEHAVIOR_COLORS[beh],
        )
        _style_3d_ax(ax, var_explained, xlim, ylim, zlim, show_ylabel=(pi == 0))

    fig.suptitle(
        "PCA of RNN Hidden State",
        fontsize=9, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.05)
    rect = _add_rounded_border(fig)
    return fig, rect


# ── 3-panel trajectory ───────────────────────────────────────────────────────


def plot_trajectories_3d(
    emb_3d: np.ndarray,
    beh_slices: dict[str, tuple[int, int]],
    beh_shapes: dict[str, tuple[int, int]],
    var_explained: np.ndarray,
    show_behaviors: list[str] | None = None,
    clip_filter: dict[str, list[int]] | None = None,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """N-panel trajectories, one per behavior, shared PCA axes.

    Each panel shows trajectories colored by time (light -> dark).
    Open circle = start, filled = end.

    Args:
        clip_filter: if provided, ``{behavior: [clip_indices]}`` — only
            plot the listed clips for each behavior.
    """
    if show_behaviors is None:
        show_behaviors = BEHAVIORS
    show_behaviors = [b for b in show_behaviors if b in beh_slices]
    n_panels = len(show_behaviors)

    fig = plt.figure(figsize=(3.0 * n_panels, 3.2))
    xlim, ylim, zlim = _compute_shared_limits(emb_3d)

    for pi, beh in enumerate(show_behaviors):
        ax = fig.add_subplot(1, n_panels, pi + 1, projection="3d")
        K_beh, T_beh = beh_shapes[beh]
        s, _ = beh_slices[beh]
        cmap = BEHAVIOR_CMAPS[beh]
        t_norm = np.linspace(0, 1, T_beh)

        # Which clips to plot
        clip_indices = range(K_beh)
        if clip_filter is not None and beh in clip_filter:
            clip_indices = clip_filter[beh]

        for ki in clip_indices:
            start = s + ki * T_beh
            pts = emb_3d[start : start + T_beh]

            if len(pts) < 2:
                continue

            segments = np.array(
                [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)],
            )
            colors = cmap(t_norm[:-1])
            lc = Line3DCollection(
                segments,
                colors=colors,
                linewidths=0.8,
                alpha=0.65,
            )
            ax.add_collection3d(lc)

            # Start (open) and end (filled)
            ax.scatter(
                [pts[0, 0]], [pts[0, 1]], [pts[0, 2]],
                c="white",
                s=20,
                edgecolors=BEHAVIOR_COLORS[beh],
                linewidths=0.6,
                zorder=5,
                depthshade=False,
            )
            ax.scatter(
                [pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]],
                c=BEHAVIOR_COLORS[beh],
                s=20,
                edgecolors=BEHAVIOR_COLORS[beh],
                linewidths=0.6,
                zorder=5,
                depthshade=False,
            )

        n_shown = len(list(clip_indices))
        ax.set_title(
            f"{BEHAVIOR_LABELS[beh]} (n={n_shown})",
            fontsize=8, fontweight="bold", pad=-2,
            color=BEHAVIOR_COLORS[beh],
        )
        _style_3d_ax(ax, var_explained, xlim, ylim, zlim, show_ylabel=(pi == 0))

    # Shared start/end legend on last panel
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", color="w",
            markeredgecolor="#555555", markeredgewidth=0.5,
            markersize=4, label="Start", linestyle="None",
        ),
        Line2D(
            [0], [0],
            marker="o", color="#555555",
            markeredgecolor="#555555", markeredgewidth=0.5,
            markersize=4, label="End", linestyle="None",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True, framealpha=0.92, edgecolor="none",
        borderpad=0.4, handletextpad=0.3, fancybox=True, fontsize=5.5,
    ).get_frame().set_linewidth(0)

    fig.suptitle(
        "PCA of RNN Hidden State (Single-Code Clips)",
        fontsize=9, fontweight="bold", y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88, wspace=0.05)
    rect = _add_rounded_border(fig)
    return fig, rect


# ── Combined single-panel plots ──────────────────────────────────────────────


def _plot_scatter_combined(
    emb_3d: np.ndarray,
    beh_slices: dict[str, tuple[int, int]],
    var_explained: np.ndarray,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """All behaviors overlaid in a single 3D scatter panel."""
    fig = plt.figure(figsize=(4.5, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim, zlim = _compute_shared_limits(emb_3d)

    for beh in BEHAVIORS:
        if beh not in beh_slices:
            continue
        s, e = beh_slices[beh]
        pts = emb_3d[s:e]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=BEHAVIOR_COLORS[beh],
            s=6, alpha=0.20,
            edgecolors="none", rasterized=True, depthshade=True,
        )

    _style_3d_ax(ax, var_explained, xlim, ylim, zlim)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=BEHAVIOR_COLORS[beh],
            markersize=5, markeredgewidth=0,
            label=BEHAVIOR_LABELS[beh], linestyle="None",
        )
        for beh in BEHAVIORS if beh in beh_slices
    ]
    leg = ax.legend(
        handles=legend_handles, loc="upper left",
        frameon=True, framealpha=0.92, edgecolor="none",
        borderpad=0.5, handletextpad=0.3, fancybox=True, fontsize=7,
    )
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    ax.set_title(
        "PCA of RNN Hidden State",
        fontsize=8, fontweight="bold", pad=-2,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    rect = _add_rounded_border(fig)
    return fig, rect


def _plot_trajectories_combined(
    emb_3d: np.ndarray,
    beh_slices: dict[str, tuple[int, int]],
    beh_shapes: dict[str, tuple[int, int]],
    var_explained: np.ndarray,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """All behaviors' trajectories overlaid in a single 3D panel."""
    fig = plt.figure(figsize=(4.5, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    xlim, ylim, zlim = _compute_shared_limits(emb_3d)

    for beh in BEHAVIORS:
        if beh not in beh_slices:
            continue
        K_beh, T_beh = beh_shapes[beh]
        s, _ = beh_slices[beh]
        cmap = BEHAVIOR_CMAPS[beh]
        t_norm = np.linspace(0, 1, T_beh)

        for ki in range(K_beh):
            start = s + ki * T_beh
            pts = emb_3d[start : start + T_beh]
            if len(pts) < 2:
                continue
            segments = np.array(
                [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)],
            )
            colors = cmap(t_norm[:-1])
            lc = Line3DCollection(
                segments, colors=colors, linewidths=0.6, alpha=0.55,
            )
            ax.add_collection3d(lc)
            ax.scatter(
                [pts[0, 0]], [pts[0, 1]], [pts[0, 2]],
                c="white", s=15,
                edgecolors=BEHAVIOR_COLORS[beh],
                linewidths=0.5, zorder=5, depthshade=False,
            )
            ax.scatter(
                [pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]],
                c=BEHAVIOR_COLORS[beh], s=15,
                edgecolors=BEHAVIOR_COLORS[beh],
                linewidths=0.5, zorder=5, depthshade=False,
            )

    _style_3d_ax(ax, var_explained, xlim, ylim, zlim)

    legend_handles = [
        Line2D(
            [0], [0], color=BEHAVIOR_COLORS[beh],
            linewidth=1.8, label=BEHAVIOR_LABELS[beh],
        )
        for beh in BEHAVIORS if beh in beh_slices
    ]
    leg = ax.legend(
        handles=legend_handles, loc="upper left",
        frameon=True, framealpha=0.92, edgecolor="none",
        borderpad=0.5, handletextpad=0.3, fancybox=True, fontsize=7,
    )
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_boxstyle("round,pad=0.3,rounding_size=0.2")

    ax.set_title(
        "PCA of RNN Hidden State",
        fontsize=8, fontweight="bold", pad=-2,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    rect = _add_rounded_border(fig)
    return fig, rect


# ── Save helper ──────────────────────────────────────────────────────────────


def _save_figure(
    fig: plt.Figure,
    rect: mpatches.FancyBboxPatch,
    stem: str,
) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    rect.set_facecolor("none")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", transparent=True)
    rect.set_facecolor("white")
    plt.close(fig)
    print(f"  Saved: {stem}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D PCA visualization of GRU hidden state per labeled behavior",
    )
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to hidden_dynamics{,_labeled}.npz "
                             "(default: auto-detect in figures/data/).")
    parser.add_argument("--level", type=str, default=None,
                        choices=[None, "behavior", "direction", "combined"],
                        help="For older nested-key labeled npz only; ignored otherwise.")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated subset of labels to keep.")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to output figure stems.")
    args = parser.parse_args()

    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_suffix = (f"_{args.output_suffix}" if args.output_suffix else "")

    label_subset = (
        [s.strip() for s in args.labels.split(",")] if args.labels else None
    )
    npz_path = Path(args.npz) if args.npz else None

    hiddens, codes, labels, level_used = load_hidden_data(
        npz_path=npz_path, level=args.level, label_subset=label_subset,
    )

    # Re-bind module-level globals so plotting functions see the right colors
    # and pretty-names for the discovered labels.  (The original code relied
    # on these being constants; rather than thread them through every helper
    # we rebind here so the 500-line body of this file stays untouched.)
    global BEHAVIORS, BEHAVIOR_LABELS, BEHAVIOR_COLORS, BEHAVIOR_CMAPS
    BEHAVIORS = list(labels)
    BEHAVIOR_LABELS = _label_display(labels)
    BEHAVIOR_COLORS = _label_colors(labels)
    BEHAVIOR_CMAPS = {b: _behavior_cmap(c) for b, c in BEHAVIOR_COLORS.items()}

    # Build flat array with behavior boundaries (all labels for shared PCA)
    all_points: list[np.ndarray] = []
    beh_slices: dict[str, tuple[int, int]] = {}
    beh_shapes: dict[str, tuple[int, int]] = {}
    offset = 0

    for beh in labels:
        if beh not in hiddens:
            continue
        arr = hiddens[beh]
        K_beh, T_beh = arr.shape[0], arr.shape[1]
        beh_shapes[beh] = (K_beh, T_beh)
        flat = arr.reshape(-1, arr.shape[-1])
        all_points.append(flat)
        beh_slices[beh] = (offset, offset + flat.shape[0])
        offset += flat.shape[0]

    all_points_arr = np.concatenate(all_points, axis=0)
    print(
        f"Loaded {len(labels)} labels (level={level_used or 'flat/legacy'}); "
        f"total points: {all_points_arr.shape[0]}, dim: {all_points_arr.shape[1]}"
    )
    for beh in labels:
        if beh in beh_shapes:
            K_b, T_b = beh_shapes[beh]
            print(f"  {beh:<22s}: K={K_b}, T={T_b}, color={BEHAVIOR_COLORS[beh]}")

    # Fit 3D PCA on ALL labels (shared coordinate frame)
    print("\nFitting 3D PCA...")
    emb, var = fit_pca_3d(all_points_arr)

    # Scatter: separate panels
    fig, rect = plot_scatter_3d(emb, beh_slices, var)
    _save_figure(fig, rect, f"hidden_scatter_pca3d{out_suffix}")

    # Scatter: combined single panel
    fig, rect = _plot_scatter_combined(emb, beh_slices, var)
    _save_figure(fig, rect, f"hidden_scatter_pca3d_combined{out_suffix}")

    # Trajectories: separate panels
    fig, rect = plot_trajectories_3d(emb, beh_slices, beh_shapes, var)
    _save_figure(fig, rect, f"hidden_trajectories_pca3d{out_suffix}")

    # Trajectories: combined single panel
    fig, rect = _plot_trajectories_combined(emb, beh_slices, beh_shapes, var)
    _save_figure(fig, rect, f"hidden_trajectories_pca3d_combined{out_suffix}")

    print("\nDone.")


if __name__ == "__main__":
    main()
