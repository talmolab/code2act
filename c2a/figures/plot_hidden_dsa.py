"""Per-syllable hidden-state dynamics analysis.

Runs three layered analyses on the same split-half rollout data and
produces a unified comparison so the reader can see how each method's
choices affect the conclusion:

1. **Vanilla DSA** (Ostrow et al. 2024 NeurIPS) on PCA-reduced hidden state.
   *Primary analysis.* Compares the Koopman/dynamics operator across
   split-half systems, no input demixing. Tests the question "do
   trajectories of different syllables have different temporal structure?"

2. **Vanilla DSA on per-trial-centered hidden state.** Per-trial mean is
   subtracted from each trial's hidden trajectory, removing the
   syllable-conditional attractor offset and isolating the recurrent
   dynamics signal. Tests "after removing where in state space each
   syllable lives, do the *local rules of motion* still differ?"

3. **InputDSA** (Huang/Ostrow et al. 2025; arXiv:2510.25943) with qpos
   PCA-reduced as the control input ``U``.  *Diagnostic only* in the
   sustained-code rollout regime — code is constant, proprio/qpos is
   highly redundant with hidden state, and the two are coupled in a
   closed loop, so the demixing assumption (exogenous ``u_t``) is
   violated. We include InputDSA results to show explicitly how
   aggressive demixing can overcorrect. The clean InputDSA experiment
   requires natural rollouts where the code varies in time — that's a
   separate script (TODO: ``run_natural_rollouts.py``).

For each method we report split-half *within-syllable* vs
*between-syllable* mean dynamics distance and a 6×6 distance heatmap
across (walk_A, walk_B, immobility_A, immobility_B, rear_A, rear_B).

A rank sweep over {5, 10, 20} validates that the result is not an
artifact of a single hyperparameter choice; the synthetic positive
control included in ``test_inputdsa_sanity.py`` confirms the package
correctly detects known-different A matrices at rank=20.

Outputs
-------
* ``outputs/hidden_dynamics/dsa_vanilla_heatmap.{pdf,png,svg}``
  — PRIMARY: vanilla DSA on PCA-32 hidden, rank=20.
* ``outputs/hidden_dynamics/dsa_centered_heatmap.{pdf,png,svg}``
  — vanilla DSA on per-trial-centered hidden state (recurrent-only).
* ``outputs/hidden_dynamics/dsa_inputdsa_heatmap.{pdf,png,svg}``
  — DIAGNOSTIC: InputDSA with qpos as control.
* ``outputs/hidden_dynamics/dsa_within_between.{pdf,png,svg}``
  — Bar chart: within vs between for all three methods, side by side.
* ``outputs/hidden_dynamics/dsa_rank_sweep.{pdf,png,svg}``
  — within/between ratio as a function of rank, all three methods.
* ``figures/data/dsa_distances.npz`` — cached distance matrices for
  all three methods + rank sweep results.

Usage::

    python -m c2a.figures.plot_hidden_dsa
    python -m c2a.figures.plot_hidden_dsa --replot
    python -m c2a.figures.plot_hidden_dsa --rank 20 --pca_dim 32 --u_pca_dim 16
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ── Paths ────────────────────────────────────────────────────────────────────
from c2a.paths import get_repo_root

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = get_repo_root() / "outputs" / "hidden_dynamics"

# ── Default label aliases (legacy npz from run_hidden_dynamics) ─────────────
# The list also serves as the preferred display order in legacy mode —
# any label not in this list trails after the matched ones in
# discovery order (see ``_discover_labels``).
LEGACY_BEHAVIORS = [
    # 3-class kinematic-heuristic scheme (run_hidden_dynamics.py)
    "walk", "immobility", "rear",
    # 5-class directional-turn scheme (run_labeled_hidden_dynamics.py
    # after the 2026-05-11 directional-turn rewire)
    "Immobile", "Rear", "Walk", "Turn_Left", "Turn_Right",
]
LEGACY_BEHAVIOR_LABELS = {
    # 3-class
    "walk": "Walking",
    "immobility": "Immobility",
    "rear": "Rearing",
    # 5-class
    "Immobile": "Immobility",
    "Rear": "Rearing",
    "Walk": "Walking",
    "Turn_Left": "Turn (Left)",
    "Turn_Right": "Turn (Right)",
}


# ── Label discovery (works with both legacy + labeled-experiment npz) ───────


def _discover_labels(
    npz: dict, level: str | None = None, label_subset: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Find ``hidden_<...>`` keys in the npz and return display labels.

    Two npz formats supported:

    * Legacy (``run_hidden_dynamics.py``): keys are ``hidden_walk``,
      ``hidden_immobility``, ``hidden_rear``. No level prefix. The
      label field directly becomes the label string.

    * Labeled (``run_labeled_hidden_dynamics.py``): keys are
      ``hidden_<level>__<label>`` where level ∈ {behavior, direction,
      combined} and label may itself contain underscores (e.g.
      ``Walk_Straight``). Filter by ``level`` to pick one.

    Args:
        npz: opened ``np.load`` object (or dict-like).
        level: only used for labeled npz — restrict to this level.
            Defaults to ``behavior`` if a labeled npz is detected and
            ``level`` is None.
        label_subset: optional whitelist of label strings to keep.

    Returns:
        ``(labels, display_labels)`` where ``labels`` is the list of
        label strings (used in npz keys) and ``display_labels`` is a
        ``{label: nice_name}`` map for plotting.
    """
    hidden_keys = [k for k in npz.keys() if k.startswith("hidden_")]
    if not hidden_keys:
        raise KeyError("npz has no hidden_<label> keys; nothing to analyze.")

    is_labeled = any("__" in k for k in hidden_keys)

    labels: list[str] = []
    display: dict[str, str] = {}

    if is_labeled:
        eff_level = level or "behavior"
        prefix = f"hidden_{eff_level}__"
        for k in hidden_keys:
            if not k.startswith(prefix):
                continue
            label = k[len(prefix):]
            labels.append(label)
            display[label] = label.replace("_", " ")
    else:
        for k in hidden_keys:
            label = k[len("hidden_"):]
            labels.append(label)
            display[label] = LEGACY_BEHAVIOR_LABELS.get(label, label)

    if label_subset is not None:
        keep = set(label_subset)
        labels = [l for l in labels if l in keep]
        display = {l: display[l] for l in labels}

    if not labels:
        raise ValueError(
            f"After filtering (level={level}, subset={label_subset}) no labels "
            f"matched. Available hidden_* keys: {hidden_keys}"
        )

    # Stable order: preserve discovery order, but keep legacy 3-class order
    # for legacy npz.
    if not is_labeled:
        order = [b for b in LEGACY_BEHAVIORS if b in labels]
        labels = order + [l for l in labels if l not in order]
    else:
        labels.sort()

    return labels, display


def _key(label: str, level: str | None, base: str) -> str:
    """Build the npz key for a label at the chosen base ('hidden', 'qpos', ...)."""
    if level is None:
        return f"{base}_{label}"
    return f"{base}_{level}__{label}"

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_RANK = 20            # validated via synthetic positive control
DEFAULT_PCA_DIM = 32         # X (hidden) dim after joint PCA
DEFAULT_U_PCA_DIM = 16       # U (qpos) dim after joint PCA
DEFAULT_N_DELAYS = 1
DEFAULT_LAMB = 1e-3
DEFAULT_BACKEND = "n4sid"
DEFAULT_SCORE_METHOD = "angular"
RANK_SWEEP = (5, 10, 20)

METHOD_LABELS = {
    "vanilla": "Vanilla DSA",
    "centered": "Vanilla DSA (centered)",
    "inputdsa": "InputDSA (qpos as U)",
}
METHOD_COLORS = {
    "vanilla": "#0072B2",
    "centered": "#009E73",
    "inputdsa": "#888888",
}


# ── Style ────────────────────────────────────────────────────────────────────


def _setup_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


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


# ── Data loading ────────────────────────────────────────────────────────────


def _load_arrays(
    npz_path: Path,
    level: str | None,
    label_subset: list[str] | None,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], list[str], dict[str, str], str | None
]:
    """Load hidden states and a control variable per label.

    Prefers ``proprio_*`` if present (actual GRU input). Falls back to
    ``qpos_*`` (joint positions) — a valid surrogate input.

    Returns:
        ``(X_per_label, U_per_label, label_order, display_map, level_used)``.
        ``level_used`` is the level prefix used in keys (``None`` for legacy
        npz, ``"behavior"|"direction"|"combined"`` for labeled npz).
    """
    raw = np.load(npz_path, allow_pickle=True)
    is_labeled = any("__" in k for k in raw.keys() if k.startswith("hidden_"))
    eff_level = (level or "behavior") if is_labeled else None

    labels, display = _discover_labels(raw, level=eff_level, label_subset=label_subset)

    X_per: dict[str, np.ndarray] = {}
    U_per: dict[str, np.ndarray] = {}
    for lab in labels:
        h_key = _key(lab, eff_level, "hidden")
        X_per[lab] = np.asarray(raw[h_key], dtype=np.float32)

        u_key_proprio = _key(lab, eff_level, "proprio")
        u_key_qpos = _key(lab, eff_level, "qpos")
        if u_key_proprio in raw:
            U_per[lab] = np.asarray(raw[u_key_proprio], dtype=np.float32)
        elif u_key_qpos in raw:
            U_per[lab] = np.asarray(raw[u_key_qpos], dtype=np.float32)
        else:
            raise KeyError(
                f"npz lacks both {u_key_proprio} and {u_key_qpos} for label {lab}"
            )
    return X_per, U_per, labels, display, eff_level


def _pca_jointly(
    per_beh: dict[str, np.ndarray], target_dim: int
) -> dict[str, np.ndarray]:
    """Fit one PCA on pooled data, project all behaviors into the same frame."""
    pooled = np.concatenate(
        [a.reshape(-1, a.shape[-1]) for a in per_beh.values()], axis=0
    )
    eff = min(target_dim, pooled.shape[1], pooled.shape[0] - 1)
    pca = PCA(n_components=eff, whiten=False).fit(pooled)
    out = {}
    for beh, a in per_beh.items():
        flat = a.reshape(-1, a.shape[-1])
        proj = pca.transform(flat).astype(np.float32)
        out[beh] = proj.reshape(a.shape[0], a.shape[1], eff)
    return out


def split_half(
    per_label: dict[str, np.ndarray],
    label_order: list[str],
    seed: int,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Split each label's K trials into two random groups → 2 systems each.

    Returns ``(systems, system_names, label_per_system)``. Labels with
    fewer than 2 trials are skipped (need split-half).
    """
    rng = np.random.default_rng(seed)
    systems, names, behs = [], [], []
    for lab in label_order:
        if lab not in per_label:
            continue
        arr = per_label[lab]
        K = arr.shape[0]
        if K < 2:
            continue
        idx = rng.permutation(K)
        h = K // 2
        for tag, sl in [("A", idx[:h]), ("B", idx[h : 2 * h])]:
            systems.append(arr[sl])
            names.append(f"{lab}_{tag}")
            behs.append(lab)
    return systems, names, behs


# ── Distance computations ────────────────────────────────────────────────────


def _to_np_2d(arr) -> np.ndarray:
    """Coerce a (possibly tensor / 3-D) DSA output to a 2-D numpy distance matrix."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    return arr


def _vanilla_dsa(
    X_systems: list[np.ndarray],
    rank: int,
    n_delays: int,
    score_method: str,
) -> np.ndarray:
    """Vanilla DSA — no input demixing."""
    from DSA import DSA as PlainDSA

    dsa = PlainDSA(
        X=X_systems,
        n_delays=n_delays,
        rank=rank,
        score_method=score_method,
        device="cpu",
        verbose=False,
    )
    return _to_np_2d(dsa.fit_score())


def _input_dsa(
    X_systems: list[np.ndarray],
    U_systems: list[np.ndarray],
    rank: int,
    n_delays: int,
    lamb: float,
    backend: str,
    score_method: str,
) -> np.ndarray:
    """InputDSA — Subspace DMDc + ControllabilitySimilarityTransformDist."""
    from DSA import InputDSA
    from DSA.dsa import (
        SubspaceDMDcConfig,
        ControllabilitySimilarityTransformDistConfig,
    )

    dmd_cfg = SubspaceDMDcConfig(
        n_delays=n_delays, rank=rank, lamb=lamb, backend=backend,
    )
    sim_cfg = ControllabilitySimilarityTransformDistConfig(
        score_method=score_method,
        compare="joint",
        return_distance_components=True,
    )
    idsa = InputDSA(
        X=X_systems, X_control=U_systems,
        dmd_config=dmd_cfg, simdist_config=sim_cfg,
        device="cpu", verbose=False,
    )
    out = idsa.fit_score()
    if isinstance(out, tuple) and len(out) == 3:
        _joint, state, _ctrl = out
    else:
        state = out
    return _to_np_2d(state)


# ── Within / between ─────────────────────────────────────────────────────────


def _wb(D: np.ndarray, behs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = len(behs)
    w, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (w if behs[i] == behs[j] else b).append(D[i, j])
    return np.asarray(w), np.asarray(b)


# ── Plotting ─────────────────────────────────────────────────────────────────


def _plot_heatmap(
    D: np.ndarray, names: list[str], display: dict[str, str],
    title: str, cbar_label: str,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """6×6 (or NxN) distance heatmap with block boundaries between labels.

    ``names[i]`` is ``"<label>_A"`` / ``"<label>_B"``. ``display`` maps
    label → display string.
    """
    n = len(names)
    n_labels = n // 2
    fig_size = (max(3.2, 0.55 * n + 0.6), max(2.8, 0.55 * n + 0.4))
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(D, cmap="viridis", aspect="equal", interpolation="none")
    # White block separators every 2 systems (one block per label)
    for b in range(2, n, 2):
        ax.axhline(y=b - 0.5, color="white", linewidth=1.2)
        ax.axvline(x=b - 0.5, color="white", linewidth=1.2)
    tick_labels = [
        # Label is everything before the trailing _A / _B; tags are A / B.
        f"{display.get(name.rsplit('_', 1)[0], name.rsplit('_', 1)[0])} "
        f"{name.rsplit('_', 1)[1]}"
        for name in names
    ]
    fs = max(3.5, 6 - 0.05 * n)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=fs, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=fs)

    finite = D[np.isfinite(D)]
    vmax = float(finite.max()) if finite.size else 1.0
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(D[i, j]):
                ax.text(
                    j, i, f"{D[i, j]:.3f}",
                    ha="center", va="center", fontsize=4.5,
                    color="white" if D[i, j] < vmax * 0.6 else "black",
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(cbar_label, fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=6)

    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


def _plot_within_between(
    distances: dict[str, np.ndarray], behs: list[str],
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """Side-by-side bar chart of within vs between for all methods."""
    methods = list(distances.keys())
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    width = 0.35
    x = np.arange(n_methods)
    rng = np.random.default_rng(42)

    within_means, between_means = [], []
    for k, method in enumerate(methods):
        D = distances[method]
        w, b = _wb(D, behs)
        wm = float(w.mean()) if w.size else np.nan
        bm = float(b.mean()) if b.size else np.nan
        ws = float(w.std(ddof=1) / np.sqrt(len(w))) if len(w) > 1 else 0
        bs = float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else 0

        ax.bar(x[k] - width / 2, wm, width, yerr=ws,
               color="#3498db", alpha=0.85, capsize=3, edgecolor="none",
               error_kw={"linewidth": 0.7},
               label="Within syllable" if k == 0 else None)
        ax.bar(x[k] + width / 2, bm, width, yerr=bs,
               color="#e74c3c", alpha=0.85, capsize=3, edgecolor="none",
               error_kw={"linewidth": 0.7},
               label="Between syllable" if k == 0 else None)
        # strip plot
        ax.scatter(
            x[k] - width / 2 + rng.uniform(-0.05, 0.05, size=len(w)),
            w, c="#3498db", s=12, alpha=0.6, edgecolors="none", zorder=3,
        )
        ax.scatter(
            x[k] + width / 2 + rng.uniform(-0.05, 0.05, size=len(b)),
            b, c="#e74c3c", s=12, alpha=0.6, edgecolors="none", zorder=3,
        )
        # b/w ratio annotation
        if not (np.isnan(wm) or np.isnan(bm) or wm == 0):
            ratio = bm / wm
            ax.text(
                x[k], max(wm, bm) * 1.12,
                f"b/w = {ratio:.2f}",
                ha="center", fontsize=6.5,
                color="#222",
                fontweight="bold" if ratio > 1.2 else "normal",
            )
        within_means.append(wm)
        between_means.append(bm)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=6.5)
    ax.set_ylabel("DSA distance (angular)", fontsize=7)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_title(
        "Within- vs Between-syllable dynamics distance",
        fontsize=8, fontweight="bold", pad=6,
    )
    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


def _plot_rank_sweep(
    sweep: dict[str, dict[int, tuple[float, float]]],
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """within/between ratio vs rank, for each method."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for method, by_rank in sweep.items():
        ranks = sorted(by_rank.keys())
        ratios = [by_rank[r][1] / max(by_rank[r][0], 1e-9) for r in ranks]
        ax.plot(
            ranks, ratios, "o-", color=METHOD_COLORS[method],
            linewidth=1.0, markersize=4,
            label=METHOD_LABELS[method],
        )
    ax.axhline(1.0, linestyle="--", color="#888", linewidth=0.5,
               label="b/w = 1 (no signal)")
    ax.set_xlabel("Fitted rank", fontsize=7)
    ax.set_ylabel("Between / within ratio", fontsize=7)
    ax.set_title("Per-syllable dynamics signal vs rank", fontsize=8,
                 fontweight="bold", pad=6)
    ax.legend(fontsize=5.5, frameon=False)
    ax.set_xticks(sorted(set(r for s in sweep.values() for r in s)))
    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


# ── Save helper ──────────────────────────────────────────────────────────────


def _save_figure(
    fig: plt.Figure, rect: mpatches.FancyBboxPatch, stem: str,
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
        description="Layered DSA / InputDSA analysis on RNN hidden states",
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK,
                        help="Primary fitted rank (also used for heatmaps)")
    parser.add_argument("--pca_dim", type=int, default=DEFAULT_PCA_DIM,
                        help="PCA dim for hidden state X")
    parser.add_argument("--u_pca_dim", type=int, default=DEFAULT_U_PCA_DIM,
                        help="PCA dim for control input U (qpos)")
    parser.add_argument("--n_delays", type=int, default=DEFAULT_N_DELAYS)
    parser.add_argument("--lamb", type=float, default=DEFAULT_LAMB)
    parser.add_argument("--backend", type=str, default=DEFAULT_BACKEND)
    parser.add_argument("--score_method", type=str,
                        default=DEFAULT_SCORE_METHOD,
                        choices=["angular", "euclidean"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to hidden_dynamics{,_labeled}.npz. "
                             "Default: figures/data/hidden_dynamics.npz; "
                             "auto-falls-back to hidden_dynamics_labeled.npz "
                             "if the legacy file is absent.")
    parser.add_argument("--level", type=str, default=None,
                        choices=[None, "behavior", "direction", "combined"],
                        help="For labeled-experiment npz: which label level "
                             "to analyze. Default 'behavior' if labeled npz, "
                             "ignored for legacy npz.")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated subset of labels to keep "
                             "(e.g. 'Walk,Rear'). Default: all available.")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to output figure stems "
                             "(useful when running multiple levels).")
    args = parser.parse_args()

    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_suffix = (f"_{args.output_suffix}" if args.output_suffix else "")
    cache_path = DATA_DIR / f"dsa_distances{out_suffix}.npz"

    # Resolve npz path with auto-fallback.
    if args.npz is not None:
        npz_path = Path(args.npz)
    else:
        npz_path = DATA_DIR / "hidden_dynamics.npz"
        if not npz_path.exists():
            alt = DATA_DIR / "hidden_dynamics_labeled.npz"
            if alt.exists():
                npz_path = alt
                print(f"NOTE: legacy npz absent; using labeled npz {alt.name}")
    if not npz_path.exists():
        raise FileNotFoundError(f"No hidden-dynamics npz at {npz_path}")
    print(f"Reading: {npz_path}")

    label_subset = (
        [s.strip() for s in args.labels.split(",")]
        if args.labels else None
    )

    if args.replot and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        D_vanilla = cached["vanilla"]
        D_centered = cached["centered"]
        D_inputdsa = cached["inputdsa"]
        names = list(cached["names"])
        behs = list(cached["behaviors"])
        display = {b: b for b in set(behs)}
        sweep = {
            m: {int(r): tuple(cached[f"sweep_{m}_r{int(r)}"])
                for r in cached["sweep_ranks"]}
            for m in ("vanilla", "centered", "inputdsa")
        }
    else:
        # ── Load + PCA ─────────────────────────────────────────────────────
        X_per_beh, U_per_beh, label_order, display, level_used = _load_arrays(
            npz_path=npz_path, level=args.level, label_subset=label_subset,
        )
        print(f"Level: {level_used or 'legacy (no level)'}")
        for lab in label_order:
            print(f"  {lab}: X={X_per_beh[lab].shape}, U={U_per_beh[lab].shape}")

        X_pca = _pca_jointly(X_per_beh, args.pca_dim)
        U_pca = _pca_jointly(U_per_beh, args.u_pca_dim)

        eff_x = next(iter(X_pca.values())).shape[-1]
        eff_u = next(iter(U_pca.values())).shape[-1]
        print(f"  PCA: X→{eff_x} dims, U→{eff_u} dims")

        # ── Split-half ─────────────────────────────────────────────────────
        X_systems, names, behs = split_half(X_pca, label_order, args.seed)
        U_systems, _, _ = split_half(U_pca, label_order, args.seed)
        # Per-trial centered version for the "recurrent-only" analysis.
        X_centered = [x - x.mean(axis=1, keepdims=True) for x in X_systems]

        print(f"\n  Systems: {len(X_systems)} ({eff_x}-dim, {X_systems[0].shape[1]}-step trajectories)")

        # ── Method 1: vanilla DSA ──────────────────────────────────────────
        print(f"\n=== Method 1: Vanilla DSA (rank={args.rank}, n_delays={args.n_delays}) ===")
        D_vanilla = _vanilla_dsa(
            X_systems,
            rank=args.rank, n_delays=args.n_delays,
            score_method=args.score_method,
        )
        w, b = _wb(D_vanilla, behs)
        print(f"  within={w.mean():.4f} ± {w.std(ddof=1):.4f}, "
              f"between={b.mean():.4f} ± {b.std(ddof=1):.4f}, "
              f"b/w = {b.mean()/max(w.mean(),1e-9):.2f}")

        # ── Method 2: vanilla DSA on per-trial centered ────────────────────
        print(f"\n=== Method 2: Vanilla DSA on per-trial CENTERED hidden ===")
        D_centered = _vanilla_dsa(
            X_centered,
            rank=args.rank, n_delays=args.n_delays,
            score_method=args.score_method,
        )
        w, b = _wb(D_centered, behs)
        print(f"  within={w.mean():.4f} ± {w.std(ddof=1):.4f}, "
              f"between={b.mean():.4f} ± {b.std(ddof=1):.4f}, "
              f"b/w = {b.mean()/max(w.mean(),1e-9):.2f}")

        # ── Method 3: InputDSA (diagnostic) ────────────────────────────────
        print(f"\n=== Method 3: InputDSA (qpos as U) — DIAGNOSTIC ===")
        D_inputdsa = _input_dsa(
            X_systems, U_systems,
            rank=args.rank, n_delays=args.n_delays,
            lamb=args.lamb, backend=args.backend,
            score_method=args.score_method,
        )
        w, b = _wb(D_inputdsa, behs)
        print(f"  within={w.mean():.4f} ± {w.std(ddof=1):.4f}, "
              f"between={b.mean():.4f} ± {b.std(ddof=1):.4f}, "
              f"b/w = {b.mean()/max(w.mean(),1e-9):.2f}")
        print(
            "  (Caveat: in sustained-code rollouts the code is constant and qpos\n"
            "   is highly redundant with hidden state; InputDSA's exogenous-input\n"
            "   assumption is violated. This row is expected to wash out signal.)"
        )

        # ── Rank sweep ─────────────────────────────────────────────────────
        print("\n=== Rank sweep ===")
        sweep: dict[str, dict[int, tuple[float, float]]] = {
            m: {} for m in ("vanilla", "centered", "inputdsa")
        }
        for r in RANK_SWEEP:
            print(f"  rank={r}:")
            Dv = _vanilla_dsa(
                X_systems, rank=r, n_delays=args.n_delays,
                score_method=args.score_method,
            )
            Dc = _vanilla_dsa(
                X_centered, rank=r, n_delays=args.n_delays,
                score_method=args.score_method,
            )
            Di = _input_dsa(
                X_systems, U_systems, rank=r, n_delays=args.n_delays,
                lamb=args.lamb, backend=args.backend,
                score_method=args.score_method,
            )
            for method, D in [("vanilla", Dv), ("centered", Dc), ("inputdsa", Di)]:
                w, b = _wb(D, behs)
                wm = float(w.mean()); bm = float(b.mean())
                sweep[method][r] = (wm, bm)
                print(f"    {method:>10s}: within={wm:.4f}, "
                      f"between={bm:.4f}, b/w={bm/max(wm,1e-9):.2f}")

        # ── Cache ──────────────────────────────────────────────────────────
        cache_dict = dict(
            vanilla=D_vanilla,
            centered=D_centered,
            inputdsa=D_inputdsa,
            names=np.array(names),
            behaviors=np.array(behs),
            sweep_ranks=np.array(list(RANK_SWEEP)),
            rank=args.rank, pca_dim=args.pca_dim, u_pca_dim=args.u_pca_dim,
            n_delays=args.n_delays, lamb=args.lamb, backend=args.backend,
            score_method=args.score_method,
        )
        for m in ("vanilla", "centered", "inputdsa"):
            for r, (wm, bm) in sweep[m].items():
                cache_dict[f"sweep_{m}_r{r}"] = np.array([wm, bm])
        np.savez_compressed(cache_path, **cache_dict)
        print(f"\nCached: {cache_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    fig, rect = _plot_heatmap(
        D_vanilla, names, display,
        title=f"Vanilla DSA (rank={args.rank}) — primary",
        cbar_label="DSA distance",
    )
    _save_figure(fig, rect, f"dsa_vanilla_heatmap{out_suffix}")

    fig, rect = _plot_heatmap(
        D_centered, names, display,
        title=f"Vanilla DSA on per-trial centered (rank={args.rank})",
        cbar_label="DSA distance",
    )
    _save_figure(fig, rect, f"dsa_centered_heatmap{out_suffix}")

    fig, rect = _plot_heatmap(
        D_inputdsa, names, display,
        title=f"InputDSA — qpos as U, rank={args.rank} (diagnostic)",
        cbar_label="State distance",
    )
    _save_figure(fig, rect, f"dsa_inputdsa_heatmap{out_suffix}")

    fig, rect = _plot_within_between(
        {"vanilla": D_vanilla, "centered": D_centered, "inputdsa": D_inputdsa},
        behs,
    )
    _save_figure(fig, rect, f"dsa_within_between{out_suffix}")

    fig, rect = _plot_rank_sweep(sweep)
    _save_figure(fig, rect, f"dsa_rank_sweep{out_suffix}")

    print("\nDone.")


if __name__ == "__main__":
    main()
