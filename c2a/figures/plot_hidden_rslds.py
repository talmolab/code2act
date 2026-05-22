"""rSLDS analysis of RNN hidden states (Linderman lab ``ssm`` package).

The hypothesis: if Code2Act's GRU has internalized syllable-conditioned
dynamics, an unsupervised dynamical-regimes model fit to the GRU's hidden
trajectories should rediscover regimes that align with the externally-
provided syllable labels (walk / immobility / rear).

We use a **recurrent SLDS** (Linderman et al. 2017) where:

    z_t  ∈ {1, …, K}           # discrete regime
    x_t  ∈ R^D                 # continuous latent
    y_t  ∈ R^N                 # observed (PCA-reduced GRU hidden state)
    p(z_{t+1} | z_t, x_t)      # recurrent transition (state-dependent)
    x_{t+1} = A_{z_{t+1}} x_t + b_{z_{t+1}} + ε
    y_t = C x_t + d + η

Setup
-----
* ``hidden_dynamics.npz`` provides 30 trials (10 per behavior × 3 behaviors)
  of 250-frame, 256-dim GRU hidden trajectories.
* PCA-reduce hidden states to ``N=32`` (fit on pooled trials → shared frame).
* Fit one rSLDS with ``K`` discrete states (default 3 — one per behavior).
* Decode most-likely discrete-state sequence per trial.
* Compare to true (constant-per-trial) behavior labels via NMI + Hungarian
  purity (same metrics used to validate KPMS in ``c2a/kpms/coverage_metrics.py``).

If the GRU implements distinct per-syllable dynamics, NMI should be high
(0.5+ for K=3) and the discovered regimes should mostly stay within trial.

Outputs
-------
* ``figures/data/rslds_results.npz`` — fit results for re-plotting.
* ``outputs/hidden_dynamics/rslds_confusion.{pdf,png,svg}``
  — Behavior × discovered-regime confusion matrix (frame-fraction).
* ``outputs/hidden_dynamics/rslds_regime_traces.{pdf,png,svg}``
  — Per-trial regime trace, K timelines per behavior.
* ``outputs/hidden_dynamics/rslds_summary.{pdf,png,svg}``
  — NMI + per-trial purity bar chart.

Requires::

    pip install ssm  # Linderman lab; rSLDS via transitions='recurrent'

Usage::

    python -m c2a.figures.plot_hidden_rslds
    python -m c2a.figures.plot_hidden_rslds --K 3 --D 8 --pca_dim 32 --num_iters 50
    python -m c2a.figures.plot_hidden_rslds --replot
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

# ── Paths ────────────────────────────────────────────────────────────────────
from c2a.paths import get_repo_root

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = get_repo_root() / "outputs" / "hidden_dynamics"

# ── Default label aliases (legacy npz) ──────────────────────────────────────
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


def _discover_labels(
    npz: dict,
    level: str | None = None,
    label_subset: list[str] | None = None,
) -> tuple[list[str], dict[str, str], str | None]:
    """Discover ``hidden_<...>`` keys in either legacy or labeled npz.

    Returns ``(labels, display_map, level_used)``.  See
    ``plot_hidden_dsa._discover_labels`` for format details.
    """
    hidden_keys = [k for k in npz.keys() if k.startswith("hidden_")]
    if not hidden_keys:
        raise KeyError("npz has no hidden_<label> keys; nothing to analyze.")

    is_labeled = any("__" in k for k in hidden_keys)
    eff_level = (level or "behavior") if is_labeled else None

    labels: list[str] = []
    display: dict[str, str] = {}
    if is_labeled:
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

    if not is_labeled:
        order = [b for b in LEGACY_BEHAVIORS if b in labels]
        labels = order + [l for l in labels if l not in order]
    else:
        labels.sort()
    return labels, display, eff_level

# Default rSLDS hyperparameters
# K=5 matches the directional-turn label scheme (Immobile / Rear / Walk
# / Turn_Left / Turn_Right) produced by run_labeled_hidden_dynamics.py.
# Drop to --K 3 if running against the legacy 3-class npz from
# run_hidden_dynamics.py; bump if you want oversegmentation slack.
DEFAULT_K = 5                    # one discrete regime per target behavior
DEFAULT_D = 8                    # continuous latent dim
DEFAULT_PCA_DIM = 32             # observed dim after PCA on hidden states
DEFAULT_NUM_ITERS = 50           # Laplace EM iterations
DEFAULT_FIT_METHOD = "laplace_em"


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


# ── Data loading + preprocessing ────────────────────────────────────────────


def load_pooled_hidden(
    pca_dim: int,
    npz_path: Path | None = None,
    level: str | None = None,
    label_subset: list[str] | None = None,
) -> tuple[
    list[np.ndarray], np.ndarray, list[str], list[str], dict[str, str], PCA
]:
    """Pool trials across all labels, fit shared PCA, return per-trial data.

    Args:
        pca_dim: target dim for shared PCA on hidden states.
        npz_path: defaults to ``figures/data/hidden_dynamics.npz`` with
            auto-fallback to ``hidden_dynamics_labeled.npz``.
        level: for labeled npz, which level to use (default ``behavior``).
        label_subset: optional whitelist of labels to keep.

    Returns:
        ``(datas, per_frame_labels, label_strings, names, display, pca)``::

            datas: list of N ``(T, pca_dim)`` arrays.
            per_frame_labels: ``(N, T)`` int — true label index per frame
                (constant per trial in our setup; the array shape is
                preserved for future natural-rollout npzs).
            label_strings: ordered list of label names corresponding to
                indices in ``per_frame_labels``.
            names: per-trial names ``"<label>_<trial>"``.
            display: ``{label: nice_name}``.
            pca: fitted ``PCA`` object.
    """
    if npz_path is None:
        npz_path = DATA_DIR / "hidden_dynamics.npz"
        if not npz_path.exists():
            alt = DATA_DIR / "hidden_dynamics_labeled.npz"
            if alt.exists():
                npz_path = alt
                print(f"NOTE: legacy npz absent; using labeled npz {alt.name}")
    raw = np.load(npz_path, allow_pickle=True)

    label_strings, display, level_used = _discover_labels(
        raw, level=level, label_subset=label_subset,
    )
    label_to_idx = {l: i for i, l in enumerate(label_strings)}

    trials_per_label: dict[str, np.ndarray] = {}
    for lab in label_strings:
        h_key = (
            f"hidden_{level_used}__{lab}" if level_used is not None
            else f"hidden_{lab}"
        )
        trials_per_label[lab] = np.asarray(raw[h_key], dtype=np.float32)

    pooled = np.concatenate(
        [t.reshape(-1, t.shape[-1]) for t in trials_per_label.values()],
        axis=0,
    )
    n_dim = pooled.shape[1]
    eff_dim = min(pca_dim, n_dim, pooled.shape[0] - 1)
    pca = PCA(n_components=eff_dim, whiten=False)
    pca.fit(pooled)

    datas: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    for lab, arr in trials_per_label.items():
        K, T, _ = arr.shape
        for k in range(K):
            proj = pca.transform(arr[k]).astype(np.float64)
            datas.append(proj)
            labels.append(np.full((T,), label_to_idx[lab], dtype=np.int64))
            names.append(f"{lab}_{k:02d}")

    per_frame_labels = np.stack(labels)
    return datas, per_frame_labels, label_strings, names, display, pca


# ── rSLDS fitting ────────────────────────────────────────────────────────────


def fit_rslds(
    datas: list[np.ndarray],
    K: int,
    D: int,
    num_iters: int,
    method: str = DEFAULT_FIT_METHOD,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[object, np.ndarray, list[np.ndarray]]:
    """Fit rSLDS via Laplace EM, return (model, elbos, per-trial discrete states).

    The ``ssm`` package's recurrent transitions implement
    p(z_{t+1} | z_t, x_t) ∝ exp(W_z x_t + b_z) — i.e., the discrete regime
    distribution depends on the continuous latent, which is what makes it
    *recurrent* SLDS rather than vanilla SLDS.
    """
    import autograd.numpy.random as anp_random
    import numpy.random as np_random
    from ssm import SLDS

    np_random.seed(seed)
    anp_random.seed(seed)

    N = datas[0].shape[1]  # observation dim
    if verbose:
        print(
            f"  rSLDS: N={N} (PCA-hidden), K={K} regimes, D={D} (continuous latent)"
        )
        print(f"  trials: {len(datas)}, T per trial: {datas[0].shape[0]}")

    # ``gaussian`` (unconstrained linear emissions) avoids the PCA-init
    # vs orthogonality assertion failure in ssm.emissions.gaussian_orthog.
    model = SLDS(
        N=N, K=K, D=D,
        transitions="recurrent",
        dynamics="gaussian",
        emissions="gaussian",
        single_subspace=True,
    )
    model.initialize(datas)

    if verbose:
        print(f"\n  Fitting rSLDS ({method}, {num_iters} iters)...")
    elbos, q_lem = model.fit(
        datas,
        method=method,
        num_iters=num_iters,
        initialize=False,
        verbose=2 if verbose else 0,
    )
    elbos = np.asarray(elbos, dtype=np.float64)

    if verbose:
        print(f"  Final ELBO: {elbos[-1]:.2f} (start {elbos[0]:.2f})")

    # Decode most-likely discrete state per trial using the *posterior mean* of
    # the continuous latent — passing zeros here decouples z from x and the
    # recurrent transition collapses to the most-marginally-likely state. The
    # variational posterior ``q_lem`` from ``fit`` exposes the per-trial
    # continuous-state means.
    q_x_means = q_lem.mean_continuous_states
    z_per_trial: list[np.ndarray] = []
    for d, qx in zip(datas, q_x_means):
        qx = np.asarray(qx)
        z = model.most_likely_states(qx, d)
        z_per_trial.append(np.asarray(z, dtype=np.int64))

    return model, elbos, z_per_trial


# ── Alignment metrics ────────────────────────────────────────────────────────


def compute_alignment(
    z_per_trial: list[np.ndarray],
    per_frame_labels: np.ndarray,
    K: int,
    n_behaviors: int,
) -> dict:
    """NMI + Hungarian-matched purity + per-trial purity + confusion matrix."""
    z_flat = np.concatenate(z_per_trial)
    labels_flat = per_frame_labels.reshape(-1)

    nmi = float(
        normalized_mutual_info_score(
            labels_flat, z_flat, average_method="arithmetic",
        )
    )

    # Confusion matrix [n_behaviors, K]
    confusion = np.zeros((n_behaviors, K), dtype=np.int64)
    np.add.at(confusion, (labels_flat, z_flat), 1)

    # Hungarian-matched purity
    row_ind, col_ind = linear_sum_assignment(-confusion)
    matched = int(confusion[row_ind, col_ind].sum())
    total = int(confusion.sum())
    purity = matched / total if total > 0 else float("nan")
    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    # Per-trial purity (fraction of frames in dominant discovered regime)
    per_trial_purity = []
    for z in z_per_trial:
        counts = np.bincount(z, minlength=K)
        per_trial_purity.append(float(counts.max() / counts.sum()))

    return {
        "nmi": nmi,
        "purity_global": purity,
        "confusion": confusion,
        "row_to_col_mapping": mapping,
        "per_trial_purity": np.asarray(per_trial_purity),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_confusion(
    confusion: np.ndarray,
    behaviors: list[str],
    display: dict[str, str],
    K: int,
    nmi: float,
    purity: float,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """Behavior × discovered-regime frame-fraction confusion."""
    n_b = len(behaviors)
    fig_w = max(3.0, 0.5 * K + 1.6)
    fig_h = max(2.4, 0.45 * n_b + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    row_sums = confusion.sum(axis=1, keepdims=True)
    norm = confusion / np.maximum(row_sums, 1)
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    for i in range(n_b):
        for j in range(K):
            ax.text(
                j, i, f"{norm[i, j]:.2f}",
                ha="center", va="center", fontsize=5.5,
                color="white" if norm[i, j] < 0.6 else "black",
            )

    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$z_{{{j}}}$" for j in range(K)], fontsize=7)
    ax.set_yticks(range(n_b))
    ax.set_yticklabels(
        [display.get(b, b) for b in behaviors],
        fontsize=max(5.5, 7 - 0.1 * n_b),
    )
    ax.set_xlabel("Discovered regime (rSLDS)", fontsize=7)
    ax.set_ylabel("True behavior", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Frame fraction", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.set_title(
        f"rSLDS regime ↔ syllable (NMI={nmi:.3f}, purity={purity:.3f})",
        fontsize=7.5, fontweight="bold", pad=6,
    )
    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


def plot_regime_traces(
    z_per_trial: list[np.ndarray],
    per_frame_labels: np.ndarray,
    behaviors: list[str],
    display: dict[str, str],
    K: int,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """Per-trial regime trace, grouped by true behavior."""
    behavior_idx = {beh: i for i, beh in enumerate(behaviors)}
    # Group trials by true behavior (constant per trial in our setup)
    by_beh: dict[str, list[int]] = {b: [] for b in behaviors}
    for ti, lbl_row in enumerate(per_frame_labels):
        true_b = behaviors[int(lbl_row[0])]  # constant per trial
        by_beh[true_b].append(ti)

    fig_h = max(2.0, 0.7 * len(behaviors))
    fig, axes = plt.subplots(
        len(behaviors), 1, figsize=(4.5, fig_h),
        sharex=True, sharey=True,
    )
    if len(behaviors) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    regime_colors = [cmap(j % 10) for j in range(K)]

    for ax, beh in zip(axes, behaviors):
        trial_ids = by_beh[beh]
        if not trial_ids:
            continue
        for row, ti in enumerate(trial_ids):
            z = z_per_trial[ti]
            for j in range(K):
                mask = z == j
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                ax.vlines(
                    idxs, row - 0.4, row + 0.4,
                    color=regime_colors[j], linewidth=0.8, alpha=0.85,
                )
        ax.set_ylim(-0.6, len(trial_ids) - 0.4)
        ax.set_yticks(range(len(trial_ids)))
        ax.set_yticklabels([f"#{t:02d}" for t in trial_ids], fontsize=5)
        ax.set_ylabel(display.get(beh, beh), fontsize=6.5)
        ax.tick_params(axis="x", labelsize=5)

    axes[-1].set_xlabel("Frame", fontsize=7)

    # Legend for regime colors
    handles = [
        plt.Line2D(
            [0], [0], color=regime_colors[j], linewidth=4,
            label=f"$z_{{{j}}}$",
        )
        for j in range(K)
    ]
    fig.legend(
        handles=handles, loc="upper right",
        bbox_to_anchor=(0.99, 0.98), fontsize=6,
        ncol=K, frameon=False,
    )
    fig.suptitle(
        "rSLDS discovered regime per trial, grouped by syllable",
        fontsize=8, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    rect = _add_rounded_border(fig)
    return fig, rect


def plot_summary(
    nmi: float,
    purity: float,
    per_trial_purity: np.ndarray,
    elbos: np.ndarray,
) -> tuple[plt.Figure, mpatches.FancyBboxPatch]:
    """3-panel: ELBO trace + per-trial purity histogram + NMI/purity badge."""
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))

    ax = axes[0]
    ax.plot(elbos, color="#222", linewidth=1.0)
    ax.set_xlabel("EM iteration", fontsize=7)
    ax.set_ylabel("ELBO", fontsize=7)
    ax.set_title("rSLDS Laplace-EM convergence", fontsize=7.5)

    ax = axes[1]
    ax.hist(
        per_trial_purity, bins=np.linspace(0, 1, 21),
        color="#3498db", edgecolor="white", linewidth=0.6,
    )
    ax.axvline(per_trial_purity.mean(), color="#e74c3c", linewidth=1.0,
               label=f"mean = {per_trial_purity.mean():.2f}")
    ax.set_xlabel("Per-trial dominant regime fraction", fontsize=7)
    ax.set_ylabel("Trials", fontsize=7)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(
        f"NMI = {nmi:.3f}   |   Hungarian purity = {purity:.3f}",
        fontsize=7,
    )

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
        description="rSLDS analysis of RNN hidden states (Linderman ssm).",
    )
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help="Number of discrete regimes")
    parser.add_argument("--D", type=int, default=DEFAULT_D,
                        help="Continuous latent dim")
    parser.add_argument("--pca_dim", type=int, default=DEFAULT_PCA_DIM,
                        help="PCA dim for observed (hidden state)")
    parser.add_argument("--num_iters", type=int, default=DEFAULT_NUM_ITERS,
                        help="Laplace-EM iterations")
    parser.add_argument("--method", type=str, default=DEFAULT_FIT_METHOD,
                        choices=["laplace_em", "bbvi"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replot", action="store_true",
                        help="Replot from cached results")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to hidden_dynamics{,_labeled}.npz "
                             "(default: auto-detect in figures/data/).")
    parser.add_argument("--level", type=str, default=None,
                        choices=[None, "behavior", "direction", "combined"],
                        help="For labeled npz, which level to use (default: "
                             "behavior). Ignored for legacy npz.")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated subset of labels to keep.")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to output figure stems.")
    args = parser.parse_args()

    _setup_nature_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_suffix = (f"_{args.output_suffix}" if args.output_suffix else "")
    cache_path = DATA_DIR / f"rslds_results{out_suffix}.npz"

    label_subset = (
        [s.strip() for s in args.labels.split(",")] if args.labels else None
    )

    if args.replot and cache_path.exists():
        print(f"Loading cached: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        z_per_trial = [np.asarray(z, dtype=np.int64) for z in cached["z_per_trial"]]
        per_frame_labels = np.asarray(cached["per_frame_labels"], dtype=np.int64)
        elbos = cached["elbos"]
        K = int(cached["K"])
        # Older caches lack label_strings — fall back to legacy 3-class.
        if "label_strings" in cached.files:
            label_strings = [str(s) for s in cached["label_strings"]]
        else:
            label_strings = list(LEGACY_BEHAVIORS)
        display = {l: LEGACY_BEHAVIOR_LABELS.get(l, l) for l in label_strings}
    else:
        # ── Load + PCA ─────────────────────────────────────────────────
        npz_path = Path(args.npz) if args.npz else None
        datas, per_frame_labels, label_strings, names, display, pca = (
            load_pooled_hidden(
                args.pca_dim, npz_path=npz_path,
                level=args.level, label_subset=label_subset,
            )
        )
        print(
            f"\nPooled {len(datas)} trials across {len(label_strings)} labels, "
            f"T={datas[0].shape[0]}, PCA→{datas[0].shape[1]} "
            f"(var explained {pca.explained_variance_ratio_.sum():.2%})"
        )
        print(f"Labels: {label_strings}")

        # ── Fit rSLDS ──────────────────────────────────────────────────
        _model, elbos, z_per_trial = fit_rslds(
            datas, K=args.K, D=args.D, num_iters=args.num_iters,
            method=args.method, seed=args.seed, verbose=True,
        )

        # ── Cache ──────────────────────────────────────────────────────
        np.savez_compressed(
            cache_path,
            z_per_trial=np.array(z_per_trial, dtype=object),
            per_frame_labels=per_frame_labels,
            elbos=elbos,
            label_strings=np.array(label_strings),
            K=args.K, D=args.D, pca_dim=args.pca_dim,
            num_iters=args.num_iters, seed=args.seed,
        )
        print(f"\nCached results: {cache_path}")
        K = args.K

    # ── Alignment ──────────────────────────────────────────────────────
    align = compute_alignment(
        z_per_trial=z_per_trial,
        per_frame_labels=per_frame_labels,
        K=K,
        n_behaviors=len(label_strings),
    )
    print("\nAlignment with true syllable labels:")
    print(f"  NMI            = {align['nmi']:.4f}")
    print(f"  Hungarian      = {align['purity_global']:.4f}")
    print(f"  per-trial mean = {align['per_trial_purity'].mean():.4f} "
          f"(std {align['per_trial_purity'].std(ddof=1):.4f})")
    print(f"  mapping        = {align['row_to_col_mapping']}")

    # ── Plot ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig, rect = plot_confusion(
        align["confusion"], label_strings, display, K,
        align["nmi"], align["purity_global"],
    )
    _save_figure(fig, rect, f"rslds_confusion{out_suffix}")

    fig, rect = plot_regime_traces(
        z_per_trial, per_frame_labels, label_strings, display, K,
    )
    _save_figure(fig, rect, f"rslds_regime_traces{out_suffix}")

    fig, rect = plot_summary(
        align["nmi"], align["purity_global"],
        align["per_trial_purity"], elbos,
    )
    _save_figure(fig, rect, f"rslds_summary{out_suffix}")

    print("\nDone.")


if __name__ == "__main__":
    main()
