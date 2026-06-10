"""Experiment: DSA + InputDSA + jPCA + rSLDS on code-grid hidden trajectories.

Reads the rollouts and clusters produced by ``run_code_grid_rollouts.py`` and,
per requested **pose** × **strategy** × **state representation**, runs the
embodied dynamics starter-kit from
``scratch/2026-06-08-motif-adaptation-interp-toolbox/``:

* **DSA** — vanilla Dynamical Similarity Analysis (Ostrow et al. 2024) over
  PCA-reduced state. Each cluster is one "system" (its (code × body)
  trajectories). NxN angular distance matrix + heatmap.
* **InputDSA** — Subspace-DMDc cross-system distance with proprioception as the
  control input ``U``. **Diagnostic only** here: in held-code closed-loop
  rollouts the code is constant and proprio is highly redundant with the
  hidden state, so InputDSA's exogenous-input assumption is violated.
* **jPCA** (Churchland) — per-system skew-symmetric fit of the linearized
  dynamics; rotational strength + top rotational plane (limit-cycle detector).
* **rSLDS** (Linderman ``ssm``) — switching linear dynamical system; for
  ``mmd``/``label`` strategies ``K = num_clusters``, for ``raw`` a small fixed
  ``K`` with a code→regime confusion. Figures: latent phase portrait colored by
  regime, per-regime eigenvalue spectrum, transition matrix.

**Strategies** (``dynamics.strategies``):

* ``mmd``   — MMD-merged clusters (``clusters_{pose}.json``).
* ``label`` — label-majority clusters (``clusters_by_label_{pose}.json``).
* ``raw``   — one singleton cluster per top code (no merging). Codes are
  ordered by behavior label so within-behavior (trivial) vs cross-behavior
  (interesting) similarity is visually separable.

**Representations** (``dynamics.representations``):

* ``hidden`` — GRU hidden state only.
* ``joint``  — ``[hidden, root-stripped qpos]``, each block standardized then
  joint-PCA'd (the embodied (hidden, body) state).

Outputs under ``dynamics_output_base_dir/<run_name>/<pose>_<strategy>/<rep>/``:
    pca_basis.npz
    dsa_distance.npz / dsa_heatmap.{png,pdf}
    inputdsa_distance.npz / inputdsa_heatmap.{png,pdf}     (diagnostic)
    jpca_metrics.npz / jpca_projection.png
    rslds_fit.pkl / rslds_elbos.npy / rslds_z_per_trial.npz
    rslds_latent_phaseportrait.png / rslds_regime_eigs.png
    rslds_transition.png / regime_traces.png
    rslds_code_regime_confusion.png                        (raw only)
    composition.json

Usage:
    cd code2act
    python -m c2a.experiments.code_grid.run_code_grid_dynamics \
        output.run_name=best_vae dynamics.poses=[walk] \
        dynamics.strategies=[mmd,label,raw] dynamics.run_rslds=true
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import hydra
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

# Same Orbax/JAX record_scalar monkeypatch as the rollouts module.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

from c2a.experiments.shared.code_labels import (
    cluster_majority_label, load_code_labels_json,
)
from c2a.paths import get_c2a_root, get_repo_root

REPO_ROOT = get_repo_root()
log = logging.getLogger(__name__)

# Behavior id → display + ordering rank (matches shared/code_labels.py ints:
# -1 UNLABELED, 0 Immobile, 1 Rear, 2 Walk, 3 Turn).
_BEHAVIOR_NAME = {-1: "Unlabeled", 0: "Immobile", 1: "Rear", 2: "Walk", 3: "Turn"}


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Load rollouts + build per-cluster systems
# ---------------------------------------------------------------------------


def _load_rollouts(
    data_dir: Path, pose: str, codes: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Load every saved rollout npz for the given pose."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for c in codes:
        p = data_dir / f"rollout_{pose}_code{int(c):03d}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing rollout {p}. Re-run run_code_grid_rollouts first."
            )
        d = np.load(p, allow_pickle=True)
        out[int(c)] = {k: d[k] for k in d.files}
    return out


def _pooled_stats(
    rollouts: dict[int, dict[str, np.ndarray]],
    codes: list[int], key: str, col_slice: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled per-feature (mean, std) across all (code, body, frame) rows."""
    pool = []
    for c in codes:
        a = rollouts[int(c)][key]
        if col_slice is not None:
            a = a[..., col_slice]
        pool.append(a.reshape(-1, a.shape[-1]))
    pool = np.concatenate(pool, axis=0)
    mean = pool.mean(axis=0, keepdims=True)
    std = pool.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def _state_feature(
    rollouts: dict[int, dict[str, np.ndarray]],
    code: int, representation: str,
    h_stats: tuple | None, q_stats: tuple | None,
) -> np.ndarray:
    """Per-(body, frame) state feature for a code: ``(K, T, F)``.

    ``hidden`` → GRU hidden state. ``joint`` → standardized
    [hidden, root-stripped qpos] concatenation.
    """
    h = rollouts[int(code)]["hidden"]  # (K, T, H)
    if representation == "hidden":
        return h
    q = rollouts[int(code)]["qpos"][..., 3:]  # drop 3 root-translation dims
    hm, hs = h_stats
    qm, qs = q_stats
    return np.concatenate([(h - hm) / hs, (q - qm) / qs], axis=-1)


def _build_trials(
    clusters: list[list[int]],
    rollouts: dict[int, dict[str, np.ndarray]],
    representation: str,
    pca_components: int,
    u_pca_dim: int,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], np.ndarray, np.ndarray]:
    """Per-cluster PCA-reduced state trials (X) + proprio trials (U).

    Returns ``(X_per_cluster, U_per_cluster, cluster_labels_per_trial,
    x_pca_basis)``. One shared X-PCA and one shared U-PCA are fit across all
    codes so every system lives in the same subspace.
    """
    from sklearn.decomposition import PCA

    all_codes = [int(c) for cl in clusters for c in cl]

    if representation == "joint":
        h_stats = _pooled_stats(rollouts, all_codes, "hidden")
        q_stats = _pooled_stats(rollouts, all_codes, "qpos", slice(3, None))
    else:
        h_stats = q_stats = None

    feats: dict[int, np.ndarray] = {}
    x_pool, u_pool = [], []
    for c in all_codes:
        f = _state_feature(rollouts, c, representation, h_stats, q_stats)
        feats[c] = f
        x_pool.append(f.reshape(-1, f.shape[-1]))
        p = rollouts[c]["proprio"]
        u_pool.append(p.reshape(-1, p.shape[-1]))

    x_pool = np.concatenate(x_pool, axis=0)
    n_comp = min(pca_components, x_pool.shape[1], x_pool.shape[0] - 1)
    x_pca = PCA(n_components=n_comp, random_state=0).fit(x_pool)

    u_pool = np.concatenate(u_pool, axis=0)
    u_comp = min(u_pca_dim, u_pool.shape[1], u_pool.shape[0] - 1)
    u_pca = PCA(n_components=u_comp, random_state=0).fit(u_pool)

    X_per: list[list[np.ndarray]] = []
    U_per: list[list[np.ndarray]] = []
    labels: list[int] = []
    for ci, cluster in enumerate(clusters):
        Xc, Uc = [], []
        for code in cluster:
            f = feats[int(code)]
            p = rollouts[int(code)]["proprio"]
            K = f.shape[0]
            for k in range(K):
                Xc.append(x_pca.transform(f[k]).astype(np.float32))
                Uc.append(u_pca.transform(p[k]).astype(np.float32))
                labels.append(ci)
        X_per.append(Xc)
        U_per.append(Uc)
    return X_per, U_per, np.asarray(labels, dtype=np.int32), x_pca.components_


# ---------------------------------------------------------------------------
# DSA + InputDSA
# ---------------------------------------------------------------------------


def _run_dsa(
    per_cluster_trials: list[list[np.ndarray]],
    rank: int = 32, n_delays: int = 5, score_method: str = "angular",
) -> np.ndarray | None:
    """Vanilla DSA: one ``(n_trials, T, pca)`` system per cluster → NxN."""
    try:
        from DSA import DSA as PlainDSA
    except Exception as e:  # noqa: BLE001
        log.warning(
            "  DSA import failed (%s). Skipping DSA/InputDSA. "
            "Fix the venv torch CUDA deps to enable them.", e,
        )
        return None
    systems = [np.stack(trials, axis=0) for trials in per_cluster_trials]
    log.info(f"  DSA: {len(systems)} systems, rank={rank}, n_delays={n_delays}")
    dsa = PlainDSA(
        X=systems, n_delays=n_delays, rank=rank,
        score_method=score_method, device="cpu", verbose=False,
    )
    D = np.asarray(dsa.fit_score())
    if D.ndim == 3:
        D = D.mean(axis=0)
    return D


def _run_input_dsa(
    X_per_cluster: list[list[np.ndarray]],
    U_per_cluster: list[list[np.ndarray]],
    rank: int, n_delays: int, lamb: float, backend: str,
    score_method: str = "angular",
) -> np.ndarray | None:
    """InputDSA (Subspace-DMDc + ControllabilitySimilarityTransformDist).

    Returns the **state** distance component. Diagnostic only for held-code
    rollouts (exogenous-input assumption violated).
    """
    try:
        from DSA import InputDSA
        from DSA.dsa import (
            SubspaceDMDcConfig,
            ControllabilitySimilarityTransformDistConfig,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("  InputDSA import failed (%s). Skipping InputDSA.", e)
        return None

    X_systems = [np.stack(trials, axis=0) for trials in X_per_cluster]
    U_systems = [np.stack(trials, axis=0) for trials in U_per_cluster]
    dmd_cfg = SubspaceDMDcConfig(
        n_delays=n_delays, rank=rank, lamb=lamb, backend=backend,
    )
    sim_cfg = ControllabilitySimilarityTransformDistConfig(
        score_method=score_method, compare="joint",
        return_distance_components=True,
    )
    log.info(f"  InputDSA: {len(X_systems)} systems, rank={rank} (diagnostic)")
    idsa = InputDSA(
        X=X_systems, X_control=U_systems,
        dmd_config=dmd_cfg, simdist_config=sim_cfg,
        device="cpu", verbose=False,
    )
    out = idsa.fit_score()
    if isinstance(out, tuple) and len(out) == 3:
        _joint, state, _ctrl = out
    else:
        out = np.asarray(out)
        state = out[..., 1] if (out.ndim == 3 and out.shape[-1] == 3) else out
    state = np.asarray(state)
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    state = np.asarray(state)
    if state.ndim == 3:
        state = state[..., 1] if state.shape[-1] == 3 else state.mean(axis=-1)
    return state


def _plot_distance_heatmap(
    D: np.ndarray,
    cluster_sizes: list[int],
    pose: str,
    out_path: Path,
    cluster_labels: list[str] | None,
    title: str,
    block_boundaries: list[int] | None = None,
    cbar_label: str = "DSA angular distance",
) -> None:
    """NxN distance heatmap; per-cell text only when small; behavior blocks."""
    n = D.shape[0]
    fig, ax = plt.subplots(figsize=(0.34 * n + 2.0, 0.34 * n + 1.6))
    im = ax.imshow(D, cmap="viridis", aspect="equal", interpolation="none")
    if cluster_labels is not None:
        ticks = [f"c{i}[{cluster_labels[i]}]({sz})"
                 for i, sz in enumerate(cluster_sizes)]
    else:
        ticks = [f"c{i}({sz})" for i, sz in enumerate(cluster_sizes)]
    fs = max(3.0, 8 - 0.11 * n)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ticks, rotation=90, fontsize=fs)
    ax.set_yticklabels(ticks, fontsize=fs)
    if block_boundaries:
        for b in block_boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=1.0)
            ax.axvline(b - 0.5, color="white", linewidth=1.0)
    if n <= 16:
        finite = D[np.isfinite(D)]
        vmax = float(finite.max()) if finite.size else 1.0
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center",
                            fontsize=5, color="white" if D[i, j] < vmax * 0.6 else "black")
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# jPCA (Churchland): skew-symmetric fit of linearized dynamics
# ---------------------------------------------------------------------------


def _fit_jpca_one(trials: list[np.ndarray]) -> dict[str, Any]:
    """Fit skew-symmetric M minimizing ||dX - X M||_F over a cluster's trials.

    Solves the Lyapunov normal equation C M + M C = D - D^T (C = X^T X,
    D = X^T dX) for the skew-constrained least-squares M. Returns rotational
    strength (R^2 of the rotational fit), top rotational frequency, eigenvalues,
    and 2-D jPC-plane projections of each trial.
    """
    from scipy.linalg import solve_sylvester

    Xs = [t[:-1] for t in trials]
    dXs = [np.diff(t, axis=0) for t in trials]
    Xc = np.concatenate(Xs, axis=0).astype(np.float64)
    dXc = np.concatenate(dXs, axis=0).astype(np.float64)
    C = Xc.T @ Xc
    Dm = Xc.T @ dXc
    try:
        M = solve_sylvester(C, C, Dm - Dm.T)
    except Exception:  # noqa: BLE001
        M = np.linalg.lstsq(Xc, dXc, rcond=None)[0]
    M = 0.5 * (M - M.T)  # enforce skew

    resid = dXc - Xc @ M
    r2 = 1.0 - float((resid ** 2).sum()) / max(float((dXc ** 2).sum()), 1e-12)

    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w.imag))
    top = order[0]
    freq = float(abs(w[top].imag))
    v = V[:, top]
    u1 = np.real(v); u2 = np.imag(v)
    n1 = np.linalg.norm(u1) + 1e-12
    n2 = np.linalg.norm(u2) + 1e-12
    plane = np.stack([u1 / n1, u2 / n2], axis=1)  # (d, 2)
    projs = [np.asarray(t @ plane) for t in trials]
    return {"r2": r2, "freq": freq, "eigs": w, "projs": projs}


def _run_jpca(per_cluster_trials: list[list[np.ndarray]]) -> list[dict[str, Any]]:
    out = []
    for trials in per_cluster_trials:
        if len(trials) == 0:
            out.append({"r2": float("nan"), "freq": float("nan"),
                        "eigs": np.array([]), "projs": []})
            continue
        out.append(_fit_jpca_one(trials))
    return out


def _plot_jpca(
    jpca: list[dict[str, Any]],
    cluster_label_strings: list[str],
    pose: str,
    out_path: Path,
    title_suffix: str,
    max_trials_per_panel: int = 12,
) -> None:
    """Grid of per-cluster rotational phase portraits + rotational-strength bar."""
    n = len(jpca)
    ncols = min(8, max(1, int(np.ceil(np.sqrt(n)))))
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(2.0 * ncols, 2.0 * nrows + 1.4))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.9])

    cmap = plt.get_cmap("tab20")
    for i, res in enumerate(jpca):
        r, c = i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        projs = res["projs"][:max_trials_per_panel]
        for pr in projs:
            ax.plot(pr[:, 0], pr[:, 1], color=cmap(i % 20), linewidth=0.5, alpha=0.7)
            ax.plot(pr[0, 0], pr[0, 1], "o", color="black", markersize=1.5)
        lab = cluster_label_strings[i] if i < len(cluster_label_strings) else "?"
        ax.set_title(f"c{i}[{lab}]\nrot R²={res['r2']:.2f}", fontsize=5.5)
        ax.set_xticks([]); ax.set_yticks([])

    # rotational-strength bar across all clusters
    axb = fig.add_subplot(gs[nrows, :])
    r2s = [res["r2"] for res in jpca]
    axb.bar(range(n), r2s, color=[cmap(i % 20) for i in range(n)])
    axb.set_ylabel("rotational R²", fontsize=7)
    axb.set_xlabel("cluster", fontsize=7)
    axb.set_xticks(range(n))
    axb.set_xticklabels(
        [f"c{i}[{cluster_label_strings[i] if i < len(cluster_label_strings) else '?'}]"
         for i in range(n)],
        rotation=90, fontsize=max(3.0, 7 - 0.1 * n),
    )
    axb.set_ylim(0, 1)
    fig.suptitle(f"jPCA rotational structure — {pose} ({title_suffix})", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# rSLDS
# ---------------------------------------------------------------------------


def _fit_rslds(
    per_cluster_trials: list[list[np.ndarray]],
    cluster_labels_per_trial: np.ndarray,
    K: int,
    D: int,
    num_iters: int,
    seed: int,
    max_trials: int,
) -> tuple[Any, np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Fit one rSLDS over all trials (subsampled per-cluster if > max_trials).

    Returns ``(model, elbos, z_per_trial, used_cluster_labels, x_per_trial)``.
    """
    from c2a.figures.plot_hidden_rslds import fit_rslds as _fit

    total = sum(len(t) for t in per_cluster_trials)
    pct = per_cluster_trials
    if total > max_trials:
        rng = np.random.RandomState(seed)
        frac = max_trials / total
        pct = []
        for trials in per_cluster_trials:
            keep = min(len(trials), max(1, int(round(len(trials) * frac))))
            idx = sorted(rng.choice(len(trials), size=keep, replace=False))
            pct.append([trials[i] for i in idx])

    datas: list[np.ndarray] = []
    used_labels: list[int] = []
    for ci, trials in enumerate(pct):
        datas += [np.asarray(t, dtype=np.float64) for t in trials]
        used_labels += [ci] * len(trials)
    log.info(
        f"  rSLDS: K={K} regimes, D={D}, {len(datas)} trials (of {total}), "
        f"T={datas[0].shape[0]}, N={datas[0].shape[1]}, num_iters={num_iters}"
    )
    model, elbos, z_per_trial, x_per_trial = _fit(
        datas, K=K, D=D, num_iters=num_iters, seed=seed, verbose=False,
        return_latents=True,
    )
    return (model, elbos, z_per_trial,
            np.asarray(used_labels, dtype=np.int32), x_per_trial)


def _regime_purity(
    z_per_trial: list[np.ndarray], cluster_labels: np.ndarray, K: int,
) -> dict[str, Any]:
    """Fraction of frames in each cluster's modal regime."""
    per_mode: dict[int, int] = {}
    n_clusters = int(cluster_labels.max()) + 1 if len(cluster_labels) else 0
    for ci in range(n_clusters):
        mask = cluster_labels == ci
        if not mask.any():
            continue
        zs = np.concatenate([z_per_trial[i] for i in np.where(mask)[0]])
        per_mode[ci] = int(np.bincount(zs.astype(np.int64), minlength=K).argmax())
    per_purity: dict[int, float] = {}
    for ci, mode in per_mode.items():
        zs = np.concatenate([z_per_trial[i] for i in np.where(cluster_labels == ci)[0]])
        per_purity[ci] = float((zs == mode).mean())
    return {
        "per_cluster_modal_regime": {str(k): v for k, v in per_mode.items()},
        "per_cluster_purity": {str(k): v for k, v in per_purity.items()},
        "mean_purity": float(np.mean(list(per_purity.values()))) if per_purity else 0.0,
    }


def _plot_regime_traces(
    z_per_trial: list[np.ndarray], cluster_labels: np.ndarray, K: int,
    pose: str, out_path: Path, cluster_label_strings: list[str] | None,
    title_suffix: str,
) -> None:
    order = np.argsort(cluster_labels, kind="stable")
    z_mat = np.stack([z_per_trial[i] for i in order], axis=0)
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.05 * len(order))))
    ax.imshow(z_mat, aspect="auto", interpolation="none", cmap="tab20",
              vmin=0, vmax=max(K - 1, 1))
    sorted_labels = cluster_labels[order]
    for b in np.where(np.diff(sorted_labels) != 0)[0]:
        ax.axhline(y=b + 0.5, color="white", linewidth=0.5)
    ax.set_xlabel("frame"); ax.set_ylabel("trials (grouped by cluster)")
    ax.set_title(f"rSLDS discrete state per trial — {pose} ({title_suffix})")
    fig.tight_layout(); fig.savefig(str(out_path), dpi=180); plt.close(fig)


def _plot_rslds_latent_phase(
    x_per_trial: list[np.ndarray], z_per_trial: list[np.ndarray], K: int,
    pose: str, out_path: Path, title_suffix: str, max_pts: int = 40000,
) -> None:
    """2-D continuous-latent phase portrait, points colored by discrete regime."""
    X = np.concatenate([x[:, :2] for x in x_per_trial if x.shape[1] >= 2], axis=0)
    Z = np.concatenate([z[:len(x)] for x, z in zip(x_per_trial, z_per_trial)
                        if x.shape[1] >= 2], axis=0)
    if X.shape[0] > max_pts:
        sel = np.random.RandomState(0).choice(X.shape[0], max_pts, replace=False)
        X, Z = X[sel], Z[sel]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cmap = plt.get_cmap("tab10")
    for k in range(K):
        m = Z == k
        if m.any():
            ax.scatter(X[m, 0], X[m, 1], s=2, alpha=0.35, color=cmap(k % 10),
                       label=f"z{k}", edgecolors="none")
    ax.set_xlabel("latent x₁"); ax.set_ylabel("latent x₂")
    ax.legend(markerscale=3, fontsize=6, loc="best", frameon=False)
    ax.set_title(f"rSLDS latent phase portrait — {pose} ({title_suffix})", fontsize=9)
    fig.tight_layout(); fig.savefig(str(out_path), dpi=180); plt.close(fig)


def _plot_rslds_eigs(
    model: Any, K: int, pose: str, out_path: Path, title_suffix: str,
) -> None:
    """Per-regime dynamics-matrix eigenvalues in the complex plane."""
    As = getattr(getattr(model, "dynamics", None), "As", None)
    if As is None:
        return
    As = np.asarray(As)
    ncols = min(K, 4)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.4 * nrows),
                             squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 200)
    for k in range(K):
        ax = axes[k // ncols][k % ncols]
        ev = np.linalg.eigvals(As[k])
        ax.plot(np.cos(theta), np.sin(theta), color="#999", linewidth=0.6)
        ax.axhline(0, color="#ccc", lw=0.4); ax.axvline(0, color="#ccc", lw=0.4)
        ax.scatter(ev.real, ev.imag, s=18, color="#0072B2", zorder=3)
        rot = np.any(np.abs(ev.imag) > 1e-3)
        ax.set_title(f"z{k} {'(rot)' if rot else ''}", fontsize=8)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.tick_params(labelsize=5)
    for j in range(K, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"rSLDS per-regime eigenvalues — {pose} ({title_suffix})", fontsize=9)
    fig.tight_layout(); fig.savefig(str(out_path), dpi=180); plt.close(fig)


def _plot_rslds_transition(
    z_per_trial: list[np.ndarray], K: int, pose: str, out_path: Path,
    title_suffix: str,
) -> None:
    """Empirical regime transition matrix P(z_{t+1}|z_t)."""
    T = np.zeros((K, K), dtype=np.float64)
    for z in z_per_trial:
        for a, b in zip(z[:-1], z[1:]):
            T[int(a), int(b)] += 1
    row = T.sum(axis=1, keepdims=True)
    P = T / np.maximum(row, 1)
    fig, ax = plt.subplots(figsize=(0.6 * K + 1.6, 0.6 * K + 1.4))
    im = ax.imshow(P, cmap="magma", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if P[i, j] < 0.6 else "black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"z{j}" for j in range(K)], fontsize=7)
    ax.set_yticklabels([f"z{j}" for j in range(K)], fontsize=7)
    ax.set_xlabel("z_{t+1}"); ax.set_ylabel("z_t")
    fig.colorbar(im, ax=ax, label="P", fraction=0.046)
    ax.set_title(f"rSLDS transitions — {pose} ({title_suffix})", fontsize=9)
    fig.tight_layout(); fig.savefig(str(out_path), dpi=180); plt.close(fig)


def _plot_code_regime_confusion(
    z_per_trial: list[np.ndarray], cluster_labels: np.ndarray, K: int,
    cluster_label_strings: list[str], pose: str, out_path: Path,
    block_boundaries: list[int] | None, title_suffix: str,
) -> None:
    """Raw strategy: code (row) → regime (col) frame-fraction confusion."""
    n_codes = int(cluster_labels.max()) + 1 if len(cluster_labels) else 0
    conf = np.zeros((n_codes, K), dtype=np.float64)
    for ci in range(n_codes):
        idx = np.where(cluster_labels == ci)[0]
        if len(idx) == 0:
            continue
        zs = np.concatenate([z_per_trial[i] for i in idx])
        conf[ci] = np.bincount(zs.astype(np.int64), minlength=K)
    conf = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(0.5 * K + 2.0, 0.18 * n_codes + 1.6))
    im = ax.imshow(conf, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"z{j}" for j in range(K)], fontsize=7)
    ax.set_yticks(range(n_codes))
    ax.set_yticklabels(
        [f"c{i}[{cluster_label_strings[i] if i < len(cluster_label_strings) else '?'}]"
         for i in range(n_codes)],
        fontsize=max(3.0, 7 - 0.08 * n_codes),
    )
    if block_boundaries:
        for b in block_boundaries:
            ax.axhline(b - 0.5, color="white", linewidth=1.0)
    ax.set_xlabel("rSLDS regime"); ax.set_ylabel("code (by behavior label)")
    fig.colorbar(im, ax=ax, label="frame fraction", fraction=0.046)
    ax.set_title(f"code → regime — {pose} ({title_suffix})", fontsize=9)
    fig.tight_layout(); fig.savefig(str(out_path), dpi=180)
    fig.savefig(str(out_path).replace(".png", ".pdf")); plt.close(fig)


def _composition_report(
    clusters: list[list[int]], usage: np.ndarray, purity_info: dict[str, Any],
    cluster_info: dict[str, Any], pose: str, out_path: Path,
    cluster_label_strings: list[str] | None,
) -> None:
    cutoff = cluster_info.get("cutoff_used")
    cutoff_ser = (None if cutoff is None or
                  (isinstance(cutoff, float) and np.isnan(cutoff)) else float(cutoff))
    payload = {
        "pose": pose,
        "merge_strategy": str(cluster_info.get("strategy")),
        "merge_cutoff_used": cutoff_ser,
        "num_clusters": len(clusters),
        "rslds": {
            "mean_purity": float(purity_info.get("mean_purity", 0.0)),
            "per_cluster_modal_regime": purity_info.get("per_cluster_modal_regime", {}),
            "per_cluster_purity": purity_info.get("per_cluster_purity", {}),
        },
        "clusters": [
            {
                "cluster_id": i, "size": len(cl),
                "codes": [int(c) for c in cl],
                "code_usages": [int(usage[c]) for c in cl],
                "usage_total": int(sum(int(usage[c]) for c in cl)),
                "label": (cluster_label_strings[i]
                          if cluster_label_strings is not None
                          and i < len(cluster_label_strings) else None),
            }
            for i, cl in enumerate(clusters)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Per (pose, strategy, representation) runner
# ---------------------------------------------------------------------------


def _run_one_pass(
    out_dir: Path,
    pose: str,
    strategy: str,
    representation: str,
    clusters: list[list[int]],
    cluster_info: dict[str, Any],
    cluster_label_strings: list[str],
    block_boundaries: list[int] | None,
    rollouts: dict[int, dict[str, np.ndarray]],
    usage: np.ndarray,
    dyn: dict[str, Any],
    seed: int,
    reuse: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{strategy}/{representation}"
    log.info(f"  [{pose}/{suffix}] {len(clusters)} clusters: "
             f"sizes {[len(c) for c in clusters[:8]]}...")

    X_per, U_per, cluster_labels, pca_basis = _build_trials(
        clusters, rollouts, representation,
        pca_components=int(dyn["pca_components"]),
        u_pca_dim=int(dyn["inputdsa_u_pca_dim"]),
    )
    np.savez(out_dir / "pca_basis.npz", components=pca_basis,
             n_components=np.int32(pca_basis.shape[0]))
    cluster_sizes = [len(c) for c in clusters]

    # --- DSA ---
    if dyn["run_dsa"]:
        dsa_npz = out_dir / "dsa_distance.npz"
        if reuse and dsa_npz.exists():
            D = np.load(dsa_npz)["distance"]
            log.info(f"    DSA: reusing {dsa_npz.name}")
        else:
            D = _run_dsa(X_per, rank=int(dyn["dsa_rank"]),
                         n_delays=int(dyn["dsa_n_delays"]))
            if D is not None:
                np.savez(dsa_npz, distance=D, cluster_sizes=np.asarray(cluster_sizes))
        if D is not None:
            _plot_distance_heatmap(
                D, cluster_sizes, pose, out_dir / "dsa_heatmap.png",
                cluster_labels=cluster_label_strings,
                title=f"DSA — {pose} ({strategy}/{representation})",
                block_boundaries=block_boundaries, cbar_label="DSA angular distance",
            )

    # --- InputDSA (diagnostic) ---
    if dyn["run_inputdsa"]:
        idsa_npz = out_dir / "inputdsa_distance.npz"
        if reuse and idsa_npz.exists():
            Di = np.load(idsa_npz)["distance"]
            log.info(f"    InputDSA: reusing {idsa_npz.name}")
        else:
            Di = _run_input_dsa(
                X_per, U_per, rank=int(dyn["dsa_rank"]),
                n_delays=int(dyn["dsa_n_delays"]),
                lamb=float(dyn["inputdsa_lamb"]), backend=str(dyn["inputdsa_backend"]),
            )
            if Di is not None:
                np.savez(idsa_npz, distance=Di, cluster_sizes=np.asarray(cluster_sizes))
        if Di is not None:
            _plot_distance_heatmap(
                Di, cluster_sizes, pose, out_dir / "inputdsa_heatmap.png",
                cluster_labels=cluster_label_strings,
                title=f"InputDSA (diagnostic) — {pose} ({strategy}/{representation})",
                block_boundaries=block_boundaries, cbar_label="InputDSA state distance",
            )

    # --- jPCA ---
    if dyn["run_jpca"]:
        jpca = _run_jpca(X_per)
        np.savez(
            out_dir / "jpca_metrics.npz",
            r2=np.asarray([r["r2"] for r in jpca]),
            freq=np.asarray([r["freq"] for r in jpca]),
        )
        _plot_jpca(jpca, cluster_label_strings, pose,
                   out_dir / "jpca_projection.png",
                   title_suffix=f"{strategy}/{representation}")

    # --- rSLDS ---
    if not dyn["run_rslds"]:
        log.info(f"    rSLDS deferred (run_rslds=false) for {pose}/{suffix}.")
        return

    is_raw = strategy == "raw"
    K = int(dyn["raw_rslds_K"]) if is_raw else len(clusters)
    model, elbos, z_per_trial, used_labels, x_per_trial = _fit_rslds(
        X_per, cluster_labels, K=K, D=int(dyn["rslds_continuous_dim"]),
        num_iters=int(dyn["rslds_iters"]), seed=seed,
        max_trials=int(dyn["rslds_max_trials"]),
    )
    np.save(out_dir / "rslds_elbos.npy", elbos)
    np.savez(out_dir / "rslds_z_per_trial.npz",
             **{f"trial_{i:04d}": z for i, z in enumerate(z_per_trial)},
             cluster_labels=used_labels)
    with (out_dir / "rslds_fit.pkl").open("wb") as f:
        pickle.dump(model, f)
    log.info(f"    rSLDS fit. ELBO {float(elbos[0]):.1f} → {float(elbos[-1]):.1f}.")

    purity = _regime_purity(z_per_trial, used_labels, K=K)
    log.info(f"    mean_purity={purity['mean_purity']:.3f}")

    _plot_regime_traces(z_per_trial, used_labels, K, pose,
                        out_dir / "regime_traces.png", cluster_label_strings,
                        title_suffix=f"{strategy}/{representation}")
    _plot_rslds_latent_phase(x_per_trial, z_per_trial, K, pose,
                             out_dir / "rslds_latent_phaseportrait.png",
                             title_suffix=f"{strategy}/{representation}")
    _plot_rslds_eigs(model, K, pose, out_dir / "rslds_regime_eigs.png",
                     title_suffix=f"{strategy}/{representation}")
    _plot_rslds_transition(z_per_trial, K, pose, out_dir / "rslds_transition.png",
                           title_suffix=f"{strategy}/{representation}")
    if is_raw:
        _plot_code_regime_confusion(
            z_per_trial, used_labels, K, cluster_label_strings, pose,
            out_dir / "rslds_code_regime_confusion.png",
            block_boundaries, title_suffix=f"{strategy}/{representation}",
        )
    _composition_report(clusters, usage, purity, cluster_info, pose,
                        out_dir / "composition.json", cluster_label_strings)


# ---------------------------------------------------------------------------
# Cluster construction per strategy
# ---------------------------------------------------------------------------


def _cluster_label_strings_from_json(
    payload: dict[str, Any], clusters: list[list[int]],
    code_labels: dict[int, dict[str, Any]],
) -> list[str]:
    out: list[str] = []
    json_clusters = payload.get("clusters", [])
    for i, cl in enumerate(clusters):
        if i < len(json_clusters) and "label" in json_clusters[i]:
            out.append(str(json_clusters[i]["label"]))
        else:
            out.append(cluster_majority_label(cl, code_labels))
    return out


def _raw_clusters_ordered(
    top_codes: list[int], code_labels: dict[int, dict[str, Any]],
) -> tuple[list[list[int]], list[str], list[int]]:
    """Singleton clusters (one per code), ordered by behavior label.

    Returns ``(clusters, label_strings, block_boundaries)`` where boundaries
    mark where the behavior id changes (for heatmap block separators).
    """
    def _beh(c: int) -> int:
        return int(code_labels.get(int(c), {}).get("behavior", -1))

    def _lbl(c: int) -> str:
        return str(code_labels.get(int(c), {}).get("label", "unlabeled"))

    ordered = sorted(top_codes, key=lambda c: (_beh(c), _lbl(c), int(c)))
    clusters = [[int(c)] for c in ordered]
    labels = [_lbl(c) for c in ordered]
    behs = [_beh(c) for c in ordered]
    boundaries = [i for i in range(1, len(behs)) if behs[i] != behs[i - 1]]
    return clusters, labels, boundaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_grid_rollouts_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Grid Dynamics (DSA + InputDSA + jPCA + rSLDS) ===")

    rollouts_base = Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    data_dir = rollouts_base / "data"
    merged_dir = rollouts_base / "merged"
    labels_dir = rollouts_base / "labels"
    if not (data_dir.exists() and merged_dir.exists()):
        raise RuntimeError(
            f"Expected rollouts + merge under {rollouts_base}. "
            "Run run_code_grid_rollouts first."
        )

    usage_data = np.load(data_dir / "usage_rank.npz")
    top_codes = [int(c) for c in usage_data["top_codes"].tolist()]
    usage = usage_data["usage"]

    code_labels_path = labels_dir / "code_labels.json"
    if not code_labels_path.exists():
        raise RuntimeError(f"Missing {code_labels_path}. Re-run rollouts.")
    code_labels = load_code_labels_json(code_labels_path)

    dyn_base = (
        Path(_abs(cfg.get("dynamics_output_base_dir",
                          "./outputs/code_grid/code_grid_dynamics")))
        / str(cfg.output.run_name)
    )
    dyn_base.mkdir(parents=True, exist_ok=True)

    d = cfg.get("dynamics", {})
    dyn = {
        "pca_components": int(d.get("pca_components", 32)),
        "run_dsa": bool(d.get("run_dsa", True)),
        "dsa_rank": int(d.get("dsa_rank", 32)),
        "dsa_n_delays": int(d.get("dsa_n_delays", 5)),
        "run_inputdsa": bool(d.get("run_inputdsa", True)),
        "inputdsa_u_pca_dim": int(d.get("inputdsa_u_pca_dim", 16)),
        "inputdsa_lamb": float(d.get("inputdsa_lamb", 1e-3)),
        "inputdsa_backend": str(d.get("inputdsa_backend", "n4sid")),
        "run_jpca": bool(d.get("run_jpca", True)),
        "run_rslds": bool(d.get("run_rslds", False)),
        "rslds_continuous_dim": int(d.get("rslds_continuous_dim", 8)),
        "rslds_iters": int(d.get("rslds_iters", 20)),
        "rslds_max_trials": int(d.get("rslds_max_trials", 250)),
        "raw_rslds_K": int(d.get("raw_rslds_K", 6)),
    }
    poses = [str(p) for p in cfg.get("poses", ["walk", "rear"])]
    strategies = [str(s) for s in d.get("strategies", ["mmd", "label", "raw"])]
    representations = [str(r) for r in d.get("representations", ["hidden", "joint"])]
    seed = int(cfg.seed)
    reuse = bool(cfg.get("reuse_dynamics", True))
    log.info(f"poses={poses} strategies={strategies} representations={representations}")

    for pose in poses:
        log.info(f"\n=== Pose: {pose} ===")
        rollouts = _load_rollouts(data_dir, pose, top_codes)

        for strategy in strategies:
            # Build clusters + per-cluster label strings + block boundaries.
            block_boundaries: list[int] | None = None
            if strategy == "raw":
                clusters, cluster_label_strings, block_boundaries = (
                    _raw_clusters_ordered(top_codes, code_labels)
                )
                cluster_info = {"strategy": "raw", "cutoff_used": None}
            elif strategy in ("mmd", "label"):
                fname = (f"clusters_{pose}.json" if strategy == "mmd"
                         else f"clusters_by_label_{pose}.json")
                path = merged_dir / fname
                if not path.exists():
                    log.warning(f"  Skipping {strategy}: {path} not found.")
                    continue
                with path.open() as f:
                    payload = json.load(f)
                clusters = [c["codes"] for c in payload["clusters"]]
                cluster_info = {"strategy": payload.get("strategy"),
                                "cutoff_used": payload.get("cutoff_used")}
                cluster_label_strings = _cluster_label_strings_from_json(
                    payload, clusters, code_labels)
            else:
                log.warning(f"  Unknown strategy {strategy!r}; skipping.")
                continue

            for rep in representations:
                log.info(f"\n--- {pose} / {strategy} / {rep} ---")
                _run_one_pass(
                    out_dir=dyn_base / f"{pose}_{strategy}" / rep,
                    pose=pose, strategy=strategy, representation=rep,
                    clusters=clusters, cluster_info=cluster_info,
                    cluster_label_strings=cluster_label_strings,
                    block_boundaries=block_boundaries,
                    rollouts=rollouts, usage=usage, dyn=dyn,
                    seed=seed, reuse=reuse,
                )

    log.info(f"\n=== Done. Dynamics artifacts under {dyn_base}/ ===")


if __name__ == "__main__":
    main()
