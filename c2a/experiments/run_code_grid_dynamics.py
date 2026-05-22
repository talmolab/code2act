"""Experiment: DSA + rSLDS on the merged-cluster hidden state trajectories.

Reads the rollouts and clusters produced by ``run_code_grid_rollouts.py``
and runs, per pose (walk / rear) separately:

* **DSA**: vanilla Dynamical Similarity Analysis (Ostrow et al. 2024) over
  PCA-reduced hidden state. Each cluster is one "system" formed by
  concatenating the hidden trajectories of all (K × |cluster|) bodies
  for that cluster. Result: NxN distance matrix + heatmap.

* **rSLDS**: one rSLDS per pose (Linderman lab ``ssm`` package), with
  K = num_merged_clusters discrete states and D = continuous-latent
  dim (configurable). Each (cluster, body) trajectory is one trial.
  After fit, we compute a "regime purity" measure: for each trial of
  cluster c, what fraction of its frames are assigned to the most common
  rSLDS regime for cluster c? High purity ⇒ behaviorally-similar codes
  cluster into a single linear dynamics regime in the GRU.

* **Composition report**: per merged cluster, list its sub-codes (with
  usage counts) so the macro-cluster identity is interpretable.

For each pose we now run TWO clustering passes:

1. ``{pose}/`` — MMD-merged clusters (``clusters_{pose}.json``).
2. ``{pose}_label/`` — label-majority clusters (``clusters_by_label_{pose}.json``).

Both passes share the rollouts and pca basis; only the cluster assignment
and downstream DSA / rSLDS / composition differ. Cluster tick labels in
DSA heatmaps and regime-trace annotations use the per-cluster majority
label injected by the rollouts script.

Outputs under ``output.base_dir/<run_name>/{walk,rear,walk_label,rear_label}/``:
    dsa_distance.npz       — NxN angular distance matrix
    dsa_heatmap.{png,pdf}  — visualization with cluster-label tick labels
    rslds_fit.pkl          — ssm.SLDS object (reused on re-run if present)
    rslds_elbos.npy
    rslds_z_per_trial.npz  — per-trial discrete state sequences
    regime_traces.png      — z-state vs time + per-cluster label annotation
    composition.json       — cluster → sub-codes + usage + majority label

Usage:
    cd code2act
    python -m c2a.experiments.run_code_grid_dynamics
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


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Build per-cluster systems from rollouts
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


def _per_cluster_trials(
    clusters: list[list[int]],
    rollouts: dict[int, dict[str, np.ndarray]],
    pca_components: int = 32,
) -> tuple[list[list[np.ndarray]], np.ndarray, np.ndarray]:
    """For each cluster, return its list of per-body hidden trajectories.

    Args:
        clusters: list of clusters; each cluster is a list of original code ids.
        rollouts: ``{code_id: {"hidden": (K, T, h), ...}}``.
        pca_components: PCA dimensionality for DSA + rSLDS input. Set to a
            value <= hidden_dim. The reference figures use 32.

    Returns:
        ``(per_cluster_trials, cluster_labels_per_trial, pca_basis)`` where
        ``per_cluster_trials[i]`` is a list of (T, pca_components) arrays
        for cluster i.
    """
    # Fit ONE PCA across all hidden trajectories (all clusters, all bodies)
    # so all systems live in the same low-d subspace. Same convention as
    # plot_hidden_dsa.py.
    from sklearn.decomposition import PCA

    pool = []
    for c, agg in rollouts.items():
        h = agg["hidden"]  # (K, T, h)
        pool.append(h.reshape(-1, h.shape[-1]))
    pool = np.concatenate(pool, axis=0)
    n_comp = min(pca_components, pool.shape[1])
    pca = PCA(n_components=n_comp, random_state=0).fit(pool)

    per_cluster_trials: list[list[np.ndarray]] = []
    cluster_labels_per_trial: list[int] = []
    for ci, cluster in enumerate(clusters):
        trials: list[np.ndarray] = []
        for code in cluster:
            h = rollouts[int(code)]["hidden"]  # (K, T, h)
            K = h.shape[0]
            for k in range(K):
                trials.append(pca.transform(h[k]))
                cluster_labels_per_trial.append(ci)
        per_cluster_trials.append(trials)
    return per_cluster_trials, np.asarray(cluster_labels_per_trial, dtype=np.int32), pca.components_


# ---------------------------------------------------------------------------
# DSA
# ---------------------------------------------------------------------------


def _run_dsa(
    per_cluster_trials: list[list[np.ndarray]],
    rank: int = 20, n_delays: int = 5, score_method: str = "angular",
) -> np.ndarray | None:
    """One concatenated system per cluster, NxN angular distance matrix.

    Returns None (with a logged warning) if the DSA package (which pulls in
    torch) cannot import in this environment.
    """
    try:
        from DSA import DSA as PlainDSA
    except Exception as e:
        log.warning(
            "  DSA import failed (%s). Skipping DSA. rSLDS still runs. "
            "Fix the venv torch CUDA deps to enable DSA.",
            e,
        )
        return None

    systems = [
        np.concatenate(trials, axis=0) for trials in per_cluster_trials
    ]
    log.info(
        f"  DSA: {len(systems)} systems, rank={rank}, n_delays={n_delays}"
    )
    dsa = PlainDSA(
        X=systems, n_delays=n_delays, rank=rank,
        score_method=score_method, device="cpu", verbose=False,
    )
    D = dsa.fit_score()
    D = np.asarray(D)
    if D.ndim == 3:
        D = D.mean(axis=0)
    return D


def _plot_dsa_heatmap(
    D: np.ndarray,
    cluster_sizes: list[int],
    pose: str,
    out_path: Path,
    cluster_labels: list[str] | None = None,
    title_suffix: str = "",
) -> None:
    """DSA heatmap with optional per-cluster majority-label tick labels."""
    fig, ax = plt.subplots(figsize=(0.55 * D.shape[0] + 1.6, 0.55 * D.shape[0] + 1.2))
    im = ax.imshow(D, cmap="viridis", aspect="equal", interpolation="none")
    ax.set_xticks(range(D.shape[0]))
    ax.set_yticks(range(D.shape[0]))
    if cluster_labels is not None:
        tick_labels = [
            f"c{i} [{cluster_labels[i]}] (n={sz})"
            for i, sz in enumerate(cluster_sizes)
        ]
    else:
        tick_labels = [f"c{i}({sz})" for i, sz in enumerate(cluster_sizes)]
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    fig.colorbar(im, ax=ax, label="DSA angular distance", fraction=0.046)
    title = f"DSA between merged clusters — {pose} pose"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# rSLDS
# ---------------------------------------------------------------------------


def _fit_rslds(
    per_cluster_trials: list[list[np.ndarray]],
    cluster_labels_per_trial: np.ndarray,
    D: int,
    num_iters: int,
    seed: int,
) -> tuple[Any, np.ndarray, list[np.ndarray]]:
    """Fit one rSLDS over all trials of all clusters (K discrete states)."""
    from c2a.figures.plot_hidden_rslds import fit_rslds as _fit

    datas: list[np.ndarray] = []
    for trials in per_cluster_trials:
        datas += trials
    K = len(per_cluster_trials)
    log.info(
        f"  rSLDS: K={K} regimes, D={D}, {len(datas)} trials, "
        f"T={datas[0].shape[0]}, N={datas[0].shape[1]}"
    )
    model, elbos, z_per_trial = _fit(
        datas, K=K, D=D, num_iters=num_iters, seed=seed, verbose=False,
    )
    return model, elbos, z_per_trial


def _load_rslds_cache(
    out_dir: Path,
) -> tuple[Any, np.ndarray, list[np.ndarray], np.ndarray] | None:
    """Reload a cached rSLDS fit if all expected files are present."""
    pkl = out_dir / "rslds_fit.pkl"
    elbos_p = out_dir / "rslds_elbos.npy"
    z_p = out_dir / "rslds_z_per_trial.npz"
    if not (pkl.exists() and elbos_p.exists() and z_p.exists()):
        return None
    try:
        with pkl.open("rb") as f:
            model = pickle.load(f)
        elbos = np.load(elbos_p)
        zd = np.load(z_p)
        cluster_labels = zd["cluster_labels"].astype(np.int32)
        trial_keys = sorted(
            (k for k in zd.files if k.startswith("trial_")),
            key=lambda s: int(s.split("_")[1]),
        )
        z_per_trial = [zd[k] for k in trial_keys]
        return model, elbos, z_per_trial, cluster_labels
    except Exception as e:
        log.warning(f"  Failed to reload cached rSLDS ({e}); will refit.")
        return None


def _regime_purity(
    z_per_trial: list[np.ndarray],
    cluster_labels: np.ndarray,
    K: int,
) -> dict[str, float]:
    """Fraction of frames in each trial assigned to its cluster's modal regime."""
    per_cluster_mode: dict[int, int] = {}
    for ci in range(K):
        mask = cluster_labels == ci
        if not mask.any():
            continue
        zs = np.concatenate([z_per_trial[i] for i in np.where(mask)[0]])
        per_cluster_mode[ci] = int(np.bincount(zs.astype(np.int64), minlength=K).argmax())

    per_cluster_purity: dict[int, float] = {}
    for ci, mode in per_cluster_mode.items():
        mask = cluster_labels == ci
        zs = np.concatenate([z_per_trial[i] for i in np.where(mask)[0]])
        per_cluster_purity[ci] = float((zs == mode).mean())

    return {
        "per_cluster_modal_regime": {str(k): v for k, v in per_cluster_mode.items()},
        "per_cluster_purity": {str(k): v for k, v in per_cluster_purity.items()},
        "mean_purity": float(np.mean(list(per_cluster_purity.values()))) if per_cluster_purity else 0.0,
    }


def _plot_regime_traces(
    z_per_trial: list[np.ndarray],
    cluster_labels: np.ndarray,
    K: int,
    pose: str,
    out_path: Path,
    cluster_label_strings: list[str] | None = None,
    title_suffix: str = "",
) -> None:
    """Stacked z-trace plot, grouped by ground-truth cluster.

    If ``cluster_label_strings`` is provided, each cluster band is annotated
    with its majority label on the right margin.
    """
    order = np.argsort(cluster_labels, kind="stable")
    z_mat = np.stack([z_per_trial[i] for i in order], axis=0)  # (n_trials, T)
    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.06 * len(order))))
    ax.imshow(
        z_mat, aspect="auto", interpolation="none", cmap="tab20", vmin=0, vmax=max(K - 1, 1),
    )
    sorted_labels = cluster_labels[order]
    bnds = np.where(np.diff(sorted_labels) != 0)[0]
    for b in bnds:
        ax.axhline(y=b + 0.5, color="white", linewidth=0.6)

    if cluster_label_strings is not None:
        # Compute band centers (in row index) per cluster id.
        T_total = z_mat.shape[1]
        band_starts = np.concatenate(
            ([0], bnds + 1, [len(sorted_labels)])
        )
        # Walk distinct cluster ids in `sorted_labels` in encountered order.
        seen: list[int] = []
        for ci in sorted_labels.tolist():
            if not seen or seen[-1] != ci:
                seen.append(ci)
        for idx, ci in enumerate(seen):
            y0 = band_starts[idx]
            y1 = band_starts[idx + 1]
            yc = 0.5 * (y0 + y1) - 0.5
            lab = cluster_label_strings[ci] if ci < len(cluster_label_strings) else "?"
            ax.text(
                T_total * 1.005, yc,
                f"c{ci} [{lab}]",
                va="center", ha="left", fontsize=6,
                color="black", clip_on=False,
            )

    ax.set_xlabel("frame")
    ax.set_ylabel("trials (grouped by cluster)")
    title = f"rSLDS discrete state per trial — {pose} pose"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Composition report
# ---------------------------------------------------------------------------


def _composition_report(
    clusters: list[list[int]],
    usage: np.ndarray,
    purity_info: dict[str, Any],
    cluster_info: dict[str, Any],
    pose: str,
    out_path: Path,
    cluster_label_strings: list[str] | None = None,
) -> None:
    cutoff = cluster_info.get("cutoff_used")
    if cutoff is None or (isinstance(cutoff, float) and np.isnan(cutoff)):
        cutoff_serialized: float | None = None
    else:
        cutoff_serialized = float(cutoff)
    payload = {
        "pose": pose,
        "merge_strategy": str(cluster_info.get("strategy")),
        "merge_cutoff_used": cutoff_serialized,
        "num_clusters": len(clusters),
        "rslds": {
            "mean_purity": float(purity_info.get("mean_purity", 0.0)),
            "per_cluster_modal_regime": purity_info.get(
                "per_cluster_modal_regime", {}
            ),
            "per_cluster_purity": purity_info.get(
                "per_cluster_purity", {}
            ),
        },
        "clusters": [
            {
                "cluster_id": i,
                "size": len(cl),
                "codes": [int(c) for c in cl],
                "code_usages": [int(usage[c]) for c in cl],
                "usage_total": int(sum(int(usage[c]) for c in cl)),
                "label": (
                    cluster_label_strings[i]
                    if cluster_label_strings is not None and i < len(cluster_label_strings)
                    else None
                ),
            }
            for i, cl in enumerate(clusters)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Per-pose-per-strategy runner
# ---------------------------------------------------------------------------


def _run_one_pass(
    out_dir: Path,
    pose: str,
    clusters: list[list[int]],
    cluster_info: dict[str, Any],
    cluster_label_strings: list[str],
    rollouts: dict[int, dict[str, np.ndarray]],
    usage: np.ndarray,
    pca_components: int,
    dsa_rank: int,
    dsa_delays: int,
    rslds_D: int,
    rslds_iters: int,
    seed: int,
    reuse_rslds: bool,
    title_suffix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        f"  Loaded {len(clusters)} clusters: sizes "
        f"{[len(c) for c in clusters[:10]]}"
    )

    per_cluster_trials, cluster_labels, pca_basis = _per_cluster_trials(
        clusters, rollouts, pca_components=pca_components,
    )
    np.savez(
        out_dir / "pca_basis.npz",
        components=pca_basis,
        n_components=np.int32(pca_basis.shape[0]),
    )

    # --- DSA ---
    log.info(f"  --- DSA ({pose} / {title_suffix}) ---")
    dsa_npz = out_dir / "dsa_distance.npz"
    dsa_png = out_dir / "dsa_heatmap.png"
    cluster_sizes = [len(c) for c in clusters]
    if reuse_rslds and dsa_npz.exists():
        D = np.load(dsa_npz)["distance"]
        log.info(f"    DSA: reusing cached {dsa_npz}")
    else:
        D = _run_dsa(
            per_cluster_trials,
            rank=dsa_rank, n_delays=dsa_delays, score_method="angular",
        )
        if D is not None:
            np.savez(
                dsa_npz, distance=D,
                cluster_sizes=np.asarray(cluster_sizes),
            )
            log.info(
                f"    DSA done. distance range "
                f"[{float(D[np.triu_indices(D.shape[0], k=1)].min()):.4f}, "
                f"{float(D[np.triu_indices(D.shape[0], k=1)].max()):.4f}]"
            )
    if D is not None:
        _plot_dsa_heatmap(
            D, cluster_sizes, pose, dsa_png,
            cluster_labels=cluster_label_strings,
            title_suffix=title_suffix,
        )

    # --- rSLDS (with cache) ---
    log.info(f"  --- rSLDS ({pose} / {title_suffix}) ---")
    cached = _load_rslds_cache(out_dir) if reuse_rslds else None
    if cached is not None:
        model, elbos, z_per_trial, cached_cluster_labels = cached
        # Cache fidelity check: same N trials and same cluster labeling.
        if (len(z_per_trial) != len(cluster_labels)
                or not np.array_equal(cached_cluster_labels, cluster_labels)):
            log.warning(
                "    rSLDS cache cluster_labels do not match current clusters; "
                "refitting."
            )
            cached = None
    if cached is None:
        model, elbos, z_per_trial = _fit_rslds(
            per_cluster_trials, cluster_labels,
            D=rslds_D, num_iters=rslds_iters, seed=seed,
        )
        np.save(out_dir / "rslds_elbos.npy", elbos)
        np.savez(
            out_dir / "rslds_z_per_trial.npz",
            **{f"trial_{i:04d}": z for i, z in enumerate(z_per_trial)},
            cluster_labels=cluster_labels,
        )
        with (out_dir / "rslds_fit.pkl").open("wb") as f:
            pickle.dump(model, f)
        log.info(
            f"    rSLDS fit. ELBO {float(elbos[0]):.1f} → {float(elbos[-1]):.1f}."
        )
    else:
        log.info(f"    rSLDS: reusing cached fit ({len(z_per_trial)} trials)")

    purity = _regime_purity(z_per_trial, cluster_labels, K=len(clusters))
    log.info(f"    mean_purity={purity['mean_purity']:.3f}")

    _plot_regime_traces(
        z_per_trial, cluster_labels, K=len(clusters),
        pose=pose, out_path=out_dir / "regime_traces.png",
        cluster_label_strings=cluster_label_strings,
        title_suffix=title_suffix,
    )

    _composition_report(
        clusters, usage, purity, cluster_info, pose,
        out_dir / "composition.json",
        cluster_label_strings=cluster_label_strings,
    )


def _cluster_label_strings_from_json(
    payload: dict[str, Any],
    clusters: list[list[int]],
    code_labels: dict[int, dict[str, Any]],
) -> list[str]:
    """Return per-cluster majority label strings.

    Prefer the precomputed ``label`` field injected by the rollouts script;
    if absent (older artifacts), recompute from ``code_labels``.
    """
    out: list[str] = []
    json_clusters = payload.get("clusters", [])
    for i, cl in enumerate(clusters):
        if i < len(json_clusters) and "label" in json_clusters[i]:
            out.append(str(json_clusters[i]["label"]))
        else:
            out.append(cluster_majority_label(cl, code_labels))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_grid_rollouts_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Grid Dynamics (DSA + rSLDS) ===")

    rollouts_base = (
        Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    )
    data_dir = rollouts_base / "data"
    merged_dir = rollouts_base / "merged"
    labels_dir = rollouts_base / "labels"
    if not (data_dir.exists() and merged_dir.exists()):
        raise RuntimeError(
            f"Expected rollouts + merge under {rollouts_base}. "
            "Run run_code_grid_rollouts first."
        )

    # Top codes from usage_rank.npz produced by rollouts.
    usage_data = np.load(data_dir / "usage_rank.npz")
    top_codes = usage_data["top_codes"].tolist()
    usage = usage_data["usage"]

    # Per-code labels (must be present; rollouts script writes it).
    code_labels_path = labels_dir / "code_labels.json"
    if not code_labels_path.exists():
        raise RuntimeError(
            f"Missing {code_labels_path}. Re-run run_code_grid_rollouts "
            "to materialize per-code majority labels."
        )
    code_labels = load_code_labels_json(code_labels_path)

    dyn_base = (
        Path(_abs(cfg.get("dynamics_output_base_dir", "./outputs/code_grid_dynamics")))
        / str(cfg.output.run_name)
    )
    dyn_base.mkdir(parents=True, exist_ok=True)

    pca_components = int(cfg.get("dynamics", {}).get("pca_components", 32))
    rslds_D = int(cfg.get("dynamics", {}).get("rslds_continuous_dim", 8))
    rslds_iters = int(cfg.get("dynamics", {}).get("rslds_iters", 50))
    dsa_rank = int(cfg.get("dynamics", {}).get("dsa_rank", 20))
    dsa_delays = int(cfg.get("dynamics", {}).get("dsa_n_delays", 5))
    seed = int(cfg.seed)
    reuse = bool(cfg.get("reuse_dynamics", True))

    for pose in ("walk", "rear"):
        log.info(f"\n=== Pose: {pose} ===")

        # Preload rollouts once per pose; reused by both strategies.
        rollouts = _load_rollouts(data_dir, pose, top_codes)

        # --- Strategy 1: MMD-merged clusters ---
        mmd_path = merged_dir / f"clusters_{pose}.json"
        with mmd_path.open() as f:
            mmd_payload = json.load(f)
        mmd_clusters = [c["codes"] for c in mmd_payload["clusters"]]
        mmd_cluster_info = {
            "strategy": mmd_payload.get("strategy"),
            "cutoff_used": mmd_payload.get("cutoff_used"),
        }
        mmd_cluster_labels = _cluster_label_strings_from_json(
            mmd_payload, mmd_clusters, code_labels,
        )
        log.info(f"\n--- MMD-merge strategy ({pose}) ---")
        _run_one_pass(
            out_dir=dyn_base / pose,
            pose=pose,
            clusters=mmd_clusters,
            cluster_info=mmd_cluster_info,
            cluster_label_strings=mmd_cluster_labels,
            rollouts=rollouts,
            usage=usage,
            pca_components=pca_components,
            dsa_rank=dsa_rank,
            dsa_delays=dsa_delays,
            rslds_D=rslds_D,
            rslds_iters=rslds_iters,
            seed=seed,
            reuse_rslds=reuse,
            title_suffix="MMD merge",
        )

        # --- Strategy 2: label-majority clusters ---
        lbl_path = merged_dir / f"clusters_by_label_{pose}.json"
        if not lbl_path.exists():
            log.warning(
                f"  Skipping label-merge pass: {lbl_path} not found "
                "(re-run rollouts to materialize)."
            )
            continue
        with lbl_path.open() as f:
            lbl_payload = json.load(f)
        lbl_clusters = [c["codes"] for c in lbl_payload["clusters"]]
        lbl_cluster_info = {
            "strategy": lbl_payload.get("strategy"),
            "cutoff_used": lbl_payload.get("cutoff_used"),
        }
        lbl_cluster_labels = _cluster_label_strings_from_json(
            lbl_payload, lbl_clusters, code_labels,
        )
        log.info(f"\n--- Label-merge strategy ({pose}) ---")
        _run_one_pass(
            out_dir=dyn_base / f"{pose}_label",
            pose=pose,
            clusters=lbl_clusters,
            cluster_info=lbl_cluster_info,
            cluster_label_strings=lbl_cluster_labels,
            rollouts=rollouts,
            usage=usage,
            pca_components=pca_components,
            dsa_rank=dsa_rank,
            dsa_delays=dsa_delays,
            rslds_D=rslds_D,
            rslds_iters=rslds_iters,
            seed=seed,
            reuse_rslds=reuse,
            title_suffix="label merge",
        )

    log.info(f"\n=== Done. Dynamics artifacts under {dyn_base}/ ===")


if __name__ == "__main__":
    main()
