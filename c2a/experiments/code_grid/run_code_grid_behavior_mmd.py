"""Experiment: per-cluster MMD/KID vs labeled GT behaviors.

For each merged cluster from ``run_code_grid_rollouts`` (both MMD-merged
and label-merged), compute MMD² (RBF) and KID (polynomial-kernel
MMD²) between the cluster's pooled rollout qpos and each canonical GT
behavior class (Immobile, Rear, Walk_L/R/S, Turn_L/R/S) extracted from
``rodent_labels.csv`` × ``rodent_reference_clips.h5``.

Both strategies are tested:

* **mmd_merge** — clusters grouped by qpos-MMD on the rollouts. The
  cluster's label is the majority of its codes' rollout-derived labels
  (TopoMimic auto-labeler over the generated motion). The MMD-vs-GT
  comparison is non-tautological now because the labels come from a
  separate signal (behavior heuristics) than the merge criterion
  (low-level qpos distance).
* **label_merge** — clusters grouped by codes sharing the same rollout-
  derived label. Tests whether codes that the labeler called "Walk_S"
  actually behave like real Walk_S segments under MMD.

Maps the TopoVNL ``topo_mimic.analysis.feature_analysis.mmd_kid`` flow
onto the code-grid context.

Outputs under ``output.base_dir/<run_name>/{mmd_merge,label_merge}/{walk,rear}/``:
    results.csv          — long-form: one row per (cluster, GT class)
    heatmap_mmd_rbf.png  — clusters × GT classes, RBF-MMD heat
    heatmap_kid.png      — same but KID
    best_match.csv       — per cluster: argmin RBF-MMD GT class + score
    matrices.npz         — raw matrices for downstream tooling
    gt_class_sizes.json  — N per GT class actually used

Usage:
    cd code2act
    python -m c2a.experiments.run_code_grid_behavior_mmd
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import json
import logging
from pathlib import Path
from typing import Any

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig


# Nature-style rcParams: sans-serif, modest line widths, sober ticks.
# Applied module-wide so heatmaps + bars share the look.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 9,
    "axes.titleweight": "regular",
    "axes.labelsize": 8,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Nature-style colormaps + accent colors. MMD uses cool sequential (blues),
# KID uses warm sequential (reds), so the two metrics are visually
# distinct without being garish. Bars use muted Nature blue with a tomato
# accent for the argmin highlight (single-color papers often use
# black + one accent, this keeps the convention).
NATURE_CMAP_MMD = "Blues"
NATURE_CMAP_KID = "Reds"
NATURE_BAR_COLOR = "#3b6eb5"        # muted Nature blue
NATURE_BAR_HIGHLIGHT = "#d8553a"    # warm accent for argmin

from c2a.codegen.shared.label_alignment import (
    BEHAVIORS,
    DIRECTIONS,
    load_label_frames,
)
from c2a.experiments.shared.behavior_mmd import (
    cluster_windows_from_rollouts,
    compute_mmd_kid,
    extract_gt_segments,
    group_gt_segments,
    gt_label_string,
    gt_windows_from_segments,
)
from c2a.paths import get_repo_root

REPO_ROOT = get_repo_root()
log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _canonical_gt_classes() -> list[str]:
    """All 12 canonical (behavior × direction) labels (4 behaviors × 3
    directions). Some cells are empty in the corpus (e.g. Immobile is
    always Straight in practice, Walk_L/R are absent), and will simply
    show n=0 in the GT side of the heatmap."""
    return [f"{b}_{d[0]}" for b in BEHAVIORS for d in DIRECTIONS]


def _load_clusters(path: Path) -> tuple[list[list[int]], list[str]]:
    """Return ``(clusters, labels)`` where ``labels[i]`` is the cluster's
    pre-computed majority label (added by run_code_grid_rollouts)."""
    with path.open() as f:
        payload = json.load(f)
    clusters = [c["codes"] for c in payload["clusters"]]
    labels = [str(c.get("label", "unlabeled")) for c in payload["clusters"]]
    return clusters, labels


def _plot_heatmap(
    M: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    cbar_label: str = "",
) -> None:
    h = max(0.35 * M.shape[0] + 1.2, 3.0)
    w = max(0.55 * M.shape[1] + 1.6, 4.0)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(M, cmap=cmap, aspect="auto", interpolation="none")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)


def _plot_per_cluster_bars(
    M: np.ndarray,
    cluster_labels: list[str],
    cluster_sizes_n_windows: list[int],
    gt_classes: list[str],
    gt_sizes: dict[str, int],
    title_prefix: str,
    metric_name: str,
    out_path: Path,
    bar_color: str = NATURE_BAR_COLOR,
    highlight_color: str = NATURE_BAR_HIGHLIGHT,
) -> None:
    """One subplot per cluster: bar per GT class with the metric value.

    Best-match (argmin among finite values) is highlighted in
    ``highlight_color``. The subplot title carries the cluster's modal
    label so it's easy to spot label↔best-match agreement.
    """
    n_clusters = M.shape[0]
    n_gt = M.shape[1]
    if n_clusters == 0 or n_gt == 0:
        return
    ncols = min(4, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(3.2 * ncols, 6.0), max(2.6 * nrows, 3.0)),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    finite_pool = M[np.isfinite(M)]
    y_max = float(finite_pool.max()) if finite_pool.size else 1.0
    for ci in range(n_clusters):
        ax = axes_flat[ci]
        vals = M[ci]
        finite_mask = np.isfinite(vals)
        if finite_mask.any():
            best_gi = int(np.nanargmin(vals))
        else:
            best_gi = -1
        colors = [
            highlight_color if gi == best_gi else bar_color
            for gi in range(n_gt)
        ]
        x_positions = np.arange(n_gt)
        bar_heights = np.where(finite_mask, vals, 0.0)
        ax.bar(x_positions, bar_heights, color=colors,
               edgecolor="black", linewidth=0.4)
        for gi in range(n_gt):
            if not finite_mask[gi]:
                ax.text(gi, 0.0, "n/a", ha="center", va="bottom",
                        fontsize=6, color="grey")
        ax.set_xticks(x_positions)
        # Tick label: GT class name + n
        xt = [f"{c}\n(n={gt_sizes.get(c, 0)})" for c in gt_classes]
        ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(metric_name, fontsize=7)
        best_str = f"\nbest→{gt_classes[best_gi]}" if best_gi >= 0 else ""
        ax.set_title(
            f"c{ci} [{cluster_labels[ci]}] "
            f"(n={cluster_sizes_n_windows[ci]}){best_str}",
            fontsize=8,
        )
        ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, axis="y", alpha=0.25)
    # Hide unused subplots
    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle(f"{title_prefix} — per-cluster bars", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(str(out_path), dpi=200)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)


def _run_one_pass(
    strategy: str,
    pose: str,
    clusters: list[list[int]],
    cluster_labels: list[str],
    rollouts_data_dir: Path,
    gt_class_segments: dict[str, list],
    n_h5_frames_per_clip: int,
    qpos_h5_path: Path,
    out_dir: Path,
    K: int,
    strip_root_dims: int,
    max_cluster_windows: int | None,
    max_gt_windows: int | None,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    canonical = _canonical_gt_classes()

    # Pre-materialize one GT window stack per canonical class.
    gt_windows: dict[str, np.ndarray] = {}
    gt_sizes_actual: dict[str, int] = {}
    for cls in canonical:
        segs = gt_class_segments.get(cls, [])
        if not segs:
            gt_windows[cls] = np.empty((0, K, 0), dtype=np.float32)
            gt_sizes_actual[cls] = 0
            continue
        W = gt_windows_from_segments(
            segs,
            qpos_h5_path=qpos_h5_path,
            n_frames_per_clip=n_h5_frames_per_clip,
            K=K,
            strip_root_dims=strip_root_dims,
            max_segments=max_gt_windows,
            rng=rng,
        )
        gt_windows[cls] = W
        gt_sizes_actual[cls] = int(W.shape[0])
    with (out_dir / "gt_class_sizes.json").open("w") as f:
        json.dump(gt_sizes_actual, f, indent=2)
    log.info(f"  [{strategy}/{pose}] GT class sizes: {gt_sizes_actual}")

    rows: list[dict[str, Any]] = []
    mmd_M = np.full((len(clusters), len(canonical)), np.nan, dtype=np.float32)
    kid_M = np.full((len(clusters), len(canonical)), np.nan, dtype=np.float32)
    n_cluster_per_row: list[int] = []

    for ci, codes in enumerate(clusters):
        X, n_pre = cluster_windows_from_rollouts(
            codes, rollouts_data_dir, pose=pose,
            K=K, strip_root_dims=strip_root_dims,
            max_windows=max_cluster_windows, rng=rng,
        )
        n_cluster_per_row.append(int(X.shape[0]))
        log.info(
            f"  [{strategy}/{pose}] cluster {ci} [{cluster_labels[ci]}] "
            f"|codes|={len(codes)} → n_windows={X.shape[0]} (pre-subsample {n_pre})"
        )
        if X.shape[0] < 2:
            continue
        for gi, cls in enumerate(canonical):
            Y = gt_windows[cls]
            if Y.shape[0] < 2 or Y.shape[2] != X.shape[2]:
                if Y.shape[2] != X.shape[2] and Y.shape[0] >= 2:
                    log.warning(
                        f"  feature-dim mismatch cluster (D={X.shape[2]}) vs "
                        f"GT '{cls}' (D={Y.shape[2]}); skipping"
                    )
                continue
            out = compute_mmd_kid(X, Y, rng=rng)
            mmd_M[ci, gi] = out["mmd_rbf"]
            kid_M[ci, gi] = out["kid"]
            rows.append({
                "cluster_id": ci,
                "cluster_label": cluster_labels[ci],
                "cluster_size_codes": len(codes),
                "cluster_codes": ",".join(str(int(c)) for c in codes),
                "n_cluster": out["n_cluster"],
                "gt_class": cls,
                "n_gt": out["n_real"],
                "mmd_rbf": float(out["mmd_rbf"]),
                "kid": float(out["kid"]),
                "sigma": float(out["sigma"]),
            })

    # CSV (long form).
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    log.info(f"  [{strategy}/{pose}] wrote {csv_path}")

    # Best-match per cluster (argmin RBF-MMD over GT classes).
    best_rows: list[dict[str, Any]] = []
    for ci in range(len(clusters)):
        finite_mask = np.isfinite(mmd_M[ci])
        if not finite_mask.any():
            best_rows.append({
                "cluster_id": ci,
                "cluster_label": cluster_labels[ci],
                "n_cluster": int(n_cluster_per_row[ci]),
                "best_gt_class": "",
                "best_mmd_rbf": float("nan"),
                "best_kid": float("nan"),
                "majority_label_matches_best": "",
            })
            continue
        gi_best = int(np.nanargmin(mmd_M[ci]))
        best_cls = canonical[gi_best]
        best_rows.append({
            "cluster_id": ci,
            "cluster_label": cluster_labels[ci],
            "n_cluster": int(n_cluster_per_row[ci]),
            "best_gt_class": best_cls,
            "best_mmd_rbf": float(mmd_M[ci, gi_best]),
            "best_kid": float(kid_M[ci, gi_best]),
            "majority_label_matches_best": bool(cluster_labels[ci] == best_cls),
        })
    best_path = out_dir / "best_match.csv"
    with best_path.open("w", newline="") as f:
        if best_rows:
            writer = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_rows)
    log.info(f"  [{strategy}/{pose}] wrote {best_path}")

    # Filter to populated GT classes only (drop the corpus-empty cells
    # like Immobile_L/R, Turn_S, Walk_L/R that have zero segments) for
    # cleaner plots. The full 12-class matrix is still in matrices.npz.
    populated_gi = [
        gi for gi, cls in enumerate(canonical)
        if gt_sizes_actual.get(cls, 0) > 0
    ]
    # Reorder so the column sequence follows the row (cluster) label
    # sequence — the diagonal then visualizes "cluster claims label L,
    # how does it score against the L-GT class?". GT classes that no
    # cluster claims as its modal label are dropped entirely (they have
    # no anchor to interpret) — keeps the heatmap focused.
    populated_classes = [canonical[gi] for gi in populated_gi]
    ordered_classes: list[str] = []
    seen: set[str] = set()
    for lab in cluster_labels:
        if lab in populated_classes and lab not in seen:
            ordered_classes.append(lab)
            seen.add(lab)
    plot_gi = [canonical.index(c) for c in ordered_classes]
    canonical_plot = ordered_classes
    mmd_M_plot = mmd_M[:, plot_gi]
    kid_M_plot = kid_M[:, plot_gi]

    # Heatmaps.
    row_labels = [
        f"c{ci} [{cluster_labels[ci]}] (n={n_cluster_per_row[ci]})"
        for ci in range(len(clusters))
    ]
    col_labels = [
        f"{cls} (n={gt_sizes_actual.get(cls, 0)})" for cls in canonical_plot
    ]
    _plot_heatmap(
        mmd_M_plot, row_labels, col_labels,
        title=f"RBF-MMD² — {strategy} / {pose}",
        out_path=out_dir / "heatmap_mmd_rbf.png",
        cmap=NATURE_CMAP_MMD, cbar_label="MMD² (RBF, σ=median)",
    )
    _plot_heatmap(
        kid_M_plot, row_labels, col_labels,
        title=f"KID — {strategy} / {pose}",
        out_path=out_dir / "heatmap_kid.png",
        cmap=NATURE_CMAP_KID, cbar_label="KID (poly k=3)",
    )

    # Per-cluster bar plots: one subplot per cluster, bars per GT class,
    # best-match (argmin) bar highlighted. Easier to read than the heatmap
    # when comparing a single cluster's profile across GT classes.
    _plot_per_cluster_bars(
        mmd_M_plot,
        cluster_labels=cluster_labels,
        cluster_sizes_n_windows=n_cluster_per_row,
        gt_classes=canonical_plot,
        gt_sizes=gt_sizes_actual,
        title_prefix=f"RBF-MMD² — {strategy} / {pose}",
        metric_name="MMD² (RBF)",
        out_path=out_dir / "bars_mmd_rbf.png",
    )
    _plot_per_cluster_bars(
        kid_M_plot,
        cluster_labels=cluster_labels,
        cluster_sizes_n_windows=n_cluster_per_row,
        gt_classes=canonical_plot,
        gt_sizes=gt_sizes_actual,
        title_prefix=f"KID — {strategy} / {pose}",
        metric_name="KID",
        out_path=out_dir / "bars_kid.png",
    )

    # Save raw matrices for downstream reuse.
    np.savez(
        out_dir / "matrices.npz",
        mmd_rbf=mmd_M,
        kid=kid_M,
        row_cluster_id=np.arange(len(clusters), dtype=np.int32),
        col_gt_class=np.asarray(canonical),
        cluster_labels=np.asarray(cluster_labels),
        n_cluster_per_row=np.asarray(n_cluster_per_row, dtype=np.int32),
    )

    # Summary stats: how often majority label matches argmin GT.
    matches = sum(
        1 for r in best_rows
        if r["majority_label_matches_best"] is True
    )
    total = sum(
        1 for r in best_rows
        if isinstance(r["majority_label_matches_best"], bool)
    )
    if total > 0:
        log.info(
            f"  [{strategy}/{pose}] majority-label matches argmin GT: "
            f"{matches}/{total} = {100*matches/total:.1f}%"
        )


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_grid_behavior_mmd",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Grid Behavior MMD/KID ===")

    rollouts_base = (
        Path(_abs(cfg.rollouts.base_dir)) / str(cfg.rollouts.run_name)
    )
    data_dir = rollouts_base / "data"
    merged_dir = rollouts_base / "merged"
    if not (data_dir.exists() and merged_dir.exists()):
        raise RuntimeError(
            f"Expected rollouts + merged under {rollouts_base}. "
            "Run run_code_grid_rollouts first."
        )

    out_base = Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    out_base.mkdir(parents=True, exist_ok=True)

    qpos_h5_path = Path(_abs(cfg.data.reference_data_path))
    if not qpos_h5_path.exists():
        raise FileNotFoundError(qpos_h5_path)
    n_h5_frames = int(cfg.data.n_h5_frames_per_clip)

    # --- GT segments ---
    log.info("Loading labeled segments from rodent_labels.csv ...")
    label_arrays = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=n_h5_frames,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    segments = extract_gt_segments(
        label_arrays.behavior,
        label_arrays.direction,
        label_arrays.clip_indices,
        min_length=int(cfg.gt.min_segment_length),
    )
    gt_class_segments = group_gt_segments(segments)
    log.info(
        f"  Found {len(segments)} labeled segments across "
        f"{len(gt_class_segments)} GT classes."
    )
    for cls in _canonical_gt_classes():
        n = len(gt_class_segments.get(cls, []))
        log.info(f"    {cls}: {n} segments")

    K = int(cfg.window.K)
    strip_root_dims = int(cfg.window.strip_root_dims)
    max_cluster_windows = cfg.window.get("max_cluster_windows", None)
    if max_cluster_windows is not None:
        max_cluster_windows = int(max_cluster_windows)
    max_gt_windows = cfg.window.get("max_gt_windows", None)
    if max_gt_windows is not None:
        max_gt_windows = int(max_gt_windows)
    seed = int(cfg.seed)

    # Both strategies carry a meaningful per-cluster label hypothesis now
    # that labels come from the TopoMimic auto-labeler over rolled-out
    # qpos (a separate signal than qpos-MMD), so we test both.
    strategy_files = {
        "mmd_merge": "clusters_{pose}.json",
        "label_merge": "clusters_by_label_{pose}.json",
    }
    for strategy, fname_tpl in strategy_files.items():
        for pose in ("walk", "rear"):
            path = merged_dir / fname_tpl.format(pose=pose)
            if not path.exists():
                log.warning(f"  Missing {path}; skipping {strategy}/{pose}")
                continue
            clusters, cluster_labels = _load_clusters(path)
            log.info(
                f"\n--- {strategy} / {pose}: {len(clusters)} clusters "
                f"(sizes {[len(c) for c in clusters[:8]]}...) ---"
            )
            _run_one_pass(
                strategy=strategy,
                pose=pose,
                clusters=clusters,
                cluster_labels=cluster_labels,
                rollouts_data_dir=data_dir,
                gt_class_segments=gt_class_segments,
                n_h5_frames_per_clip=n_h5_frames,
                qpos_h5_path=qpos_h5_path,
                out_dir=out_base / strategy / pose,
                K=K,
                strip_root_dims=strip_root_dims,
                max_cluster_windows=max_cluster_windows,
                max_gt_windows=max_gt_windows,
                seed=seed,
            )

    log.info(f"\n=== Done. Artifacts under {out_base}/ ===")


if __name__ == "__main__":
    main()
