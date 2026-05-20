"""Experiment: label-free code distinctness (P1 + P2 + P3).

Post-processes a ``run_code_grid_rollouts`` output directory plus the
tokenizer codes npz to produce three label-free figures arguing that the
code basis is a *finer, reproducible* partition of motion space than the
MMD merge or the 4-class behavior ethogram:

* **P1** — within-code vs between-code MMD vs a code-shuffled null. Within-
  code MMD splits a code's K bodies into two halves and measures their
  MMD (a reproducibility "noise floor"); between-code MMD is the off-
  diagonal of the precomputed distance matrix; the null re-groups all
  rollouts into random pseudo-codes. Within ≪ between ≈ null ⇒ codes are
  reproducible primitives, not chaotic divergence.
* **P2** — behavioral-phylogeny dendrogram + reordered MMD heatmap, with
  the within-code noise floor drawn on the dendrogram. Merges far above
  the floor are real distinctions the MMD/label merge coarsens away.
* **P3** — NMI(code ; 4-class ethogram), effective code count, and a
  behavior→code spread matrix. Partial NMI with effective codes ≫ 4 is
  the "finer than labels" evidence.

Companion to ``run_code_grid_rollouts`` / ``run_code_grid_dynamics``.

Usage:
    cd code2act
    python -m c2a.experiments.run_code_distinctness
"""

import json
import logging
from pathlib import Path

import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from c2a.codegen.shared.label_alignment import (  # noqa: E402
    BEHAVIORS, UNLABELED, load_label_frames,
)
from c2a.experiments.shared.code_grid import mmd_squared  # noqa: E402
from c2a.paths import get_repo_root  # noqa: E402

log = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
MAX_POOL = 1500  # cap pooled qpos vectors per MMD argument, for speed


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _pool(bodies: list[np.ndarray], strip_root: bool, rng) -> np.ndarray:
    """Pool a list of ``(T, nq)`` trajectories into ``(<=MAX_POOL, feat)``."""
    flat = np.concatenate(
        [b.reshape(b.shape[0], -1) for b in bodies], axis=0,
    ).astype(np.float32)
    if strip_root and flat.shape[1] > 3:
        flat = flat[:, 3:]
    if flat.shape[0] > MAX_POOL:
        flat = flat[rng.choice(flat.shape[0], MAX_POOL, replace=False)]
    return flat


def _split_half_mmd(
    bodies: list[np.ndarray], sigma: float, strip_root: bool, rng,
) -> float:
    """MMD between the two halves of a list of K body trajectories."""
    half = len(bodies) // 2
    if half < 1 or len(bodies) - half < 1:
        return float("nan")
    a = _pool(bodies[:half], strip_root, rng)
    b = _pool(bodies[half:], strip_root, rng)
    return float(np.sqrt(max(0.0, mmd_squared(a, b, sigma))))


# ---------------------------------------------------------------------------
# P1 — within / between / null
# ---------------------------------------------------------------------------


def _p1_distances(
    qpos_per_code: dict[int, np.ndarray],
    D: np.ndarray,
    code_ids: list[int],
    sigma: float,
    strip_root: bool,
    n_null: int,
    rng,
) -> dict[str, np.ndarray]:
    """Return within-code, between-code, and shuffled-null MMD arrays."""
    # within-code: split each code's K bodies in half.
    within = []
    for c in code_ids:
        q = qpos_per_code[c]  # (K, T, nq)
        bodies = [q[k] for k in range(q.shape[0])]
        within.append(_split_half_mmd(bodies, sigma, strip_root, rng))
    within = np.asarray(within, dtype=np.float32)

    # between-code: off-diagonal upper triangle of the precomputed matrix.
    iu = np.triu_indices(D.shape[0], k=1)
    between = D[iu].astype(np.float32)

    # shuffled null: pool every (code, body) trajectory, then repeatedly
    # draw a random pseudo-code of K bodies and split-half MMD it.
    all_bodies: list[np.ndarray] = []
    K = None
    for c in code_ids:
        q = qpos_per_code[c]
        K = q.shape[0]
        all_bodies.extend(q[k] for k in range(q.shape[0]))
    null = []
    for _ in range(n_null):
        idx = rng.choice(len(all_bodies), size=K, replace=False)
        grp = [all_bodies[i] for i in idx]
        null.append(_split_half_mmd(grp, sigma, strip_root, rng))
    null = np.asarray(null, dtype=np.float32)

    return {"within": within, "between": between, "null": null}


def _plot_p1(dists: dict[str, np.ndarray], pose: str, path: Path) -> dict:
    within = dists["within"][~np.isnan(dists["within"])]
    between = dists["between"]
    null = dists["null"][~np.isnan(dists["null"])]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = [within, null, between]
    labels = [
        f"within-code\n(n={within.size})",
        f"shuffled null\n(n={null.size})",
        f"between-code\n(n={between.size})",
    ]
    colors = ["#2c7fb8", "#888888", "#d95f0e"]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("MMD (posture, root-stripped)")
    ax.set_title(
        f"P1 - {pose}: code reproducibility vs distinctness\n"
        f"within << between and within << null  =>  reproducible codes"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "within_median": float(np.median(within)),
        "between_median": float(np.median(between)),
        "null_median": float(np.median(null)),
        "within_over_between": float(
            np.median(within) / max(np.median(between), 1e-9)
        ),
        "within_over_null": float(
            np.median(within) / max(np.median(null), 1e-9)
        ),
    }


# ---------------------------------------------------------------------------
# P2 — behavioral phylogeny
# ---------------------------------------------------------------------------


def _plot_p2(
    D: np.ndarray, code_ids: list[int], floor: float, pose: str, path: Path,
) -> None:
    """Dendrogram + reordered MMD heatmap, with the within-code floor line."""
    cond = squareform(D, checks=False)
    Z = hierarchy.linkage(cond, method="average")  # UPGMA

    fig, (ax_d, ax_h) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    dend = hierarchy.dendrogram(
        Z, labels=[str(c) for c in code_ids], ax=ax_d,
        color_threshold=0, above_threshold_color="#444444",
    )
    ax_d.axhline(
        floor, color="#d95f0e", ls="--", lw=1.6,
        label=f"within-code noise floor ({floor:.3f})",
    )
    ax_d.set_title(
        f"P2 — {pose}: behavioral phylogeny\n"
        f"merges far above the floor = real distinctions"
    )
    ax_d.set_xlabel("code id")
    ax_d.set_ylabel("MMD merge height")
    ax_d.legend(loc="upper right", fontsize=8)

    order = dend["leaves"]
    D_re = D[np.ix_(order, order)]
    im = ax_h.imshow(D_re, cmap="magma", aspect="auto")
    ax_h.set_xticks(range(len(order)))
    ax_h.set_yticks(range(len(order)))
    ax_h.set_xticklabels([code_ids[i] for i in order], fontsize=6, rotation=90)
    ax_h.set_yticklabels([code_ids[i] for i in order], fontsize=6)
    ax_h.set_title(f"{pose}: MMD matrix (dendrogram-ordered)")
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="MMD")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# P3 — NMI vs ethogram
# ---------------------------------------------------------------------------


def _nmi(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Normalized mutual information I(x;y)/sqrt(H(x)H(y)), plus MI, H(x), H(y)."""
    cx = np.unique(x, return_inverse=True)[1]
    cy = np.unique(y, return_inverse=True)[1]
    n = cx.size
    joint = np.zeros((cx.max() + 1, cy.max() + 1), dtype=np.float64)
    np.add.at(joint, (cx, cy), 1.0)
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    outer = px[:, None] * py[None, :]
    mi = float((joint[nz] * np.log(joint[nz] / outer[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    nmi = mi / np.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0
    return nmi, mi, hx, hy


def _p3(cfg: DictConfig, out_dir: Path, rng) -> dict:
    codes_npz = np.load(_abs(cfg.codes_path), allow_pickle=True)
    num_codes = int(codes_npz["num_states"])
    clip_len = int(codes_npz["train_codes"].shape[1])

    behavior = load_label_frames(
        str(_abs(cfg.data.labels_path)),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    ).behavior  # (N_clips, clip_len)

    code_flat, beh_flat = [], []
    for codes_key, idx_key in [
        ("train_codes", "train_indices"), ("test_codes", "test_indices"),
    ]:
        codes_arr = codes_npz[codes_key]
        idx_arr = codes_npz[idx_key]
        for row, clip in zip(codes_arr, idx_arr):
            clip = int(clip)
            if clip >= behavior.shape[0]:
                continue
            code_flat.append(np.asarray(row, dtype=np.int64))
            beh_flat.append(behavior[clip].astype(np.int64))
    code_flat = np.concatenate(code_flat)
    beh_flat = np.concatenate(beh_flat)

    mask = (beh_flat != UNLABELED) & (code_flat >= 0)
    code_lab = code_flat[mask]
    beh_lab = beh_flat[mask]

    nmi, mi, h_code, h_beh = _nmi(code_lab, beh_lab)
    # shuffled-code null: NMI should collapse toward 0.
    null_nmi = []
    for _ in range(int(cfg.get("n_null_samples", 200))):
        null_nmi.append(_nmi(rng.permutation(code_lab), beh_lab)[0])
    null_nmi = np.asarray(null_nmi)

    # effective code count = perplexity of the code marginal.
    counts = np.bincount(code_flat[code_flat >= 0], minlength=num_codes)
    p = counts / counts.sum()
    h_usage = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    eff_codes = float(np.exp(h_usage))

    # behavior -> code spread matrix (column-normalized: per-behavior dist).
    n_beh = len(BEHAVIORS)
    spread = np.zeros((n_beh, num_codes), dtype=np.float64)
    np.add.at(spread, (beh_lab, code_lab), 1.0)
    col = spread.sum(axis=1, keepdims=True)
    spread_norm = spread / np.maximum(col, 1e-9)

    fig, (ax_t, ax_h) = plt.subplots(
        2, 1, figsize=(11, 5.6),
        gridspec_kw={"height_ratios": [1, 3.2]},
    )
    ax_t.axis("off")
    ax_t.text(
        0.0, 0.5,
        f"P3 - codes are finer than the ethogram\n"
        f"NMI(code ; 4-class behavior) = {nmi:.3f}   "
        f"(shuffled-code null = {null_nmi.mean():.3f} +/- {null_nmi.std():.3f})\n"
        f"effective code count (perplexity) = {eff_codes:.1f}  "
        f"vs {n_beh} behavior classes\n"
        f"partial NMI with effective codes >> {n_beh}  =>  codes resolve "
        f"sub-structure the ethogram cannot",
        va="center", ha="left", fontsize=11, family="monospace",
    )
    im = ax_h.imshow(spread_norm, cmap="viridis", aspect="auto")
    ax_h.set_yticks(range(n_beh))
    ax_h.set_yticklabels(BEHAVIORS)
    ax_h.set_xlabel("code id")
    ax_h.set_ylabel("behavior")
    ax_h.set_title("each behavior's frame mass spreads across many codes")
    fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02, label="P(code | behavior)")
    fig.tight_layout()
    fig.savefig(out_dir / "P3_nmi_spread.png", dpi=150)
    plt.close(fig)

    return {
        "nmi_code_ethogram": float(nmi),
        "nmi_shuffled_null_mean": float(null_nmi.mean()),
        "nmi_shuffled_null_std": float(null_nmi.std()),
        "mutual_information_nats": float(mi),
        "entropy_code_nats": float(h_code),
        "entropy_ethogram_nats": float(h_beh),
        "effective_code_count": eff_codes,
        "num_behavior_classes": n_beh,
        "num_codes": num_codes,
        "n_labeled_frames": int(code_lab.size),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_distinctness_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Distinctness experiment (P1 + P2 + P3) ===")
    rng = np.random.RandomState(int(cfg.get("seed", 0)))

    out_base = _abs(cfg.output.base_dir) / str(cfg.output.run_name)
    data_dir = out_base / "data"
    merged_dir = out_base / "merged"
    dist_dir = out_base / "distinctness"
    dist_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"No rollout data dir at {data_dir} — run run_code_grid_rollouts first."
        )

    strip_root = bool(cfg.get("strip_root_pos", True))
    n_null = int(cfg.get("n_null_samples", 200))
    summary: dict = {"run": str(out_base), "poses": {}}

    for pose in list(cfg.poses):
        dist_path = merged_dir / f"distance_{pose}.npz"
        if not dist_path.exists():
            log.warning(f"[{pose}] no {dist_path.name}; skipping P1/P2.")
            continue
        d = np.load(dist_path, allow_pickle=True)
        D = np.asarray(d["distance"], dtype=np.float32)
        code_ids = [int(c) for c in d["code_ids"].tolist()]
        sigma = float(d["sigma"])

        qpos_per_code: dict[int, np.ndarray] = {}
        missing = []
        for c in code_ids:
            f = data_dir / f"rollout_{pose}_code{c:03d}.npz"
            if not f.exists():
                missing.append(c)
                continue
            qpos_per_code[c] = np.load(f)["qpos"]
        if missing:
            log.warning(
                f"[{pose}] missing {len(missing)} rollout npz(s): {missing[:8]}"
            )
        code_ids = [c for c in code_ids if c in qpos_per_code]
        if len(code_ids) < 2:
            log.warning(f"[{pose}] <2 codes with rollouts; skipping.")
            continue
        # Restrict D to codes that actually have rollouts.
        keep = [i for i, c in enumerate(d["code_ids"].tolist())
                if int(c) in qpos_per_code]
        D = D[np.ix_(keep, keep)]

        log.info(f"[{pose}] P1: {len(code_ids)} codes, sigma={sigma:.4g}")
        dists = _p1_distances(
            qpos_per_code, D, code_ids, sigma, strip_root, n_null, rng,
        )
        p1 = _plot_p1(dists, pose, dist_dir / f"P1_within_between_{pose}.png")

        floor = float(np.nanmedian(dists["within"]))
        _plot_p2(D, code_ids, floor, pose, dist_dir / f"P2_phylogeny_{pose}.png")
        log.info(
            f"[{pose}] within/between={p1['within_over_between']:.3f} "
            f"within/null={p1['within_over_null']:.3f}"
        )
        summary["poses"][pose] = p1

    log.info("P3: NMI(code ; ethogram) ...")
    summary["P3"] = _p3(cfg, dist_dir, rng)
    log.info(
        f"P3: NMI={summary['P3']['nmi_code_ethogram']:.3f} "
        f"(null {summary['P3']['nmi_shuffled_null_mean']:.3f}), "
        f"effective codes={summary['P3']['effective_code_count']:.1f}"
    )

    with (dist_dir / "distinctness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"=== Done. Figures + summary under {dist_dir}/ ===")


if __name__ == "__main__":
    main()
