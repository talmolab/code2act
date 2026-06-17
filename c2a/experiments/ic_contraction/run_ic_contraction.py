"""Experiment: initial-condition contraction (the "one-to-many" test).

The paper's boldest mechanistic claim is that distilling the imitation encoder
forces each discrete code into a *one-to-many* mapping: a single held code
realizes the *same* motif from *any* initial condition.  This script tests that
directly by re-using the held-code rollouts already produced by
``run_code_grid_rollouts`` (100 bodies started from diverse poses, the code held
constant for the whole rollout, decoder-only so only the code drives behavior)
and asking whether the bodies *converge* over time.

It is pure post-processing of saved ``.npz`` — no MuJoCo / GPU needed.

For every condition (tokenizer) × space (root-stripped pose; GRU hidden) it
computes, per code:

* ``sigma_inst(t)`` — RMS cross-body dispersion at each timestep (z-scored
  features).  Decreasing ⇒ bodies are collapsing onto a shared trajectory.
  This is phase-*sensitive*: two bodies on the same limit cycle but at
  different phases keep a residual dispersion.
* ``attractor_spread`` — RMS spread of the pooled late-time state cloud, i.e.
  the floor ``sigma_inst`` can reach for a limit cycle (full convergence
  *up to phase*).  ``~0`` for a fixed point, ``>0`` for a cycle.
* ``d_manifold(t)`` — *phase-invariant* distance to the shared late-time motif
  manifold, estimated leave-half-out with PCA + kNN.  Contracts to a noise
  floor whenever bodies reach the shared attractor regardless of phase; this is
  the rigorous convergence signal for cyclic behaviors.
* contraction ratios ``rho = median(late) / median(early)`` for both curves.

A **cross-code null** — bodies pooled across *different* codes (no shared
target) — is the control that should *not* contract.

Codes are colored / aggregated by the rollout-derived behavior label
(``labels/code_labels.json``: Immobile / Rear / Turn / Walk), i.e. what the
decoder *produces* for each code.

Usage::

    cd code2act
    python -m c2a.experiments.behavior.run_ic_contraction
    # subset of conditions / single space:
    python -m c2a.experiments.behavior.run_ic_contraction spaces='[pose]'
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from c2a.codegen.shared.label_alignment import BEHAVIORS  # noqa: E402
from c2a.paths import get_repo_root  # noqa: E402

log = logging.getLogger(__name__)
REPO_ROOT = get_repo_root()

# Behavior -> color (matches run_code_distinctness ordering of BEHAVIORS).
BEHAVIOR_COLORS = {
    "Immobile": "#8d929e",  # slate
    "Rear": "#9e66cc",      # purple
    "Turn": "#f2992a",      # orange
    "Walk": "#4cb46b",      # green
    "Other": "#cccccc",
}
BEHAVIOR_ORDER = list(BEHAVIORS) + ["Other"]


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix.

    Vendored from ``c2a.experiments.fidelity.run_inception_distance`` to keep
    this script import-light (that module pulls in heavy deps at import time).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return r.reshape(q.shape[:-1] + (3, 3))


def pose_features(qpos: np.ndarray) -> np.ndarray:
    """Root-stripped, yaw-removed pose features from qpos ``(..., 74)``.

    Mirrors the KID feature preprocessing: the root quaternion (dims 3:7) is
    replaced by a yaw-removed 6D rotation and the planar (x, y) root position is
    dropped, keeping height ``z``, pitch/roll, and all joint angles.  This makes
    the feature invariant to *where* and in *which heading* a body walks, so
    that two bodies executing the same motif in different directions are scored
    as identical.
    """
    quat = qpos[..., 3:7]
    R = _quat_to_rotmat(quat)
    heading = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    ch, sh = np.cos(-heading), np.sin(-heading)
    R_yaw_inv = np.zeros(R.shape[:-2] + (3, 3), dtype=qpos.dtype)
    R_yaw_inv[..., 0, 0] = ch
    R_yaw_inv[..., 0, 1] = -sh
    R_yaw_inv[..., 1, 0] = sh
    R_yaw_inv[..., 1, 1] = ch
    R_yaw_inv[..., 2, 2] = 1.0
    R_local = np.einsum("...ij,...jk->...ik", R_yaw_inv, R)
    rot_6d = np.concatenate([R_local[..., :, 0], R_local[..., :, 1]], axis=-1)
    z = qpos[..., 2:3]
    joints = qpos[..., 7:]
    return np.concatenate([z, rot_6d, joints], axis=-1)


def extract_features(npz, space: str) -> np.ndarray:
    """Return ``(B, T, F)`` features for the requested space."""
    if space == "pose":
        return pose_features(npz["qpos"].astype(np.float32))
    if space == "hidden":
        return npz["hidden"].astype(np.float32)
    raise ValueError(f"unknown space {space!r}")


# ---------------------------------------------------------------------------
# Condition discovery / loading
# ---------------------------------------------------------------------------


def discover_poses(data_dir: Path) -> list[str]:
    poses = set()
    for f in data_dir.glob("rollout_*_code*.npz"):
        m = re.search(r"rollout_(.+?)_code\d+\.npz", f.name)
        if m:
            poses.add(m.group(1))
    return sorted(poses)


def load_code(
    data_dir: Path, poses: list[str], code_id: int, space: str, max_bodies: int,
) -> np.ndarray | None:
    """Load + pool a code's rollouts across poses → features ``(B, T, F)``."""
    parts = []
    for pose in poses:
        f = data_dir / f"rollout_{pose}_code{code_id:03d}.npz"
        if not f.exists():
            continue
        with np.load(f) as d:
            parts.append(extract_features(d, space))
    if not parts:
        return None
    # Pose rollouts can have differing T (none observed, but be safe): clip.
    T = min(p.shape[1] for p in parts)
    feat = np.concatenate([p[:, :T] for p in parts], axis=0)
    if feat.shape[0] > max_bodies:
        feat = feat[:max_bodies]
    return feat


def load_labels(labels_path: Path) -> dict[int, dict]:
    """code_id -> {behavior:int, label:str, behavior_name:str}."""
    if not labels_path.exists():
        return {}
    raw = json.loads(labels_path.read_text())
    out = {}
    for cid, rec in raw.get("codes", {}).items():
        bi = int(rec.get("behavior", -1))
        name = BEHAVIORS[bi] if 0 <= bi < len(BEHAVIORS) else "Other"
        out[int(cid)] = {
            "behavior": bi,
            "behavior_name": name,
            "label": rec.get("label", str(cid)),
        }
    return out


def condition_code_ids(base: Path, poses: list[str]) -> list[int]:
    """Code ids that have at least one rollout file, sorted by usage if known."""
    ids = set()
    for f in (base / "data").glob("rollout_*_code*.npz"):
        m = re.search(r"_code(\d+)\.npz", f.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Z-score statistics (streaming, mmap, strided)
# ---------------------------------------------------------------------------


def zscore_stats(
    data_dir: Path, poses: list[str], code_ids: list[int], space: str,
    stride: int, max_bodies: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-dim (mean, std) over a strided subsample of all codes' frames."""
    n = 0
    s = None
    s2 = None
    for cid in code_ids:
        for pose in poses:
            f = data_dir / f"rollout_{pose}_code{cid:03d}.npz"
            if not f.exists():
                continue
            with np.load(f, mmap_mode="r") as d:
                key = "qpos" if space == "pose" else "hidden"
                raw = np.asarray(d[key][:max_bodies, ::stride])
            feat = (
                pose_features(raw.astype(np.float32))
                if space == "pose"
                else raw.astype(np.float32)
            )
            flat = feat.reshape(-1, feat.shape[-1])
            if s is None:
                s = flat.sum(0, dtype=np.float64)
                s2 = (flat.astype(np.float64) ** 2).sum(0)
            else:
                s += flat.sum(0, dtype=np.float64)
                s2 += (flat.astype(np.float64) ** 2).sum(0)
            n += flat.shape[0]
    mean = s / max(n, 1)
    var = np.maximum(s2 / max(n, 1) - mean**2, 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def sigma_inst(feat: np.ndarray) -> np.ndarray:
    """RMS-over-dims cross-body std at each timestep → ``(T,)``."""
    var_b = feat.var(axis=0)              # (T, F)
    return np.sqrt(var_b.mean(axis=1))    # (T,)


def attractor_spread(feat: np.ndarray, late_idx: np.ndarray) -> float:
    """RMS spread of the pooled late-time state cloud (phase-desync floor)."""
    pooled = feat[:, late_idx].reshape(-1, feat.shape[-1])
    return float(np.sqrt(pooled.var(axis=0).mean()))


def manifold_distance(
    feat: np.ndarray, late_idx: np.ndarray, grid: np.ndarray,
    pca_dim: int, knn: int, subsample: int, rng: np.random.RandomState,
) -> np.ndarray:
    """Phase-invariant distance-to-shared-motif over a time grid → ``(len(grid),)``.

    Leave-half-out: build the motif manifold from one half's late-window states,
    measure the other half's distance to it at every grid time, and symmetrize.
    """
    B = feat.shape[0]
    if B < 4:
        return np.full(grid.shape[0], np.nan, dtype=np.float32)
    perm = rng.permutation(B)
    halves = [perm[: B // 2], perm[B // 2:]]
    curves = []
    for ref, qry in [(halves[0], halves[1]), (halves[1], halves[0])]:
        M = feat[ref][:, late_idx].reshape(-1, feat.shape[-1])
        if M.shape[0] > subsample:
            M = M[rng.choice(M.shape[0], subsample, replace=False)]
        dim = min(pca_dim, M.shape[1], M.shape[0] - 1)
        pca = PCA(n_components=dim).fit(M)
        Mp = pca.transform(M)
        nn = NearestNeighbors(n_neighbors=min(knn, Mp.shape[0])).fit(Mp)
        d_t = []
        for t in grid:
            Q = pca.transform(feat[qry][:, t])
            dist, _ = nn.kneighbors(Q)
            d_t.append(float(dist.mean()))
        curves.append(np.asarray(d_t, dtype=np.float32))
    return np.mean(curves, axis=0)


def ratio(curve: np.ndarray, early: np.ndarray, late: np.ndarray) -> float:
    e = np.nanmedian(curve[early])
    l = np.nanmedian(curve[late])
    return float(l / e) if e > 1e-9 else float("nan")


# ---------------------------------------------------------------------------
# Per-condition analysis
# ---------------------------------------------------------------------------


def analyze_condition(
    cond: dict, cfg: DictConfig, rng: np.random.RandomState,
) -> dict | None:
    base = _abs(cfg.rollouts_base) / cond["name"]
    data_dir = base / "data"
    if not data_dir.is_dir():
        log.warning(f"[{cond['name']}] no data dir; skipping.")
        return None
    poses = list(cfg.poses) if cfg.poses else discover_poses(data_dir)
    code_ids = condition_code_ids(base, poses)
    labels = load_labels(base / "labels" / "code_labels.json")
    a = cfg.analysis
    max_bodies = int(a.max_bodies)

    result = {
        "name": cond["name"],
        "display": cond.get("display", cond["name"]),
        "group": cond.get("group", ""),
        "poses": poses,
        "code_ids": code_ids,
        "spaces": {},
    }
    log.info(
        f"[{cond['name']}] poses={poses} codes={len(code_ids)} "
        f"labels={len(labels)}"
    )

    for space in list(cfg.spaces):
        mean, std = zscore_stats(
            data_dir, poses, code_ids, space,
            int(a.zscore_stride), max_bodies,
        )
        per_code = {}
        sig_curves, mfd_curves = [], []
        null_pool = []  # one body per code for the cross-code null
        T_ref = None
        early = late = grid = None

        for cid in code_ids:
            feat = load_code(data_dir, poses, cid, space, max_bodies)
            if feat is None or feat.shape[0] < 2:
                continue
            feat = (feat - mean) / std
            T = feat.shape[1]
            if T_ref is None:
                T_ref = T
                n_e = max(2, int(round(a.early_frac * T)))
                n_l = max(2, int(round(a.late_frac * T)))
                early = np.arange(0, n_e)
                late = np.arange(T - n_l, T)
                grid = np.linspace(
                    0, T - 1, int(a.manifold_time_points)
                ).astype(int)
            sig = sigma_inst(feat)
            spread = attractor_spread(feat, late)
            mfd = manifold_distance(
                feat, late, grid, int(a.manifold_pca_dim),
                int(a.manifold_knn), int(a.manifold_subsample), rng,
            )
            grid_early = np.arange(0, max(2, int(round(a.early_frac * len(grid)))))
            grid_late = np.arange(
                len(grid) - max(2, int(round(a.late_frac * len(grid)))), len(grid)
            )
            lab = labels.get(cid, {"behavior_name": "Other", "label": str(cid)})
            per_code[cid] = {
                "behavior": lab["behavior_name"],
                "label": lab["label"],
                "sigma_inst": sig,
                "sigma_norm": sig / max(np.median(sig[early]), 1e-9),
                "attractor_spread": spread,
                "attractor_spread_norm": spread / max(np.median(sig[early]), 1e-9),
                "d_manifold": mfd,
                "d_manifold_norm": mfd / max(np.nanmedian(mfd[grid_early]), 1e-9),
                "rho_inst": ratio(sig, early, late),
                "rho_manifold": ratio(mfd, grid_early, grid_late),
            }
            sig_curves.append(per_code[cid]["sigma_norm"])
            mfd_curves.append(per_code[cid]["d_manifold_norm"])
            null_pool.append(feat[rng.randint(feat.shape[0])])  # 1 body
            del feat

        if not per_code:
            log.warning(f"[{cond['name']}/{space}] no codes; skipping.")
            continue

        # Cross-code null: dispersion of one-body-per-code pooled together.
        null_arr = np.stack(null_pool, axis=0)  # (n_codes, T, F)
        null_sig = sigma_inst(null_arr)
        null_norm = null_sig / max(np.median(null_sig[early]), 1e-9)
        null_rho = ratio(null_sig, early, late)

        result["spaces"][space] = {
            "per_code": per_code,
            "T": int(T_ref),
            "early": early,
            "late": late,
            "grid": grid,
            "null_sigma_norm": null_norm,
            "null_rho": null_rho,
        }
        med_rho_i = np.nanmedian([v["rho_inst"] for v in per_code.values()])
        med_rho_m = np.nanmedian([v["rho_manifold"] for v in per_code.values()])
        log.info(
            f"[{cond['name']}/{space}] median rho_inst={med_rho_i:.3f} "
            f"rho_manifold={med_rho_m:.3f} null_rho={null_rho:.3f} "
            f"({len(per_code)} codes)"
        )
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _beh_color(name: str) -> str:
    return BEHAVIOR_COLORS.get(name, BEHAVIOR_COLORS["Other"])


def plot_condition_curves(res: dict, space: str, out: Path) -> None:
    """σ_inst(t) + d_manifold(t), behavior-colored, with group means + null."""
    sp = res["spaces"][space]
    pc = sp["per_code"]
    T = sp["T"]
    grid = sp["grid"]
    t_ax = np.arange(T) / (T - 1)
    g_ax = grid / (T - 1)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    by_beh: dict[str, list] = {}
    for cid, v in pc.items():
        c = _beh_color(v["behavior"])
        axA.plot(t_ax, v["sigma_norm"], color=c, alpha=0.25, lw=0.8)
        axB.plot(g_ax, v["d_manifold_norm"], color=c, alpha=0.25, lw=0.8)
        by_beh.setdefault(v["behavior"], []).append(cid)

    for beh, cids in by_beh.items():
        c = _beh_color(beh)
        sig = np.nanmean([pc[i]["sigma_norm"] for i in cids], axis=0)
        mfd = np.nanmean([pc[i]["d_manifold_norm"] for i in cids], axis=0)
        axA.plot(t_ax, sig, color=c, lw=2.6, label=f"{beh} (n={len(cids)})")
        axB.plot(g_ax, mfd, color=c, lw=2.6, label=f"{beh} (n={len(cids)})")

    axA.plot(t_ax, sp["null_sigma_norm"], color="k", ls="--", lw=1.8,
             label="cross-code null")
    axA.axhline(1.0, color="grey", lw=0.8, ls=":")
    axA.set_title(f"{res['display']} — instantaneous dispersion ({space})")
    axA.set_xlabel("rollout fraction")
    axA.set_ylabel(r"$\sigma_{\mathrm{inst}}(t)\,/\,\sigma_{\mathrm{inst}}(0)$")
    axA.set_ylim(0, max(1.3, axA.get_ylim()[1]))
    axA.legend(fontsize=7, loc="upper right")

    axB.axhline(1.0, color="grey", lw=0.8, ls=":")
    axB.set_title(f"phase-invariant convergence to motif ({space})")
    axB.set_xlabel("rollout fraction")
    axB.set_ylabel(r"$d_{\mathrm{manifold}}(t)\,/\,d(0)$")
    axB.set_ylim(0, max(1.3, axB.get_ylim()[1]))
    axB.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Bodies started from diverse poses converge to a shared per-code motif "
        "(curves below 1 = contraction)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_condition_summary(res: dict, out: Path) -> None:
    """Per-code contraction ratios, grouped/colored by behavior, pose+hidden."""
    spaces = [s for s in ("pose", "hidden") if s in res["spaces"]]
    fig, axes = plt.subplots(
        1, len(spaces), figsize=(6.2 * len(spaces), 4.4), squeeze=False,
    )
    for ax, space in zip(axes[0], spaces):
        pc = res["spaces"][space]["per_code"]
        rows = []
        for cid, v in pc.items():
            rows.append((v["behavior"], cid, v["rho_inst"], v["rho_manifold"],
                         v["attractor_spread_norm"]))
        rows.sort(key=lambda r: (BEHAVIOR_ORDER.index(r[0])
                                 if r[0] in BEHAVIOR_ORDER else 99, r[3]))
        xs = np.arange(len(rows))
        for i, (beh, cid, ri, rm, asn) in enumerate(rows):
            c = _beh_color(beh)
            ax.scatter(i, rm, color=c, s=44, zorder=3, edgecolor="k", lw=0.4)
            ax.scatter(i, ri, color=c, s=22, marker="x", zorder=2)
        ax.axhline(1.0, color="grey", ls=":", lw=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(r[1]) for r in rows], rotation=90, fontsize=6)
        ax.set_ylabel("contraction ratio (late / early)")
        ax.set_xlabel("code id")
        ax.set_title(f"{res['display']} — {space}\n"
                     "● phase-invariant  ✕ instantaneous")
        ax.set_ylim(0, max(1.2, ax.get_ylim()[1]))
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=_beh_color(b),
                              label=b, markeredgecolor="k")
                   for b in BEHAVIOR_ORDER if any(r[0] == b for r in rows)]
        ax.legend(handles=handles, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_overview(results: list[dict], space: str, out: Path) -> None:
    """Small-multiples: behavior-group mean σ_inst(t) per condition."""
    res = [r for r in results if space in r["spaces"]]
    n = len(res)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, r in enumerate(res):
        ax = axes[k // ncol][k % ncol]
        sp = r["spaces"][space]
        pc = sp["per_code"]
        T = sp["T"]
        t_ax = np.arange(T) / (T - 1)
        by_beh: dict[str, list] = {}
        for cid, v in pc.items():
            by_beh.setdefault(v["behavior"], []).append(cid)
        for beh, cids in by_beh.items():
            sig = np.nanmean([pc[i]["sigma_norm"] for i in cids], axis=0)
            ax.plot(t_ax, sig, color=_beh_color(beh), lw=2.0,
                    label=f"{beh}")
        ax.plot(t_ax, sp["null_sigma_norm"], color="k", ls="--", lw=1.4)
        ax.axhline(1.0, color="grey", lw=0.7, ls=":")
        ax.set_title(r["display"], fontsize=9)
        ax.set_ylim(0, 1.35)
        ax.set_xlabel("rollout fraction")
        if k % ncol == 0:
            ax.set_ylabel(r"$\sigma_{\mathrm{inst}}/\sigma_0$")
        ax.legend(fontsize=6, loc="upper right")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"Initial-condition contraction across tokenizers ({space} space) — "
        "dashed = cross-code null", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ratio_matrix(results: list[dict], space: str, out: Path) -> None:
    """Heatmap: condition × behavior median phase-invariant contraction ratio."""
    res = [r for r in results if space in r["spaces"]]
    behs = [b for b in BEHAVIOR_ORDER if b != "Other"]
    M = np.full((len(res), len(behs)), np.nan)
    for i, r in enumerate(res):
        pc = r["spaces"][space]["per_code"]
        for j, beh in enumerate(behs):
            vals = [v["rho_manifold"] for v in pc.values()
                    if v["behavior"] == beh]
            if vals:
                M[i, j] = np.nanmedian(vals)
    fig, ax = plt.subplots(figsize=(1.3 * len(behs) + 3, 0.6 * len(res) + 2.5))
    im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=1.1, aspect="auto")
    ax.set_xticks(range(len(behs)))
    ax.set_xticklabels(behs, rotation=30, ha="right")
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels([r["display"] for r in res], fontsize=8)
    for i in range(len(res)):
        for j in range(len(behs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.7 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="median phase-invariant ratio (late/early)")
    ax.set_title(f"Contraction by tokenizer × behavior ({space})\n"
                 "lower = stronger one-to-many convergence")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_arrays(res: dict, out_dir: Path) -> None:
    for space, sp in res["spaces"].items():
        payload = {
            "T": sp["T"], "grid": sp["grid"],
            "null_sigma_norm": sp["null_sigma_norm"],
            "null_rho": sp["null_rho"],
        }
        for cid, v in sp["per_code"].items():
            payload[f"sigma_norm_{cid}"] = v["sigma_norm"]
            payload[f"d_manifold_norm_{cid}"] = v["d_manifold_norm"]
        np.savez(out_dir / f"contraction_{res['name']}_{space}.npz", **payload)


def summarize(results: list[dict]) -> dict:
    out = {"conditions": {}}
    for r in results:
        c = {"display": r["display"], "group": r["group"],
             "poses": r["poses"], "spaces": {}}
        for space, sp in r["spaces"].items():
            pc = sp["per_code"]
            by_beh: dict[str, list] = {}
            for v in pc.values():
                by_beh.setdefault(v["behavior"], []).append(v)
            beh_stats = {
                beh: {
                    "n": len(vs),
                    "median_rho_inst": float(np.nanmedian([x["rho_inst"] for x in vs])),
                    "median_rho_manifold": float(np.nanmedian([x["rho_manifold"] for x in vs])),
                    "median_attractor_spread_norm": float(
                        np.nanmedian([x["attractor_spread_norm"] for x in vs])),
                }
                for beh, vs in by_beh.items()
            }
            c["spaces"][space] = {
                "n_codes": len(pc),
                "median_rho_inst": float(np.nanmedian([v["rho_inst"] for v in pc.values()])),
                "median_rho_manifold": float(np.nanmedian([v["rho_manifold"] for v in pc.values()])),
                "null_rho": float(sp["null_rho"]),
                "per_behavior": beh_stats,
                "per_code": {
                    str(cid): {
                        "behavior": v["behavior"], "label": v["label"],
                        "rho_inst": v["rho_inst"],
                        "rho_manifold": v["rho_manifold"],
                        "attractor_spread_norm": v["attractor_spread_norm"],
                    }
                    for cid, v in pc.items()
                },
            }
        out["conditions"][r["name"]] = c
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs",
            config_name="ic_contraction_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Initial-condition contraction experiment ===")
    rng = np.random.RandomState(int(cfg.get("seed", 0)))
    out_dir = _abs(cfg.output.base_dir) / str(cfg.output.run_name)
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    results = []
    for cond in cfg.conditions:
        cond = OmegaConf.to_container(cond, resolve=True)
        res = analyze_condition(cond, cfg, rng)
        if res is None or not res["spaces"]:
            continue
        results.append(res)
        save_arrays(res, data_dir)
        for space in res["spaces"]:
            plot_condition_curves(
                res, space, fig_dir / f"curves_{res['name']}_{space}.png")
        plot_condition_summary(
            res, fig_dir / f"summary_{res['name']}.png")

    if not results:
        log.error("No conditions produced results.")
        return

    for space in list(cfg.spaces):
        if any(space in r["spaces"] for r in results):
            plot_overview(results, space, fig_dir / f"overview_{space}.png")
            plot_ratio_matrix(results, space, fig_dir / f"ratio_matrix_{space}.png")

    summary = summarize(results)
    (out_dir / "contraction_summary.json").write_text(
        json.dumps(summary, indent=2))

    # Console headline table.
    log.info("\n=== Contraction summary (median ratios; <1 = convergence) ===")
    for name, c in summary["conditions"].items():
        for space, s in c["spaces"].items():
            log.info(
                f"  {c['display']:30s} [{space:6s}] "
                f"rho_inst={s['median_rho_inst']:.3f} "
                f"rho_manifold={s['median_rho_manifold']:.3f} "
                f"null={s['null_rho']:.3f}  (n={s['n_codes']})"
            )
    log.info(f"\n=== Done. Artifacts under {out_dir} ===")


if __name__ == "__main__":
    main()
