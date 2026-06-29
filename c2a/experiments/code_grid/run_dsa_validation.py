"""Is the per-code DSA distance matrix (Fig 3 c,d) genuine dynamics or fitted noise?

DSA fits a reduced Hankel/delay-embedded DMD operator to each code's GRU
hidden-state trajectories and compares operators by Procrustes (angular distance),
yielding the pairwise matrix in the paper. If the per-code operators merely fit
noise, that matrix and its block structure are spurious. We test this three ways,
grounded in the DMD-noise and surrogate-data literature:

  T1  Held-out prediction R^2 of the reduced Hankel-DMD operator. A DMD that
      captures real dynamics predicts HELD-OUT trajectories (one-step and k-step)
      far above a time-shuffled null; a noise fit decays to the null floor.
      (Bagheri 2014; Ostrow et al. 2023 HAVOK-DMD is a predictive linear model.)
  T2  Eigenvalue noise diagnostics: the |lambda| spectrum (noise biases it inward,
      toward 0 / spurious damping) and the forward-vs-backward operator
      discrepancy (a direct noise meter). (Bagheri 2014; Hemati/Dawson de-biasing.)
  T3  Split-half reliability of the FULL DSA distance matrix via the Mantel
      statistic (correlation of the two halves' off-diagonals) against a
      code-label-shuffle null. A reproducible matrix is signal; a seed-unstable
      one is noise. (Mantel 1967.)

This reproduces the pipeline's DSA input (per-code hidden, shared PCA to 32 dims,
n_delays=5, rank=32) so the verdict applies to the paper's matrix.

Run:
    python -m c2a.experiments.code_grid.run_dsa_validation --run kpms
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from c2a.paths import get_repo_root

GROUP_COLORS = {"Immobile": "#4C72B0", "Rear": "#DD8452",
                "Turn": "#8172B3", "Walk": "#55A868", "Other": "#999999"}


# --------------------------------------------------------------------------- #
# Data: reproduce the pipeline's per-code hidden systems (shared PCA to 32)
# --------------------------------------------------------------------------- #
def load_systems(run: str, pca_dim: int, max_codes: int, n_bodies: int):
    root = get_repo_root()
    data_dir = root / "outputs/code_grid/code_grid_rollouts" / run / "data"
    comp_path = (root / "outputs/code_grid/code_grid_dynamics" / run
                 / "random_raw/hidden/composition.json")
    comp = json.loads(comp_path.read_text())
    clusters = comp["clusters"]
    if max_codes:
        clusters = clusters[:max_codes]
    codes, labels, hidden = [], [], {}
    for cl in clusters:
        c = int(cl["codes"][0])
        f = data_dir / f"rollout_random_code{c:03d}.npz"
        if not f.exists():
            continue
        h = np.asarray(np.load(f)["hidden"])[:n_bodies]   # (K, T, H)
        codes.append(c)
        labels.append(str(cl.get("label", "Other")).split("_")[0])
        hidden[c] = h
    # shared PCA across all codes/bodies/frames (as the pipeline does)
    pool = np.concatenate([h.reshape(-1, h.shape[-1]) for h in hidden.values()], 0)
    pca = PCA(n_components=pca_dim, random_state=0).fit(pool)
    systems = {c: np.stack([pca.transform(hidden[c][k])
                            for k in range(hidden[c].shape[0])], 0)
               for c in codes}   # c -> (K, T, pca_dim)
    return codes, labels, systems, pca


# --------------------------------------------------------------------------- #
# Reduced Hankel-DMD (faithful to DSA's HAVOK fit) with train/test control
# --------------------------------------------------------------------------- #
def _hankel(traj: np.ndarray, d: int) -> np.ndarray:
    """(T, dim) -> (T-d+1, d*dim); row k is the window [k .. k+d-1] flattened."""
    T = traj.shape[0]
    return np.stack([traj[k:k + d].reshape(-1) for k in range(T - d + 1)], 0)


def _basis(trajs: list[np.ndarray], d: int, r: int):
    """Top-r right singular vectors (spatial modes) of the pooled delay matrix,
    plus the retained-energy fraction at the truncation."""
    H = np.concatenate([_hankel(t, d) for t in trajs], 0)
    Hc = H - H.mean(0, keepdims=True)
    # economy SVD of (snapshots x features); Vt rows are feature modes
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    energy = float((S[:r] ** 2).sum() / (S ** 2).sum())
    return H.mean(0), Vt[:r], energy


def _reduced_pairs(trajs, d, mean, W):
    """Project each trajectory's Hankel to reduced coords and return the
    within-trajectory (z_t, z_{t+1}) pair stacks (no cross-trajectory pairs)."""
    cur, nxt = [], []
    for t in trajs:
        Z = (_hankel(t, d) - mean) @ W.T          # (rows, r)
        cur.append(Z[:-1]); nxt.append(Z[1:])
    return np.concatenate(cur, 0), np.concatenate(nxt, 0)


def _fit_operator(Zc, Zn):
    """z_{t+1} = A z_t  (least squares)."""
    B, *_ = np.linalg.lstsq(Zc, Zn, rcond=None)   # Zc B = Zn
    return B.T                                    # A


def _r2(true, pred):
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def held_out_r2(trajs, d, r, ksteps, rng, shuffle=False):
    """Fit reduced DMD on a train split, score one-step + k-step prediction on
    held-out trajectories. ``shuffle`` time-permutes each trajectory (the noise
    null). Returns dict with r2_1, r2_k, eigs, fb_discrepancy, energy."""
    trajs = list(trajs)
    if shuffle:
        trajs = [t[rng.permutation(t.shape[0])] for t in trajs]
    n = len(trajs)
    idx = rng.permutation(n)
    tr = [trajs[i] for i in idx[: int(0.7 * n)]]
    te = [trajs[i] for i in idx[int(0.7 * n):]]
    mean, W, energy = _basis(tr, d, r)
    Zc, Zn = _reduced_pairs(tr, d, mean, W)
    A = _fit_operator(Zc, Zn)
    Ab = _fit_operator(Zn, Zc)                    # backward operator
    # held-out one-step
    Zc_te, Zn_te = _reduced_pairs(te, d, mean, W)
    r2_1 = _r2(Zn_te, Zc_te @ A.T)
    # held-out k-step rollout within each test trajectory
    r2_k = np.nan
    if ksteps > 1:
        tr_all, pr_all = [], []
        Ak = np.linalg.matrix_power(A, ksteps)
        for t in te:
            Z = (_hankel(t, d) - mean) @ W.T
            if Z.shape[0] > ksteps:
                tr_all.append(Z[ksteps:]); pr_all.append(Z[:-ksteps] @ Ak.T)
        if tr_all:
            r2_k = _r2(np.concatenate(tr_all), np.concatenate(pr_all))
    eigs = np.linalg.eigvals(A)
    eigs_b = np.linalg.eigvals(Ab)
    # forward-backward discrepancy: forward eigs vs reciprocal backward eigs
    fb = float(np.mean(np.abs(np.sort_complex(eigs)
                              - 1.0 / np.sort_complex(eigs_b))))
    return {"r2_1": float(r2_1), "r2_k": float(r2_k),
            "abs_eigs": np.abs(eigs), "fb_discrepancy": fb,
            "energy": energy}


# --------------------------------------------------------------------------- #
# T3: split-half reliability of the DSA matrix (Mantel)
# --------------------------------------------------------------------------- #
def _dsa_matrix(systems_list, rank, n_delays):
    from DSA import DSA as PlainDSA
    dsa = PlainDSA(X=systems_list, n_delays=n_delays, rank=rank,
                   score_method="angular", device="cpu", verbose=False)
    D = np.asarray(dsa.fit_score())
    if D.ndim == 3:
        D = D.mean(0)
    return D


def _offdiag(D):
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def split_half_mantel(codes, systems, rank, n_delays, seeds, n_perm, rng):
    """Mantel correlation between DSA matrices from two disjoint body-halves, with
    a code-label-shuffle null. Returns per-seed r and null distribution."""
    results = []
    for s in range(seeds):
        rs = np.random.RandomState(1000 + s)
        Da_sys, Db_sys = [], []
        for c in codes:
            S = systems[c]
            perm = rs.permutation(S.shape[0])
            half = S.shape[0] // 2
            Da_sys.append(S[perm[:half]])
            Db_sys.append(S[perm[half:2 * half]])
        Da = _dsa_matrix([s for s in Da_sys], rank, n_delays)
        Db = _dsa_matrix([s for s in Db_sys], rank, n_delays)
        a, b = _offdiag(Da), _offdiag(Db)
        r = float(spearmanr(a, b).correlation)
        # label-shuffle null: permute Db rows+cols
        null = []
        n = Da.shape[0]
        for _ in range(n_perm):
            p = rng.permutation(n)
            null.append(float(spearmanr(a, _offdiag(Db[np.ix_(p, p)])).correlation))
        pval = (1 + np.sum(np.asarray(null) >= r)) / (n_perm + 1)
        results.append({"seed": s, "mantel_r": r, "p": float(pval),
                        "null_mean": float(np.mean(null))})
        print(f"  [T3] seed {s}: Mantel r={r:.3f} p={pval:.3f} "
              f"(null mean {np.mean(null):.3f})", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="kpms")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--n-delays", type=int, default=5)
    ap.add_argument("--ksteps", type=int, default=25)
    ap.add_argument("--n-bodies", type=int, default=100)
    ap.add_argument("--max-codes", type=int, default=0)
    ap.add_argument("--mantel-seeds", type=int, default=8)
    ap.add_argument("--mantel-perm", type=int, default=999)
    ap.add_argument("--skip-mantel", action="store_true")
    ap.add_argument("--out", default="outputs/code_grid/dsa_validation")
    args = ap.parse_args()

    root = get_repo_root()
    out_dir = root / args.out / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)

    print(f"[load] run={args.run} ...", flush=True)
    codes, labels, systems, _ = load_systems(
        args.run, args.pca_dim, args.max_codes, args.n_bodies)
    print(f"[load] {len(codes)} codes, system shape "
          f"{systems[codes[0]].shape}", flush=True)

    # ---- T1 + T2: per-code held-out R^2 and eigenvalue diagnostics ----
    rows = []
    for c, lab in zip(codes, labels):
        real = held_out_r2(systems[c], args.n_delays, args.rank, args.ksteps,
                           np.random.RandomState(7), shuffle=False)
        null = held_out_r2(systems[c], args.n_delays, args.rank, args.ksteps,
                           np.random.RandomState(7), shuffle=True)
        rows.append({"code": c, "label": lab,
                     "r2_1": real["r2_1"], "r2_k": real["r2_k"],
                     "r2_1_null": null["r2_1"], "r2_k_null": null["r2_k"],
                     "fb_discrepancy": real["fb_discrepancy"],
                     "abs_eig_max": float(real["abs_eigs"].max()),
                     "abs_eig_median": float(np.median(real["abs_eigs"])),
                     "svd_energy": real["energy"]})
        print(f"  [T1] code {c:3d} {lab:9s} R2_1={real['r2_1']:.3f} "
              f"(null {null['r2_1']:.3f})  R2_{args.ksteps}={real['r2_k']:.3f} "
              f"(null {null['r2_k']:.3f})  |lam|med={np.median(real['abs_eigs']):.3f} "
              f"fb={real['fb_discrepancy']:.3f} energy={real['energy']:.2f}",
              flush=True)

    r2_1 = np.array([r["r2_1"] for r in rows])
    r2_1n = np.array([r["r2_1_null"] for r in rows])
    r2_k = np.array([r["r2_k"] for r in rows])
    r2_kn = np.array([r["r2_k_null"] for r in rows])

    # ---- T3: split-half Mantel reliability ----
    mantel = []
    if not args.skip_mantel:
        print("[T3] split-half Mantel reliability ...", flush=True)
        mantel = split_half_mantel(codes, systems, args.rank, args.n_delays,
                                   args.mantel_seeds, args.mantel_perm, rng)

    # ---- verdict ----
    summary = {
        "run": args.run, "n_codes": len(codes),
        "T1_r2_1_median": float(np.median(r2_1)),
        "T1_r2_1_null_median": float(np.median(r2_1n)),
        "T1_r2_k_median": float(np.nanmedian(r2_k)),
        "T1_r2_k_null_median": float(np.nanmedian(r2_kn)),
        "T2_fb_discrepancy_median": float(np.median([r["fb_discrepancy"] for r in rows])),
        "T2_abs_eig_median": float(np.median([r["abs_eig_median"] for r in rows])),
        "T2_svd_energy_median": float(np.median([r["svd_energy"] for r in rows])),
        "T3_mantel_r_median": float(np.median([m["mantel_r"] for m in mantel])) if mantel else None,
        "T3_mantel_null_median": float(np.median([m["null_mean"] for m in mantel])) if mantel else None,
        "per_code": rows, "mantel": mantel,
    }
    (out_dir / "dsa_validation.json").write_text(json.dumps(summary, indent=2))

    # ---- figure ----
    _plot(out_dir, rows, mantel, args)
    _verdict(summary, args)
    print(f"wrote {out_dir}/dsa_validation.json")


def _plot(out_dir, rows, mantel, args):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    labs = [r["label"] for r in rows]
    cols = [GROUP_COLORS.get(l, "#999") for l in labs]
    # (a) held-out one-step R2: real vs null per code
    x = np.arange(len(rows))
    ax[0].bar(x - 0.2, [r["r2_1"] for r in rows], 0.4, color=cols, label="real")
    ax[0].bar(x + 0.2, [r["r2_1_null"] for r in rows], 0.4, color="0.7",
              label="time-shuffled null")
    ax[0].set_xticks(x); ax[0].set_xticklabels([r["code"] for r in rows],
                                               rotation=90, fontsize=6)
    ax[0].set_ylabel("held-out 1-step $R^2$"); ax[0].legend(fontsize=8)
    ax[0].set_title("T1: per-code DMD prediction (held out)", fontsize=10, loc="left")
    # (b) k-step real vs null scatter
    ax[1].scatter([r["r2_k_null"] for r in rows], [r["r2_k"] for r in rows],
                  c=cols, s=40, edgecolor="white")
    lim = [-0.1, 1.0]
    ax[1].plot(lim, lim, "0.7", ls=":")
    ax[1].set_xlabel(f"null {args.ksteps}-step $R^2$")
    ax[1].set_ylabel(f"real {args.ksteps}-step $R^2$")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_title(f"T1: {args.ksteps}-step prediction", fontsize=10, loc="left")
    # (c) Mantel reliability
    if mantel:
        rs = [m["mantel_r"] for m in mantel]
        nl = [m["null_mean"] for m in mantel]
        ax[2].boxplot([rs, nl], tick_labels=["split-half", "label-shuffle\nnull"])
        ax[2].set_ylabel("Mantel $r$ (Spearman)")
        ax[2].set_ylim(-0.3, 1.0)
        ax[2].axhline(0, color="0.7", lw=0.8)
    ax[2].set_title("T3: distance-matrix reliability", fontsize=10, loc="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"dsa_validation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _verdict(s, args):
    print("\n================= DSA VALIDATION VERDICT =================")
    print(f"run={s['run']}  ({s['n_codes']} codes)")
    print(f"T1 held-out 1-step R^2 : real median {s['T1_r2_1_median']:.3f} "
          f"vs null {s['T1_r2_1_null_median']:.3f}")
    print(f"T1 held-out {args.ksteps}-step R^2: real median {s['T1_r2_k_median']:.3f} "
          f"vs null {s['T1_r2_k_null_median']:.3f}")
    print(f"T2 |lambda| median     : {s['T2_abs_eig_median']:.3f}  "
          f"(noise collapses toward 0)")
    print(f"T2 fwd/bwd discrepancy : {s['T2_fb_discrepancy_median']:.3f}  "
          f"(large = noise-biased operator)")
    print(f"T2 rank-{args.rank} SVD energy : {s['T2_svd_energy_median']:.2f}")
    if s["T3_mantel_r_median"] is not None:
        print(f"T3 split-half Mantel r : {s['T3_mantel_r_median']:.3f}  "
              f"(label-shuffle null {s['T3_mantel_null_median']:.3f})")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
