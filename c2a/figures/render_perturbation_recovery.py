"""Paper figure: held codes are attractors the controller actively defends.

Stage-2 figure for the perturbation–recovery experiment
(``c2a.experiments.perturbation.run_perturbation_recovery``). Three panels:

(a) **Kick → recovery (intuition).** A ghost-trail of one representative body
    driven by a held code: frozen copies before the impulse (grey), at the
    impulse (red), and recovering afterward (red→green), over the root
    trajectory. The shove visibly throws the body off; the controller walks it
    back onto the code's motif.
(b) **Recovery curves.** Phase-invariant distance ``d(t)`` to the code's
    (unperturbed) attractor cloud, behavior-colored, at a representative impulse
    amplitude. Each curve sits at its converged floor, spikes at the impulse,
    then decays back toward 1 — the controller defends the attractor. A
    wrong-code control (distance to a *different* code's attractor) stays
    elevated: recovery is code-specific.
(c) **Basin.** Recovery fraction vs. impulse magnitude: the controller returns
    to the attractor across the whole sweep (pose space), even for shoves that
    displace the body to many times the converged spread.

Reads the rollouts + summary written by the experiment (no new rollouts for the
quantitative panels; the ghost panel re-renders saved qpos).

Usage::

    python -m c2a.figures.render_perturbation_recovery --run kpms --amp 2.4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from c2a.paths import get_repo_root
from c2a.experiments.perturbation.run_perturbation_recovery import (
    BEHAVIOR_COLORS, _build_cloud, _dist_curve, _features,
)
from c2a.experiments.shared.code_grid import mmd_squared, _median_pairwise

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BEHAVIOR_ORDER = ["Immobile", "Rear", "Walk", "Turn"]


def _load(run_dir: Path, cid: int, amp: float, key: str) -> np.ndarray:
    f = run_dir / "data" / f"rollout_code{cid:03d}_amp{amp:g}.npz"
    return np.load(f)[key]


def _final_pose_pool(q: np.ndarray, frames: int, rng, max_pool: int = 1500
                     ) -> np.ndarray:
    """Root-stripped posture samples from the final ``frames`` of each body.

    ``q`` is ``(K, T, nq)``. Drops the first 3 (world-translation) qpos dims and
    pools the final window across bodies into ``(<=max_pool, nq-3)``, matching the
    distinctness MMD's root strip. The window spans many gait cycles, so the pooled
    posture marginal is phase-invariant (a recovered limit-cycle reads as recovered
    regardless of where in the cycle it ends)."""
    w = q[:, -frames:, 3:]
    flat = w.reshape(-1, w.shape[-1]).astype(np.float32)
    if flat.shape[0] > max_pool:
        flat = flat[rng.choice(flat.shape[0], max_pool, replace=False)]
    return flat


def _recovery_deviation(run_dir: Path, codes: list[int], amps: list[float],
                        wrong: dict[int, int], frames: int = 250
                        ) -> tuple[dict[int, list[float]], dict[int, float]]:
    """Post-impulse pose deviation per code, versus impulse strength, plus a
    foreign-code control.

    For each code we split its ``K`` bodies into a reference half (always
    unperturbed) and a test half. The deviation at impulse strength ``a`` is the
    MMD between the final-window posture distribution of the test half at amplitude
    ``a`` and the unperturbed reference half. At ``a=0`` both halves are
    unperturbed, so the value is the finite-sample / cycle-phase noise floor.
    Same root strip and Gaussian-MMD kernel (median bandwidth) as the distinctness
    analysis, so a small deviation means the body re-settled onto the unperturbed
    posture distribution (it recovered).

    The control (per code) is the MMD between the code's own unperturbed reference
    and a different code's unperturbed posture, on the same kernel: the distance
    between distinct attractors. Recovery curves staying well below this level show
    the body returns to its own attractor rather than to any posture. Returns
    ``(own_curves, control_levels)``."""
    rng = np.random.RandomState(0)
    refs: dict[int, np.ndarray] = {}
    tests: dict[int, np.ndarray] = {}
    q0s: dict[int, np.ndarray] = {}
    foreign: dict[int, np.ndarray] = {}
    for c in codes:
        q0 = _load(run_dir, c, 0.0, "qpos")
        q0s[c] = q0
        perm = rng.permutation(q0.shape[0])
        h = q0.shape[0] // 2
        refs[c] = _final_pose_pool(q0[perm[:h]], frames, rng)
        tests[c] = perm[h:]
        foreign[c] = _final_pose_pool(q0, frames, rng)
    own: dict[int, list[float]] = {}
    ctrl: dict[int, float] = {}
    for c in codes:
        ref, test_idx = refs[c], tests[c]
        pools = [
            _final_pose_pool((q0s[c] if a == 0.0 else
                              _load(run_dir, c, a, "qpos"))[test_idx], frames, rng)
            for a in amps
        ]
        sigma = _median_pairwise(np.concatenate([ref] + pools, axis=0))
        own[c] = [float(np.sqrt(max(0.0, mmd_squared(p, ref, sigma))))
                  for p in pools]
        ctrl[c] = float(np.sqrt(max(0.0, mmd_squared(
            ref, foreign[wrong[c]], sigma))))
    return own, ctrl


def _global_stats(run_dir: Path, codes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of pose features pooled over every code's amp=0 rollout.

    A single shared standardization makes distances to *different* codes'
    attractor clouds directly comparable (needed for the code-specificity
    control: distance to a foreign attractor must be on the same scale as
    distance to the own attractor)."""
    parts = []
    for c in codes:
        f = _features(_load(run_dir, c, 0.0, "qpos"), "pose")
        parts.append(f.reshape(-1, f.shape[-1]))
    flat = np.concatenate(parts, axis=0)
    mean, std = flat.mean(0), flat.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _cloud(run_dir: Path, cid: int, mean, std, cloud_idx, rng):
    f0 = (_features(_load(run_dir, cid, 0.0, "qpos"), "pose") - mean) / std
    return _build_cloud(f0, cloud_idx, 10, 5, 3000, rng)


def _dist_to(run_dir: Path, body_cid: int, amp: float, pca, nn,
             mean, std, grid) -> np.ndarray:
    fz = (_features(_load(run_dir, body_cid, amp, "qpos"), "pose") - mean) / std
    return _dist_curve(fz, pca, nn, grid)


def _phase_color(t: int, tp: int, post: int, alpha: float = 1.0) -> list[float]:
    """Grey on the attractor, red at the shove, fading red→grey as it returns.

    Before the impulse the body is grey (on the attractor). At the impulse it is
    full red (displaced); over the recovery the red fades back to grey, so the
    final (recovered) body is grey again, i.e. back to baseline.
    """
    grey = [0.62, 0.62, 0.66]
    red = [0.90, 0.16, 0.16]
    if t < tp:
        return grey + [min(alpha, 0.92)]
    u = float(np.clip((t - tp) / max(post, 1), 0, 1))
    return [red[k] * (1 - u) + grey[k] * u for k in range(3)] + [alpha]


def _select_body_frames(q_all: np.ndarray, t_perturb: int, n: int,
                        prefer_flat: bool = False, prefer_reared: bool = False):
    """Pick a representative body that recovers to the held code's *own* attractor
    (whatever its posture: low for walk, high for rear) and choose ``n`` snapshot
    frames that read as a smooth time-lapse: two on-attractor baseline frames, the
    displaced trough at the impulse, then a uniform-in-time sweep through the
    recovery. Behavior-agnostic: anchors are by deviation from the code's settled
    root height, not by absolute height, so a walk code does not get rendered as a
    rear.

    ``prefer_flat`` (locomotor codes): among the well-recovering bodies, choose
    the one whose torso stays most horizontal before and after the shove, so a
    walk baseline reads as a low prowling walk rather than an upright, rear-like
    stance. Returns (body, frame idxs)."""
    T = q_all.shape[1]
    pre = slice(max(0, t_perturb - 120), t_perturb)
    late = slice(min(t_perturb + 260, T - 1), T)
    attractor_z = float(np.median(q_all[:, pre, 2].mean(axis=1)))  # settled height
    late_z = q_all[:, late, 2].mean(axis=1)
    rec_err = np.abs(late_z - attractor_z)
    if prefer_flat or prefer_reared:
        def _pitch(quat):  # torso tilt from horizontal (deg); ~0 flat, ~90 rear
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            return np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
        base_pitch = np.abs(_pitch(q_all[:, pre, 3:7])).mean(axis=1)
        late_pitch = np.abs(_pitch(q_all[:, late, 3:7])).mean(axis=1)
        cand = np.where(rec_err <= np.quantile(rec_err, 0.6))[0]
        if cand.size == 0:
            cand = np.arange(q_all.shape[0])
        if prefer_reared:
            # postural codes: body that stays most upright (reared) before AND
            # after the shove, so a rear baseline reads as a clear rear->rear.
            score = np.minimum(base_pitch, late_pitch)
            body = int(cand[np.argmax(score[cand])])
        else:
            body = int(cand[np.argmin(base_pitch[cand] + late_pitch[cand])])
    else:
        # representative body = best recovery back to the attractor height.
        body = int(np.argmin(rec_err))
    q = q_all[body]
    base_z = float(np.mean(q[pre, 2]))

    n = max(5, int(n))
    # two baseline frames just before the impulse (on the attractor)
    b0 = max(0, t_perturb - 70)
    b1 = max(1, t_perturb - 20)
    # trough = max displacement from the attractor height just after the shove
    tw = np.arange(t_perturb, min(T, t_perturb + 90))
    pk = int(tw[np.argmax(np.abs(q[tw, 2] - base_z))]) if tw.size else t_perturb
    # end = late frame closest to the attractor height (recovered)
    ew = np.arange(min(t_perturb + 260, T - 1), T)
    end = int(ew[np.argmin(np.abs(q[ew, 2] - base_z))]) if ew.size else T - 1
    sweep = [int(round(t)) for t in np.linspace(pk, end, max(2, n - 2))]
    idxs = [b0, b1] + sweep
    return body, idxs


def _crop_dark(img: np.ndarray, thr: int = 110, pad: int = 10) -> np.ndarray:
    """Crop the arena (sky/floor) to tighten the rendered row. The bodies are
    grey or red (high red channel); the ambient-lit arena floor is blue (low red
    channel), so cropping on the RED channel isolates the bodies cleanly even
    when the floor is bright."""
    g = img[:, :, 0]   # red channel: grey/red bodies high, blue floor low
    rows = np.where(g.max(axis=1) > thr)[0]
    cols = np.where(g.max(axis=0) > thr)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad)
    return img[r0:r1, c0:c1]


def _render_row(env, q: np.ndarray, idxs: list[int], colors: list[list[float]],
                width: int = 1050, height: int = 760) -> np.ndarray:
    """Render one code's snapshots laid out left→right (translation dropped)."""
    import math
    from c2a.figures.render_behavior_parade_trail import (
        build_parade_trail_model, recolor_copies, render_trail_frame,
    )
    dx = 0.15
    copy_qpos = []
    for i, t in enumerate(idxs):
        qf = q[int(t)].copy()
        qf[0] = i * dx
        qf[1] = 0.0
        copy_qpos.append(qf)
    mid_x = (len(idxs) - 1) * dx / 2.0
    mean_z = float(np.mean([qf[2] for qf in copy_qpos]))
    # Aim at the bodies; the camera sits above and to the side (Fig-2 style).
    lookat = np.array([mid_x, 0.0, max(mean_z, 0.05)])
    segs = [(np.array([[-0.05, 0.0, 0.002],
                       [(len(idxs) - 1) * dx + 0.05, 0.0, 0.002]]),
             [0.5, 0.5, 0.5, 0.22])]
    fovy = 45.0
    row_w = (len(idxs) - 1) * dx + 0.12
    # Match the parade figure's vantage: camera above looking down (negative
    # elevation) at a near-side azimuth, so the row spreads cleanly left-to-right
    # (no recession/overlap) while the elevated angle shows each body's 3D form.
    cam_dist = max(1.0 * row_w / (2 * math.tan(math.radians(fovy) / 2)), 0.34)
    model, base_nq = build_parade_trail_model(
        env, num_copies=len(idxs), trace_segments=segs,
        camera_lookat=lookat, camera_distance=cam_dist,
        camera_elevation=-14.0, camera_azimuth=-90.0, camera_fovy=fovy)
    recolor_copies(model, colors)
    return render_trail_frame(model, base_nq, copy_qpos, width=width, height=height)


def _render_kick_strip(run_dir: Path, examples: list[tuple], t_perturb: int,
                       checkpoint: str, n_copies: int = 11,
                       bottom_crop: float = 0.0,
                       prefer_flat: bool = False,
                       prefer_reared: bool = False) -> np.ndarray | None:
    """Two (or more) code examples side by side, each a grey→red→grey row of
    snapshots: baseline (grey) → shoved (red) → recovered (grey). ``prefer_flat``
    biases body selection toward a horizontal (walk-like) posture for locomotor
    codes so the baseline does not read as a rear."""
    try:
        from c2a.figures.render_codebook_atlas import build_env
        from c2a.experiments.shared.code_grid import add_banner
        root = get_repo_root()
        env = build_env(str(root / checkpoint))
        single = len(examples) == 1
        # scale width with the number of snapshots so each body keeps resolution
        per = 330 if single else 205
        rwidth = int(min(3800 if single else 2700, per * n_copies))
        rows = []
        for cid, amp, label in examples:
            q_all = np.asarray(_load(run_dir, cid, amp, "qpos"), dtype=np.float64)
            body, idxs = _select_body_frames(q_all, t_perturb, n_copies,
                                             prefer_flat=prefer_flat,
                                             prefer_reared=prefer_reared)
            q = q_all[body]
            cpost = max(idxs[-1] - t_perturb, 1)
            colors = [_phase_color(int(t), t_perturb, cpost) for t in idxs]
            img = _crop_dark(_render_row(env, q, idxs, colors, width=rwidth))
            if bottom_crop > 0:
                img = img[: max(1, int(img.shape[0] * (1.0 - bottom_crop)))]
            if not single:
                img = add_banner(img, label, banner_h=30, bg=(255, 255, 255),
                                 fg=(20, 20, 20), font_size=20)
            rows.append(img)
        H = max(r.shape[0] for r in rows)
        rows = [r if r.shape[0] == H else
                np.concatenate([r, np.full((H - r.shape[0], r.shape[1], 3),
                                           255, np.uint8)], axis=0)
                for r in rows]
        sep = np.full((H, 14, 3), 255, np.uint8)
        out = rows[0]
        for r in rows[1:]:
            out = np.concatenate([out, sep, r], axis=1)
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"filmstrip render failed ({e}); quantitative panels only.")
        return None


# ---------------------------------------------------------------------------
# Per-steering-strategy summary figure (3 lines, one per strategy)
# ---------------------------------------------------------------------------
# Each steering strategy (kpms / kvae / mlb) gets ONE line per panel, averaged
# over that strategy's held codes. The 3 strategies are distinguished by a
# tab10 color scheme + a single shared legend (this figure is per-strategy, not
# 9-class-colored). Strategies without a perturbation_recovery dir are skipped.
STRATEGY_ORDER = ["kpms", "best_vae", "mlb"]
STRATEGY_LABELS = {
    "kpms": "Keypoint-MoSeq (kpms)",
    "best_vae": "Kinematics-VAE (kvae)",
    "mlb": "Manually labeled (mlb)",
}


def _mean_curve(summary: dict, space: str, metric: str
                ) -> tuple[list[float], np.ndarray]:
    """Per-amplitude mean (over codes) of a perturbation-summary metric.

    Reads ``summary["codes"][c]["spaces"][space]["metrics"][i][metric]`` for
    every code ``c`` and averages across codes at each amplitude index, ignoring
    ``None`` entries (e.g. ``recovery_frac`` is ``None`` at amplitude 0). Returns
    ``(amplitudes, mean_per_amplitude)`` with ``NaN`` where no code has a value.
    """
    amps = [float(a) for a in summary["amplitudes"]]
    cols: list[list[float]] = [[] for _ in amps]
    for _c, rec in summary["codes"].items():
        sp = rec.get("spaces", {}).get(space)
        if sp is None:
            continue
        for i, m in enumerate(sp["metrics"]):
            if i >= len(cols):
                break
            v = m.get(metric)
            # Exclude non-finite and physically-diverged outliers: a body that
            # blows up under the strongest shove produces an astronomical
            # distance (e.g. 1e25) that would dominate the mean. Distances are
            # in units of the converged spread, so O(10) is already extreme.
            if v is not None and np.isfinite(float(v)) and abs(float(v)) < 1e3:
                cols[i].append(float(v))
    mean = np.array([np.mean(col) if col else np.nan for col in cols])
    return amps, mean


def _deviation_with_control(base: Path, s: str, summ: dict):
    """Per-strategy pose deviation vs impulse (mean over codes) + the foreign-code
    control level and the unperturbed floor, via the rigorous root-stripped MMD
    (:func:`_recovery_deviation`). Returns ``(amps, dev_mean, floor, foreign)`` or
    ``None`` on failure."""
    amps = [float(a) for a in summ["amplitudes"]]
    codes = [int(c) for c in summ["codes"].keys()]
    if len(codes) < 2:
        return None
    wrong = {c: codes[(i + 1) % len(codes)] for i, c in enumerate(codes)}
    try:
        own, ctrl = _recovery_deviation(base / s, codes, amps, wrong)
    except Exception as e:  # noqa: BLE001
        log.warning(f"deviation failed for {s}: {e}")
        return None
    dev = np.nanmean(np.array([own[c] for c in codes if c in own], float), axis=0)
    foreign = float(np.nanmean([ctrl[c] for c in codes if c in ctrl]))
    return np.array(amps), dev, float(dev[0]), foreign


def _beh_mean_recovery(summ: dict, bcodes: list[int], n_amp: int) -> np.ndarray:
    """Mean pose recovery_frac over a behavior's codes, per amplitude (NaN-safe)."""
    out = []
    for i in range(n_amp):
        vals = []
        for c in bcodes:
            v = summ["codes"][str(c)]["spaces"]["pose"]["metrics"][i].get("recovery_frac")
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(out)


def _render_only_strategy(s: str, summ: dict, base: Path, out_dir: Path) -> None:
    """Standalone panels for ONE steering strategy, used by the main contraction
    figure (mlb): pose recovery fraction ``(b)`` and post-impulse pose deviation
    ``(c)`` are each rendered as their own equally sized square panel, plus a
    shared horizontal legend, so the figure can place ``(a)`` confusion, ``(b)``
    and ``(c)`` at the same size on one row with the legend beneath all three.
    Reuses ``_beh_mean_recovery`` / ``_recovery_deviation``. Emits
    ``perturbation_recovery_<s>_{rec,dev,legend}.{pdf,png}``."""
    from matplotlib.lines import Line2D

    amps = [float(a) for a in summ["amplitudes"]]
    codes = [int(c) for c in summ["codes"]]
    beh_of = {int(c): summ["codes"][c].get("behavior") for c in summ["codes"]}
    wrong = {c: codes[(i + 1) % len(codes)] for i, c in enumerate(codes)}
    try:
        own, ctrl = _recovery_deviation(base / s, codes, amps, wrong)
    except Exception as e:  # noqa: BLE001
        log.warning(f"deviation failed for {s}: {e}"); own, ctrl = {}, {}

    seen_beh = [b for b in BEHAVIOR_ORDER
                if any(beh_of[c] == b for c in codes)]

    def _save(fig, suffix, pad=0.15):
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"perturbation_recovery_{s}_{suffix}.{ext}",
                        dpi=200, bbox_inches="tight", pad_inches=pad,
                        facecolor="white")
        plt.close(fig)

    # ---- panel (b): recovery fraction ----
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.text(-0.02, 1.04, "(b)", transform=ax.transAxes, fontsize=26,
            fontweight="bold", va="bottom")
    ax.set_ylabel("recovery frac.", fontsize=22)
    for b in BEHAVIOR_ORDER:
        bc = [c for c in codes if beh_of[c] == b]
        if not bc:
            continue
        rec = _beh_mean_recovery(summ, bc, len(amps))
        ok = ~np.isnan(rec)
        if ok.any():
            ax.plot(np.array(amps)[ok], rec[ok], "o-", lw=2.0,
                    color=BEHAVIOR_COLORS.get(b, "0.5"))
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.set_xlabel("impulse strength (a.u.)")
    fig.tight_layout(); _save(fig, "rec")

    # ---- panel (c): post-impulse deviation ----
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.text(-0.02, 1.04, "(c)", transform=ax.transAxes, fontsize=26,
            fontweight="bold", va="bottom")
    ax.set_ylabel("pose deviation", fontsize=22)
    for b in BEHAVIOR_ORDER:
        bc = [c for c in codes if beh_of[c] == b]
        dv = [own[c] for c in bc if c in own]
        if dv:
            ax.plot(amps, np.nanmean(np.array(dv), axis=0), "o-", lw=2.0,
                    color=BEHAVIOR_COLORS.get(b, "0.5"))
    if own:
        ax.axhline(float(np.nanmean([own[c][0] for c in codes if c in own])),
                   color="0.45", ls=":", lw=1.3)
    if ctrl:
        ax.axhline(float(np.nanmean([ctrl[c] for c in codes if c in ctrl])),
                   color="0.6", ls="--", lw=1.2)
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
    ax.set_xlabel("impulse strength (a.u.)")
    fig.tight_layout(); _save(fig, "dev")

    # ---- shared horizontal legend (placed beneath a, b, c) ----
    # Behaviors as blocky color swatches, matching the codebook / DSA figures'
    # color coding; the floor and control references stay as line styles.
    from matplotlib.patches import Patch
    beh_h = [Patch(facecolor=BEHAVIOR_COLORS.get(b, "0.5"), edgecolor="none",
                   label=b) for b in BEHAVIOR_ORDER if b in seen_beh]
    ctrl_h = [Line2D([], [], color="0.45", ls=":", label="unperturbed floor"),
              Line2D([], [], color="0.6", ls="--", label="foreign code (control)")]
    figL, axL = plt.subplots(figsize=(12.0, 0.55)); axL.axis("off")
    axL.legend(handles=beh_h + ctrl_h, loc="center", ncol=len(beh_h) + 2,
               frameon=False, fontsize=17, columnspacing=1.6, handletextpad=0.5,
               handlelength=1.4, handleheight=1.4)
    _save(figL, "legend", pad=0.05)
    print(f"wrote {out_dir}/perturbation_recovery_{s}_{{rec,dev,legend}}.pdf/.png "
          f"(3-part, {s})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="outputs/perturbation/perturbation_recovery",
                    help="directory holding one subdir per steering strategy")
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--only", default=None,
                    help="emit a standalone 2-panel (recovery frac. | deviation) "
                         "figure for just this strategy -> "
                         "perturbation_recovery_<strategy>.pdf (main Fig 3 c,d). "
                         "Without it, the all-strategy figure is produced.")
    ap.add_argument("--no-filmstrip", action="store_true",
                    help="skip the appendix kick→recovery filmstrip render")
    ap.add_argument("--film-pick", nargs="*", default=None,
                    help="override the filmstrip code for a given "
                         "strategy/behavior cell, e.g. 'mlb:Rear=4'. Defaults to "
                         "the first code of that behavior. Repeatable.")
    args = ap.parse_args()

    # Parse per-cell filmstrip code overrides -> {(strategy, behavior): code}.
    film_pick: dict[tuple[str, str], int] = {}
    for spec in (args.film_pick or []):
        key, _, val = spec.partition("=")
        strat, _, beh = key.partition(":")
        film_pick[(strat, beh)] = int(val)

    plt.rcParams.update({"font.size": 22, "axes.labelsize": 22,
                         "xtick.labelsize": 19, "ytick.labelsize": 19,
                         "legend.fontsize": 18})
    root = get_repo_root()
    base = root / args.base
    want = args.strategies if args.strategies else STRATEGY_ORDER
    strategies = [s for s in want
                  if (base / s / "perturbation_summary.json").exists()]
    if not strategies:
        log.warning(f"no strategy summaries under {base}; nothing to plot")
        return
    log.info(f"strategies with data: {strategies}")
    cmap = plt.get_cmap("tab10")
    colors = {s: cmap(i % 10) for i, s in enumerate(STRATEGY_ORDER)}
    summaries = {s: json.loads((base / s / "perturbation_summary.json").read_text())
                 for s in strategies}
    out_dir = root / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single-strategy mode: emit just the 2-panel (c,d) figure and return.
    if args.only:
        if args.only not in strategies:
            log.warning(f"--only {args.only}: no perturbation summary under "
                        f"{base / args.only}; nothing to plot")
            return
        _render_only_strategy(args.only, summaries[args.only], base, out_dir)
        return

    # --- Main figure: one ROW per steering strategy (kpms, kvae, mlb); two
    #     columns -- pose recovery fraction (left) and post-impulse pose deviation
    #     with the unperturbed floor + foreign-code control band (right). Within
    #     each panel, one line per behavior category. Strategies without data
    #     (mlb, pending retrain) get a labelled TBD row. Panels are labelled
    #     (a)..(f), starting from (a).
    from matplotlib.lines import Line2D
    BEHAVIOR_ORDER = ["Immobile", "Rear", "Walk", "Turn"]
    tags = [["(a)", "(b)"], ["(c)", "(d)"], ["(e)", "(f)"]]
    fig, axes = plt.subplots(len(STRATEGY_ORDER), 2, squeeze=False,
                             figsize=(10.0, 3.7 * len(STRATEGY_ORDER)))
    seen_beh: list[str] = []
    for r, s in enumerate(STRATEGY_ORDER):
        ax_rec, ax_dev = axes[r]
        ax_rec.text(-0.05, 1.06, tags[r][0], transform=ax_rec.transAxes,
                    fontsize=25, fontweight="bold", va="bottom")
        ax_dev.text(-0.05, 1.06, tags[r][1], transform=ax_dev.transAxes,
                    fontsize=25, fontweight="bold", va="bottom")
        ax_rec.set_ylabel(f"{STRATEGY_LABELS[s].split(' (')[0]}\npose recovery frac.",
                          fontsize=18)
        ax_dev.set_ylabel("pose deviation", fontsize=18)
        if s not in strategies:
            for ax in (ax_rec, ax_dev):
                ax.text(0.5, 0.5, "pending mlb retrain (TBD)", ha="center",
                        va="center", color="0.55", fontsize=17,
                        fontweight="bold", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            continue
        summ = summaries[s]
        amps = [float(a) for a in summ["amplitudes"]]
        codes = [int(c) for c in summ["codes"]]
        beh_of = {int(c): summ["codes"][c].get("behavior") for c in summ["codes"]}
        wrong = {c: codes[(i + 1) % len(codes)] for i, c in enumerate(codes)}
        try:
            own, ctrl = _recovery_deviation(base / s, codes, amps, wrong)
        except Exception as e:  # noqa: BLE001
            log.warning(f"deviation failed for {s}: {e}"); own, ctrl = {}, {}
        for b in BEHAVIOR_ORDER:
            bc = [c for c in codes if beh_of[c] == b]
            if not bc:
                continue
            col = BEHAVIOR_COLORS.get(b, "0.5")
            rec = _beh_mean_recovery(summ, bc, len(amps))
            ok = ~np.isnan(rec)
            if ok.any():
                ax_rec.plot(np.array(amps)[ok], rec[ok], "o-", lw=1.8,
                            color=col, label=b)
            dv = [own[c] for c in bc if c in own]
            if dv:
                ax_dev.plot(amps, np.nanmean(np.array(dv), axis=0), "o-",
                            lw=1.8, color=col, label=b)
            if b not in seen_beh:
                seen_beh.append(b)
        if own:
            ax_dev.axhline(float(np.nanmean([own[c][0] for c in codes if c in own])),
                           color="0.45", ls=":", lw=1.1)
        if ctrl:
            ax_dev.axhline(float(np.nanmean([ctrl[c] for c in codes if c in ctrl])),
                           color="0.6", ls="--", lw=1.0)
        ax_rec.set_ylim(0, 1.02); ax_rec.grid(alpha=0.3)
        ax_dev.set_ylim(bottom=0); ax_dev.grid(alpha=0.3)
    axes[-1][0].set_xlabel("impulse strength (a.u.)")
    axes[-1][1].set_xlabel("impulse strength (a.u.)")
    beh_h = [Line2D([], [], marker="o", color=BEHAVIOR_COLORS.get(b, "0.5"),
                    label=b) for b in BEHAVIOR_ORDER if b in seen_beh]
    ctrl_h = [Line2D([], [], color="0.45", ls=":", label="unperturbed floor"),
              Line2D([], [], color="0.6", ls="--", label="foreign code (control)")]
    fig.legend(handles=beh_h + ctrl_h, loc="lower center",
               ncol=len(beh_h) + 2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"perturbation_recovery_figure.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_dir}/perturbation_recovery_figure.pdf/.png "
          f"(per-behavior, per-strategy a-f)")

    # --- Appendix: kick→recovery filmstrip GRID. Columns = recovery from a
    #     postural (Rear) vs a locomotor (Walk) code; rows = steering strategy.
    #     Each cell is a grey→red→grey snapshot strip. mlb rows are TBD. ---
    if not args.no_filmstrip:
        col_specs = [("Rear", "recovery from rear"), ("Walk", "recovery from walk")]
        cells: dict[tuple[str, str], np.ndarray] = {}
        for s in strategies:
            summ = summaries[s]
            amps = [float(a) for a in summ["amplitudes"]]
            amp = amps[max(1, len(amps) // 2)] if len(amps) > 1 else amps[-1]
            t_perturb = int(summ.get("t_perturb", 800))
            ckpt = str(summ.get("checkpoint", "")).replace(str(root) + "/", "")
            for beh, _ in col_specs:
                bc = [int(c) for c, v in summ["codes"].items()
                      if v.get("behavior") == beh]
                if not bc:
                    continue
                cid = film_pick.get((s, beh), bc[0])
                if cid not in bc:  # guard against a stale override
                    log.warning(f"film-pick {s}:{beh}={cid} not a {beh} code; "
                                f"using {bc[0]}")
                    cid = bc[0]
                img = _render_kick_strip(base / s, [(cid, amp, "")], t_perturb,
                                         ckpt, n_copies=9, bottom_crop=0.16,
                                         prefer_flat=(beh == "Walk"),
                                         prefer_reared=(beh == "Rear"))
                if img is not None:
                    cells[(s, beh)] = img
                    print(f"  filmstrip {s}/{beh}: code {cid}, amp {amp}")
        if cells:
            nrow, ncol = len(STRATEGY_ORDER), len(col_specs)
            # Pad every strip to a uniform (maxH x maxW) with white so all cells
            # are the same size -> columns align and aspect="equal" leaves no
            # distortion. figsize matches the uniform-cell aspect.
            maxH = max(im.shape[0] for im in cells.values())
            maxW = max(im.shape[1] for im in cells.values())

            def _pad(im):
                ph, pw = maxH - im.shape[0], maxW - im.shape[1]
                return np.pad(im, ((ph // 2, ph - ph // 2),
                                   (pw // 2, pw - pw // 2), (0, 0)),
                              constant_values=255)

            short = {"kpms": "kpms", "best_vae": "kvae", "mlb": "mlb"}
            ar = maxW / maxH
            cell_w = 5.2
            fig2, axes2 = plt.subplots(nrow, ncol, squeeze=False,
                                       figsize=(cell_w * ncol,
                                                (cell_w / ar) * nrow))
            for ci, (_, lbl) in enumerate(col_specs):
                axes2[0][ci].set_title(lbl, fontsize=13, fontweight="bold")
            for ri, s in enumerate(STRATEGY_ORDER):
                axes2[ri][0].set_ylabel(short.get(s, s), fontsize=12,
                                        fontweight="bold", labelpad=6)
                for ci, (beh, _) in enumerate(col_specs):
                    ax = axes2[ri][ci]
                    ax.set_xticks([]); ax.set_yticks([])
                    if (s, beh) in cells:
                        ax.imshow(_pad(cells[(s, beh)]), aspect="equal")
                    else:
                        txt = ("pending mlb retrain (TBD)" if s not in strategies
                               else "no code")
                        col = "0.55" if s not in strategies else "0.7"
                        ax.text(0.5, 0.5, txt, ha="center", va="center",
                                color=col, fontsize=12, fontweight="bold",
                                transform=ax.transAxes)
                        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                        for sp in ax.spines.values():
                            sp.set_color("0.8")
            fig2.subplots_adjust(wspace=0.04, hspace=0.08)
            for ext in ("pdf", "png"):
                fig2.savefig(out_dir / f"perturbation_filmstrip.{ext}",
                             dpi=200, bbox_inches="tight", facecolor="white")
            plt.close(fig2)
            print(f"wrote {out_dir}/perturbation_filmstrip.pdf/.png "
                  f"({ncol} cols x {nrow} rows)")
        else:
            print("filmstrip render unavailable; skipped")


if __name__ == "__main__":
    main()
