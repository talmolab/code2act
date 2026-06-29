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


def _select_body_frames(q_all: np.ndarray, t_perturb: int, n: int):
    """Pick the body that rears strongly *and recovers*, and choose ``n``
    snapshot frames that read as a smooth time-lapse: two reared baseline
    frames, the toppled trough at the impulse, then a dense, evenly-time-spaced
    sweep through the recovery back to a clearly-reared late frame. Rearing is
    intermittent (the body rears up and down), so the baseline/trough/end
    anchors are selected by root height — not a fixed time offset — guaranteeing
    the first and last bodies are in the reared posture; the recovery sweep is
    uniform in time so the row fills the long axis with a continuous arc.
    Returns (body, frame idxs)."""
    T = q_all.shape[1]
    late_lo = min(t_perturb + 260, T - 1)
    body = int(np.argmax(np.median(q_all[:, late_lo:, 2], axis=1)))
    q = q_all[body]

    def _arg(a, b, fn):
        w = np.arange(max(0, a), min(T, b))
        return int(w[fn(q[w, 2])])

    n = max(5, int(n))
    b0 = _arg(t_perturb - 95, t_perturb - 52, np.argmax)   # reared baseline 1
    b1 = _arg(t_perturb - 48, t_perturb - 8, np.argmax)    # reared baseline 2
    pk = _arg(t_perturb + 2, t_perturb + 70, np.argmin)    # toppled (impulse)
    end = _arg(t_perturb + 260, min(t_perturb + 560, T - 1), np.argmax)  # reared
    # uniform-in-time recovery sweep from the trough to the recovered posture,
    # inclusive of both endpoints (pk and end); two reared baselines lead it.
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
                       bottom_crop: float = 0.0) -> np.ndarray | None:
    """Two (or more) code examples side by side, each a grey→red→grey row of
    snapshots: baseline (grey) → shoved (red) → recovered (grey)."""
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
            body, idxs = _select_body_frames(q_all, t_perturb, n_copies)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="kpms")
    ap.add_argument("--amp", type=float, default=2.4,
                    help="amplitude shown in the d(t) panel")
    ap.add_argument("--film-code", type=int, default=4,
                    help="code id for the filmstrip (default 4 = Rear: upright "
                         "→ toppled by the shove → back upright, an in-place "
                         "posture that reads clearly)")
    ap.add_argument("--film-amp", type=float, default=2.4)
    ap.add_argument("--film-frames", type=int, default=11,
                    help="number of frozen-body snapshots in the (a) filmstrip; "
                         "more frames fill the long axis with a smoother "
                         "baseline→topple→recovery→reared time-lapse")
    ap.add_argument("--film-bottom-crop", type=float, default=0.28,
                    help="fraction of the filmstrip height to trim from the bottom "
                         "(removes the floor reflection / shadow)")
    ap.add_argument("--checkpoint",
                    default="model_checkpoints/260611_073323_750085")
    ap.add_argument("--base",
                    default="outputs/perturbation/perturbation_recovery")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 10.5,
    })

    root = get_repo_root()
    run_dir = root / args.base / args.run
    summary = json.loads((run_dir / "perturbation_summary.json").read_text())
    t_perturb = int(summary["t_perturb"])
    T = int(summary["hold_frames"])
    amps = [float(a) for a in summary["amplitudes"]]
    settle = 60
    grid = np.arange(T)
    cloud_idx = np.arange(t_perturb + settle, T)
    rng = np.random.RandomState(0)

    # codes ordered by behavior
    code_beh = {int(c): rec["behavior"] for c, rec in summary["codes"].items()}
    codes = sorted(code_beh, key=lambda c: BEHAVIOR_ORDER.index(code_beh[c])
                   if code_beh[c] in BEHAVIOR_ORDER else 99)
    # wrong-code partner = the next behavior's code (cyclic)
    wrong = {c: codes[(i + 1) % len(codes)] for i, c in enumerate(codes)}

    # ---- figure ----
    # Single example: a postural code (Rear) shown as baseline → shove →
    # recovery, full width.
    examples = [(args.film_code, args.film_amp, "Rear")]
    film = _render_kick_strip(run_dir, examples, t_perturb, args.checkpoint,
                              n_copies=int(args.film_frames),
                              bottom_crop=float(args.film_bottom_crop))

    # Post-impulse pose deviation curves + foreign-code control (panel c).
    dev, dev_ctrl = _recovery_deviation(run_dir, codes, amps, wrong)

    # Layout: (a) filmstrip across the top at full width (keeping its rendered
    # aspect), then a bottom row of two curves, (b) behavior recall and (c) pose
    # deviation, both versus impulse strength and sharing the per-behavior colors.
    fig_w = 12.0
    margin = 0.06
    film_w = 1.0 - 2 * margin
    if film is not None:
        img_h, img_w = film.shape[:2]
        ar = img_w / img_h
    else:
        ar = 5.0
    film_h_in = (film_w * fig_w) / ar
    curve_h_in = 2.7
    gap_in = 0.95          # vertical gap between (a) and the (b)/(c) row
    top_pad_in = 0.15
    bot_pad_in = 0.55      # room for the curve x-axis labels
    fig_h = top_pad_in + film_h_in + gap_in + curve_h_in + bot_pad_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    fhf = film_h_in / fig_h
    ax_film = fig.add_axes(
        [margin, 1.0 - (top_pad_in + film_h_in) / fig_h, film_w, fhf])
    chf = curve_h_in / fig_h
    cy0 = bot_pad_in / fig_h
    ax_basin = fig.add_axes([0.09, cy0, 0.36, chf])
    ax_dev = fig.add_axes([0.59, cy0, 0.36, chf])

    # (a) ghost-trail filmstrip
    ax_film.set_axis_off()
    if film is not None:
        ax_film.imshow(film)

    # round-trip recall curves (panel b) from run_perturbation_roundtrip.
    rt_dir = root / "outputs/perturbation/perturbation_roundtrip/default"
    rt_sum = json.loads((rt_dir / "perturbation_roundtrip_summary.json").read_text())
    rt = dict(np.load(rt_dir / "curves.npz"))
    rt_amps = [float(a) for a in rt["amps"]]
    rt_codes = sorted((int(c) for c in rt_sum["codes"]),
                      key=lambda c: BEHAVIOR_ORDER.index(rt_sum["codes"][str(c)]["behavior"])
                      if rt_sum["codes"][str(c)]["behavior"] in BEHAVIOR_ORDER else 99)

    # (b) absolute round-trip behavior recall in the recovered window vs impulse
    # magnitude (amp 0 = unperturbed baseline); higher is better recovery.
    for c in rt_codes:
        cc = rt_sum["codes"][str(c)]
        ys = [cc["amps"][f"{a:g}"]["final_beh"] for a in rt_amps]
        ax_basin.plot(rt_amps, ys, marker="o", lw=2.0,
                      color=BEHAVIOR_COLORS.get(cc["behavior"], "k"),
                      label=cc["behavior"])
    ax_basin.set_ylim(0, 1.0)
    ax_basin.set_xlabel("impulse strength")
    ax_basin.set_ylabel("behavior recall")
    ax_basin.legend(fontsize=9, loc="lower left")

    # (c) post-impulse pose deviation: MMD between the recovered-window posture of
    # perturbed vs unperturbed rollouts, per code, vs impulse magnitude. Lower is
    # better (re-settled onto the same posture). Two references bound it: the lower
    # dashed line is the unperturbed (amp 0) split-half floor (perfect recovery),
    # and the upper grey band is the distance to a different code's attractor (the
    # control) — staying below it means the body returns to its OWN attractor.
    floor = float(np.median([dev[c][0] for c in codes]))
    ctrl_vals = [dev_ctrl[c] for c in codes]
    ax_dev.axhspan(min(ctrl_vals), max(ctrl_vals), color="0.85", zorder=0)
    ax_dev.axhline(float(np.mean(ctrl_vals)), color="0.55", ls="--", lw=1.0,
                   label="different code (control)")
    # behavior curves share (b)'s color legend; here we label only the two grey
    # reference lines so it is clear what they mean.
    for c in codes:
        ax_dev.plot(amps, dev[c], marker="o", lw=2.0,
                    color=BEHAVIOR_COLORS.get(code_beh[c], "k"))
    ax_dev.axhline(floor, color="0.6", ls=":", lw=1.0,
                   label="unperturbed floor")
    ax_dev.set_ylim(bottom=0)
    ax_dev.set_xlabel("impulse strength")
    ax_dev.set_ylabel("pose MMD\n(perturbed vs unperturbed)")
    ax_dev.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # bold panel labels at each panel's left edge
    ax_film.text(0.0, 1.01, "(a)", transform=ax_film.transAxes,
                 fontsize=15, fontweight="bold", va="bottom")
    ax_basin.text(0.0, 1.04, "(b)", transform=ax_basin.transAxes,
                  fontsize=15, fontweight="bold", va="bottom")
    ax_dev.text(0.0, 1.04, "(c)", transform=ax_dev.transAxes,
                fontsize=15, fontweight="bold", va="bottom")

    out_dir = root / "outputs" / "perturbation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"perturbation_recovery_figure.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_dir}/perturbation_recovery_figure.pdf / .png")


if __name__ == "__main__":
    main()
