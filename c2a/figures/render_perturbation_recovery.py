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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BEHAVIOR_ORDER = ["Immobile", "Rear", "Walk", "Turn"]


def _load(run_dir: Path, cid: int, amp: float, key: str) -> np.ndarray:
    f = run_dir / "data" / f"rollout_code{cid:03d}_amp{amp:g}.npz"
    return np.load(f)[key]


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
        return grey + [min(alpha, 0.6)]
    u = float(np.clip((t - tp) / max(post, 1), 0, 1))
    return [red[k] * (1 - u) + grey[k] * u for k in range(3)] + [alpha]


def _select_body_frames(q_all: np.ndarray, t_perturb: int, n: int):
    """Pick the body that rears strongly *and recovers*, and choose snapshot
    frames by posture (root height): a reared baseline, the toppled trough,
    intermediates, and a clearly-reared late frame. Rearing is intermittent
    (the body rears up and down), so selecting by height — not a fixed time
    offset — guarantees the first and last bodies are in the reared posture.
    Returns (body, frame idxs)."""
    T = q_all.shape[1]
    late_lo = min(t_perturb + 260, T - 1)
    body = int(np.argmax(np.median(q_all[:, late_lo:, 2], axis=1)))
    q = q_all[body]

    def _arg(a, b, fn):
        w = np.arange(max(0, a), min(T, b))
        return int(w[fn(q[w, 2])])

    before = _arg(t_perturb - 80, t_perturb - 8, np.argmax)   # reared
    pk = _arg(t_perturb + 2, t_perturb + 70, np.argmin)       # toppled
    end = _arg(t_perturb + 260, t_perturb + 470, np.argmax)   # reared
    mids = [int(pk + f * (end - pk))
            for f in np.linspace(0.34, 0.66, max(0, n - 3))]
    return body, [before, pk] + mids + [end]


def _crop_dark(img: np.ndarray, thr: int = 16, pad: int = 6) -> np.ndarray:
    """Crop near-black margins (sky/floor) to tighten the rendered row."""
    g = img.max(axis=2)
    rows = np.where(g.max(axis=1) > thr)[0]
    cols = np.where(g.max(axis=0) > thr)[0]
    if rows.size == 0 or cols.size == 0:
        return img
    r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + pad)
    c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + pad)
    return img[r0:r1, c0:c1]


def _render_row(env, q: np.ndarray, idxs: list[int], colors: list[list[float]],
                width: int = 1050, height: int = 430) -> np.ndarray:
    """Render one code's snapshots laid out left→right (translation dropped)."""
    import math
    from c2a.figures.render_behavior_parade_trail import (
        build_parade_trail_model, recolor_copies, render_trail_frame,
    )
    dx = 0.13
    copy_qpos = []
    for i, t in enumerate(idxs):
        qf = q[int(t)].copy()
        qf[0] = i * dx
        qf[1] = 0.0
        copy_qpos.append(qf)
    mid_x = (len(idxs) - 1) * dx / 2.0
    mean_z = float(np.mean([qf[2] for qf in copy_qpos]))
    # Aim above the bodies so the camera tilts up and the floor drops toward
    # (and below) the bottom edge — less ground in frame.
    lookat = np.array([mid_x, 0.0, max(mean_z, 0.03) + 0.10])
    segs = [(np.array([[-0.05, 0.0, 0.002],
                       [(len(idxs) - 1) * dx + 0.05, 0.0, 0.002]]),
             [0.5, 0.5, 0.5, 0.22])]
    fovy = 50.0
    row_w = (len(idxs) - 1) * dx + 0.12
    cam_dist = max(0.9 * row_w / (2 * math.tan(math.radians(fovy) / 2)), 0.34)
    model, base_nq = build_parade_trail_model(
        env, num_copies=len(idxs), trace_segments=segs,
        camera_lookat=lookat, camera_distance=cam_dist,
        camera_elevation=8.0, camera_azimuth=-90.0, camera_fovy=fovy)
    recolor_copies(model, colors)
    return render_trail_frame(model, base_nq, copy_qpos, width=width, height=height)


def _render_kick_strip(run_dir: Path, examples: list[tuple], t_perturb: int,
                       checkpoint: str, n_copies: int = 5) -> np.ndarray | None:
    """Two (or more) code examples side by side, each a grey→red→grey row of
    snapshots: baseline (grey) → shoved (red) → recovered (grey)."""
    try:
        from c2a.figures.render_codebook_atlas import build_env
        from c2a.experiments.shared.code_grid import add_banner
        root = get_repo_root()
        env = build_env(str(root / checkpoint))
        single = len(examples) == 1
        rwidth = 1750 if single else 1050
        rows = []
        for cid, amp, label in examples:
            q_all = np.asarray(_load(run_dir, cid, amp, "qpos"), dtype=np.float64)
            body, idxs = _select_body_frames(q_all, t_perturb, n_copies)
            q = q_all[body]
            cpost = max(idxs[-1] - t_perturb, 1)
            colors = [_phase_color(int(t), t_perturb, cpost) for t in idxs]
            img = _crop_dark(_render_row(env, q, idxs, colors, width=rwidth))
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
    film = _render_kick_strip(run_dir, examples, t_perturb, args.checkpoint)

    # Manual layout so panel (a) keeps the rendered image's EXACT aspect
    # (no distortion), spans the full width, and stays compact; the two
    # quantitative panels sit in a row beneath it.
    fig_w = 12.0
    left, right = 0.07, 0.98
    axw = right - left                       # axes width (figure fraction)
    if film is not None:
        img_h, img_w = film.shape[:2]
        a_h_in = (axw * fig_w) * img_h / img_w   # height preserving aspect
    else:
        a_h_in = 1.8
    bottom_h_in = 3.1                        # height of the (b)/(c) row
    pad = 0.55
    fig_h = a_h_in + bottom_h_in + 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    a_h = a_h_in / fig_h
    b_h = bottom_h_in / fig_h
    top = 0.95
    ax_film = fig.add_axes([left, top - a_h, axw, a_h])
    by = top - a_h - (pad / fig_h) - b_h
    ax_curve = fig.add_axes([left, by, 0.40, b_h])
    ax_basin = fig.add_axes([0.57, by, 0.40, b_h])

    # (a) ghost trail
    ax_film.set_axis_off()
    if film is not None:
        ax_film.imshow(film)

    # (b) recovery curves at args.amp — shared standardization so the
    # foreign-attractor control is on the same scale.
    ai = int(np.argmin(np.abs(np.array(amps) - args.amp)))
    amp = amps[ai]
    t_ax = grid / 100.0   # seconds (100 Hz control)
    mean, std = _global_stats(run_dir, codes)
    clouds = {c: _cloud(run_dir, c, mean, std, cloud_idx, rng) for c in codes}
    pre = grid < t_perturb
    for c in codes:
        pca, nn = clouds[c]
        d = _dist_to(run_dir, c, amp, pca, nn, mean, std, grid)
        floor = max(float(np.median(d[pre])), 1e-9)
        ax_curve.plot(t_ax, d / floor, lw=2.2,
                      color=BEHAVIOR_COLORS.get(code_beh[c], "k"),
                      label=f"{code_beh[c]} (code {c})")
    # code-specificity control: a recovering body measured against a
    # DIFFERENT code's attractor (normalized by its OWN floor) never
    # approaches the foreign attractor — it stays well above 1. We pick the
    # cleanest-recovering body and the behaviourally most different attractor
    # available (a locomotor vs. a held-posture code).
    beh_to_code = {code_beh[c]: c for c in codes}
    ctrl_src = beh_to_code.get("Walk", codes[0])
    ctrl_foreign = beh_to_code.get("Immobile",
                                   wrong.get(ctrl_src, codes[-1]))
    if ctrl_foreign == ctrl_src:
        ctrl_foreign = wrong[ctrl_src]
    pca_o, nn_o = clouds[ctrl_src]
    d_o = _dist_to(run_dir, ctrl_src, amp, pca_o, nn_o, mean, std, grid)
    floor_o = max(float(np.median(d_o[pre])), 1e-9)
    pca_w, nn_w = clouds[ctrl_foreign]
    d_w = _dist_to(run_dir, ctrl_src, amp, pca_w, nn_w, mean, std, grid)
    ax_curve.plot(t_ax, d_w / floor_o, lw=1.8, ls="--", color="k",
                  label=f"{code_beh[ctrl_src]} body → {code_beh[ctrl_foreign]} "
                        "attractor (control)")
    ax_curve.axvline(t_perturb / 100.0, color="red", lw=1.0, ls=":")
    ax_curve.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax_curve.set_xlabel("time (s)")
    ax_curve.set_ylabel(r"distance to attractor  $d(t)/d_{\mathrm{floor}}$")
    ax_curve.legend(fontsize=10, loc="upper right")

    # (c) basin
    for c in codes:
        m = summary["codes"][str(c)]["spaces"]["pose"]["metrics"]
        fr = [mm["recovery_frac"] for mm in m]
        xs = [a for a, f in zip(amps, fr) if f is not None]
        ys = [f for f in fr if f is not None]
        ax_basin.plot(xs, ys, marker="o", lw=2.0,
                      color=BEHAVIOR_COLORS.get(code_beh[c], "k"),
                      label=f"{code_beh[c]}")
    ax_basin.set_ylim(0, 1.05)
    ax_basin.set_xlabel("impulse amplitude (qvel std, rad/s | m/s)")
    ax_basin.set_ylabel("recovery fraction")
    ax_basin.legend(fontsize=10, loc="lower left")

    # Parenthesized bold panel labels, aligned to each graph's own left edge
    # (x=0 in axes coords) so (a) and (b) line up with the plot edges.
    ax_film.text(0.0, 1.05, "(a)", transform=ax_film.transAxes,
                 fontsize=15, fontweight="bold", va="bottom")
    ax_curve.text(0.0, 1.04, "(b)", transform=ax_curve.transAxes,
                  fontsize=15, fontweight="bold", va="bottom")
    ax_basin.text(0.0, 1.04, "(c)", transform=ax_basin.transAxes,
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
