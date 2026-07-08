"""Demo video: held codes are attractors the controller actively defends.

Per steering strategy, one video with **two panels side by side**
(``recovery from rear`` | ``recovery from walk``). Each panel overlays **5
bodies** animated through a velocity-impulse kick and the controller's recovery,
phase-colored **grey → red → grey** (on the attractor → shoved → recovered) on a
clean WHITE arena.

Data: the per-strategy ``perturbation_dir`` written by
``c2a.experiments.perturbation.run_perturbation_recovery``
(``data/rollout_code<cid>_amp<amp>.npz`` with ``qpos`` of shape ``(K, T, nq)``,
plus ``perturbation_summary.json`` giving ``t_perturb`` and each code's
behavior). The Rear code and the Walk code are read from the summary; a mid
impulse amplitude (default ``1.2``) is used.

``mlb`` is skipped until its checkpoint / data exist (it is filtered out by
``strategies.available("perturbation_dir")``).

Usage::

    C2A_ROOT="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
      python3.12 bash_scripts/_pyrun.py c2a.demos.kick_recovery_demo
    # ... --strategy kpms --amp 1.2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

from c2a.paths import get_repo_root
from c2a.demos import demo_render as DR
from c2a.demos.strategies import STRATEGIES, available

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Phase coloring: teal on the attractor / recovered (matches grid_gt_transitions
# so the held body is clearly visible against the white floor), red at the shove.
_TEAL = [0.12, 0.50, 0.57]
_RED = [0.90, 0.16, 0.16]


def _phase_color_teal(t: int, tp: int, post: int, alpha: float = 1.0
                      ) -> list[float]:
    """Teal before the impulse, full red at it, fading red→teal as it recovers."""
    if t < tp:
        return _TEAL + [alpha]
    u = float(np.clip((t - tp) / max(post, 1), 0.0, 1.0))
    return [_RED[k] * (1 - u) + _TEAL[k] * u for k in range(3)] + [alpha]


def _rollout_path(run_dir: Path, cid: int, amp: float) -> Path:
    return run_dir / "data" / f"rollout_code{cid:03d}_amp{amp:g}.npz"


def _select_recovering_bodies(
    q_all: np.ndarray, t_perturb: int, n: int = 5
) -> list[int]:
    """Pick ``n`` bodies that recover cleanly back to the code's attractor.

    A body recovers well when its late-window root height returns to the median
    pre-impulse (attractor) height. Returns the ``n`` best such body indices
    (behavior-agnostic: rear settles high, walk settles low — both judged by
    deviation from their own settled height).
    """
    T = q_all.shape[1]
    pre = slice(max(0, t_perturb - 120), t_perturb)
    late = slice(min(t_perturb + 260, T - 1), T)
    attractor_z = float(np.median(q_all[:, pre, 2].mean(axis=1)))
    late_z = q_all[:, late, 2].mean(axis=1)
    order = np.argsort(np.abs(late_z - attractor_z))
    return [int(b) for b in order[:n]]


def _render_panel(
    env, run_dir: Path, cid: int, amp: float, t_perturb: int,
    *, n_bodies: int, win_pre: int, step: int,
    width: int, height: int,
) -> np.ndarray:
    """One code's teal→red→teal 5-body recovery panel.

    The window spans ``[t_perturb - win_pre, T]`` — generous pre-kick context so
    sustained motion (e.g. walking) is visible before the shove, the kick, then
    the full recovery. Bodies are anchored to the kick frame so the impulse sits
    at the origin where the camera looks.
    """
    q_all = np.asarray(np.load(_rollout_path(run_dir, cid, amp))["qpos"],
                       dtype=np.float64)
    T = q_all.shape[1]
    bodies_idx = _select_recovering_bodies(q_all, t_perturb, n_bodies)
    start = max(0, t_perturb - win_pre)
    frames_abs = np.arange(start, T, step)
    kick_local = int(np.argmin(np.abs(frames_abs - t_perturb)))  # anchor frame

    bodies = [
        DR.anchor_xy(q_all[b, frames_abs], ref_frame=kick_local)
        for b in bodies_idx
    ]
    post_span = max(T - t_perturb, 1)

    def color_fn(i: int, t: int):
        return _phase_color_teal(int(frames_abs[t]), t_perturb, post_span)

    log.info("    code %d amp %g: bodies %s, %d frames",
             cid, amp, bodies_idx, len(frames_abs))
    return DR.render_overlay_panel(
        env, bodies, color_fn, width=width, height=height,
        camera_elevation=-22.0, camera_azimuth=-120.0, camera_fovy=48.0,
    )


def run_strategy(name: str, amp: float, fps: int) -> Path | None:
    cfg = STRATEGIES[name]
    root = get_repo_root()
    run_dir = root / cfg.perturbation_dir
    summary = json.loads((run_dir / "perturbation_summary.json").read_text())
    t_perturb = int(summary["t_perturb"])
    if amp not in summary["amplitudes"]:
        amp = min(summary["amplitudes"][1:],
                  key=lambda a: abs(a - amp))  # nearest non-zero
    codes = summary["codes"]
    beh_to_cid = {info["behavior"]: int(cid) for cid, info in codes.items()}
    log.info("[%s] t_perturb=%d amp=%g code map=%s", name, t_perturb, amp,
             beh_to_cid)

    panels, titles = [], []
    for beh, title in [("Rear", "recovery from rear"),
                       ("Walk", "recovery from walk")]:
        if beh not in beh_to_cid:
            log.warning("[%s] no %s code in summary; skipping panel", name, beh)
            continue
        env = DR.build_env(cfg.policy_ckpt)
        panel = _render_panel(
            env, run_dir, beh_to_cid[beh], amp, t_perturb,
            n_bodies=5, win_pre=300, step=6,
            width=1120, height=1040,
        )
        panels.append(panel)
        titles.append(title)

    if not panels:
        log.error("[%s] no panels rendered", name)
        return None

    video = DR.tile_horizontal(panels, titles, banner_h=48, font_size=34)
    out = root / "outputs" / "demos" / "kick_recovery" / f"{name}.mp4"
    DR.write_video(video, out, fps=fps)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default=None,
                   help="one strategy (kpms|kvae); default = all available")
    p.add_argument("--amp", type=float, default=1.2,
                   help="impulse amplitude (snaps to nearest available)")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    names = [args.strategy] if args.strategy else available("perturbation_dir")
    log.info("strategies: %s", names)
    for name in names:
        out = run_strategy(name, args.amp, args.fps)
        if out:
            log.info("[%s] wrote %s", name, out)


if __name__ == "__main__":
    main()
