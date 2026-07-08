"""Demo 2 — animated behavior-parade motion video (kpms + kvae side by side).

One body executes the consecutive ``walk -> immobile (groom) -> rear`` code
sequence as a *video*: the body is animated frame-by-frame in the clean dark
arena, colored by the 9-class color of the current behavior phase, with the
gait-stance strip (FL / FR / HL / HR foot contacts) drawn beneath and a
playhead that advances with the frame. The two steering strategies are laid out
side by side with strategy banners.

Unlike :mod:`c2a.figures.render_behavior_parade_trail` (a *static* frozen-ghost
trail PDF), this renders a moving single body. It reuses that module's env
construction, exemplar selection, per-phase 9-class coloring, foot-contact
computation, and the dark-arena trail model.

Reference: ``Code2Act-website/public/behavior_parade_side.mp4``.

Data: per-strategy ``parade_dir`` from :mod:`c2a.demos.strategies` (a behavior
parade rollout dir with ``data/parade_rollouts.npz`` + ``code_selection.json``).
Built for the strategies whose ``parade_dir`` is set (kpms + kvae); ``mlb``
drops in automatically once its data path is wired in ``strategies.py``.

Outputs:
    outputs/demos/parade_motion/<strategy>.mp4   (per strategy)
    outputs/demos/parade_motion/parade_side.mp4  (combined side-by-side)

Usage:
    C2A_ROOT="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python3.12 bash_scripts/_pyrun.py c2a.demos.parade_motion_demo
    # smoke (few frames, one strategy):
    ... c2a.demos.parade_motion_demo --strategy kpms --n-frames 6
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
import mujoco
import numpy as np

from c2a.demos.strategies import STRATEGIES, available
from c2a.experiments.shared.code_grid import add_banner
from c2a.figures.render_behavior_parade_trail import (
    FPS,
    LIMBS,
    RUN_CONFIG,
    compute_foot_contacts,
    phase_nine,
    recolor_copies,
    render_trail_frame,
    select_parade_exemplar,
    _phase_of,
)
from c2a.paths import get_repo_root

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
REPO_ROOT = get_repo_root()
OUTPUT_DIR = REPO_ROOT / "outputs" / "demos" / "parade_motion"

# Phase label shown on the body overlay / gait strip.
PHASE_DISPLAY = {"walk": "Walk", "groom": "Immobility", "immobile": "Immobility",
                 "rear": "Rear", "turn": "Turn"}


def _load_env(strategy: str):
    """Build the ``MoSeqImitation`` env for one strategy (env-only, no policy).

    Mirrors :func:`c2a.figures.render_behavior_parade_trail.main`'s env
    construction: the rendered motion comes from the saved parade rollout, so
    the env exists only to supply the (correct) walker XML / rescale and the
    arena model. Checkpoint + codes come from that module's ``RUN_CONFIG``.
    """
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from c2a.moseq_env_wrapper import MoSeqImitation

    rcfg = RUN_CONFIG[strategy]
    ckpt_path = str(REPO_ROOT / rcfg["checkpoint"])
    log.info(f"[{strategy}] env from ckpt={rcfg['checkpoint']} codes={rcfg['codes']}")
    cfg = checkpointing.load_config_from_checkpoint(
        ckpt_path, step_prefix="MoSeqPPONetwork"
    )
    cfg = OmegaConf.create(cfg)
    _, _, env_cfg = utils.prepare_config(cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False

    splits_path = REPO_ROOT / "data" / "rodent" / "rodent_balanced_splits.json"
    with open(splits_path) as f:
        splits = json.load(f)
    test_indices = splits["balanced"]["test_indices"]

    codes_data = np.load(REPO_ROOT / rcfg["codes"])
    test_codes = codes_data["test_codes"]

    test_clips = ReferenceClips(
        data_path=str(REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    return MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes)


def _load_parade(parade_dir: Path):
    """Load parade qpos + phase boundaries; return ``(qpos_exemplar, boundaries)``.

    ``boundaries`` is ``[(name, start, end), ...]`` in temporal order, taken from
    the saved ``behavior_names/starts/ends`` arrays.
    """
    d = np.load(parade_dir / "data" / "parade_rollouts.npz", allow_pickle=True)
    qpos_all = np.asarray(d["qpos"], dtype=np.float64)  # (K, T, nq)
    names = [str(x) for x in d["behavior_names"]]
    starts = [int(x) for x in np.asarray(d["behavior_starts"]).tolist()]
    ends = [int(x) for x in np.asarray(d["behavior_ends"]).tolist()]
    boundaries = list(zip(names, starts, ends))
    body_idx = select_parade_exemplar(qpos_all, boundaries)
    log.info(f"  parade body {body_idx}/{qpos_all.shape[0]}, T={qpos_all.shape[1]}")
    return qpos_all[body_idx], boundaries


def _build_body_model(
    env,
    camera_lookat: np.ndarray,
    camera_distance: float,
    camera_elevation: float,
    camera_azimuth: float,
    camera_fovy: float,
) -> tuple[mujoco.MjModel, int]:
    """Single-body model in the default (dark checker) arena (suffix ``-t0``).

    Mirrors :func:`render_behavior_parade_trail.build_parade_trail_model`
    (same ``ARENA_XML_PATH`` dark arena as ``behavior_parade_side.mp4``) but for
    a single moving body and without the frozen-trail capsules — this demo
    animates one body frame by frame. Returns ``(model, base_nq)``.
    """
    from vnl_playground.tasks.rodent import consts as rodent_consts
    from vnl_playground.tasks import utils as vnl_utils

    arena_xml = str(rodent_consts.ARENA_XML_PATH)
    walker_path = str(env._walker_xml_path)
    rescale = env.reference_clips._config["model"]["SCALE_FACTOR"]

    spec = mujoco.MjSpec.from_file(arena_xml)
    walker = mujoco.MjSpec.from_file(walker_path)
    if rescale != 1.0:
        walker = vnl_utils.dm_scale_spec(walker, rescale)
    frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
    gb = frame.attach_body(walker.body("walker"), "", suffix="-t0")
    gb.add_freejoint(name="root-t0")

    el_rad = np.radians(camera_elevation)
    az_rad = np.radians(camera_azimuth)
    cam_pos = camera_lookat + np.array([
        camera_distance * np.cos(el_rad) * np.cos(az_rad),
        camera_distance * np.cos(el_rad) * np.sin(az_rad),
        -camera_distance * np.sin(el_rad),
    ])
    forward = camera_lookat - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)
    s = 0.5 / np.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1.0)
    cam_quat = [0.25 / s, (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
    spec.worldbody.add_camera(name="trail_cam", pos=cam_pos.tolist(),
                              quat=cam_quat, fovy=camera_fovy)

    overhead = spec.worldbody.add_light(name="trail_overhead")
    overhead.pos = [camera_lookat[0], camera_lookat[1], 3.0]
    overhead.dir = [0, 0, -1]
    overhead.diffuse = [0.9, 0.9, 0.9]
    overhead.specular = [0.2, 0.2, 0.2]

    model = spec.compile()
    base_nq = model.nq
    # Ambient/headlight fill so every body face is lit regardless of cam angle.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse = [0.45, 0.45, 0.45]
    model.vis.headlight.specular = [0.1, 0.1, 0.1]
    for i in range(model.nlight):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i)
        if name and "tracking_light" in name:
            model.light_diffuse[i] = [0, 0, 0]
            model.light_specular[i] = [0, 0, 0]
    return model, base_nq


def _render_gait_strip(
    contacts: dict[str, np.ndarray],
    boundaries: list[tuple[str, int, int]],
    phase_color: dict[int, tuple[float, float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Static gait-stance strip (no playhead) as an ``(height, width, 3)`` RGB.

    Each limb row is filled where the foot is in stance, colored by the 9-class
    color of the behavior phase at that time. Phase boundaries are dashed.
    """
    total_T = boundaries[-1][2]
    total_sec = total_T / FPS
    n = len(LIMBS)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0.06, 0.22, 0.92, 0.74])

    def _draw(li, t0, t1, pidx):
        ax.fill_between([t0 / FPS, t1 / FPS], li - 0.42, li + 0.42,
                        color=phase_color[pidx], alpha=0.85, edgecolor="none")

    for li, limb in enumerate(LIMBS):
        stance = contacts[limb]
        in_st, start_t, cur = False, 0, None
        for t in range(len(stance)):
            ph = _phase_of(t, boundaries) if stance[t] else None
            if stance[t] and not in_st:
                start_t, cur, in_st = t, ph, True
            elif in_st and (not stance[t] or ph != cur):
                _draw(li, start_t, t, cur)
                if stance[t]:
                    start_t, cur = t, ph
                else:
                    in_st = False
        if in_st:
            _draw(li, start_t, len(stance), cur)

    for _, bs, _be in boundaries:
        if bs > 0:
            ax.axvline(bs / FPS, color="0.35", lw=1.0, ls="--", alpha=0.7)

    ax.set_xlim(0, total_sec)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(LIMBS, fontsize=13, color="0.15")
    ax.set_xlabel("Time (s)", fontsize=13, color="0.15")
    ax.tick_params(axis="x", labelsize=7, colors="0.15")
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("0.6")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if img.shape[:2] != (height, width):
        import cv2
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def _overlay_playhead(strip: np.ndarray, frac: float, pad_x: int) -> np.ndarray:
    """Draw a vertical playhead on a copy of the gait strip at fraction ``frac``."""
    out = strip.copy()
    H, W = out.shape[:2]
    x0 = pad_x
    x1 = W - 4
    px = int(x0 + frac * (x1 - x0))
    px = min(max(px, x0), x1)
    out[:, max(0, px - 1):px + 2] = [210, 30, 30]
    return out


def render_strategy(strategy: str, n_frames: int, fps: int,
                    panel_w: int, panel_h: int, cam_dist: float) -> list[np.ndarray]:
    """Render one strategy's parade-motion frames (body panel + gait strip).

    Returns a list of ``n_frames`` ``(H, W, 3)`` uint8 frames (banner + body +
    gait strip).
    """
    cfg = STRATEGIES[strategy]
    parade_dir = REPO_ROOT / cfg.parade_dir
    env = _load_env(strategy)
    qpos, boundaries = _load_parade(parade_dir)
    T = qpos.shape[0]

    # Per-phase 9-class color + display label.
    phase_color: dict[int, tuple[float, float, float]] = {}
    phase_disp: dict[int, str] = {}
    for pi, (beh, s, e) in enumerate(boundaries):
        lab, rgb = phase_nine(beh, qpos[s:e])
        phase_color[pi] = rgb
        phase_disp[pi] = PHASE_DISPLAY.get(beh.lower(), beh.capitalize())
        log.info(f"  phase {pi} {beh}: frames {s}-{e} -> {lab} rgb={tuple(round(c,2) for c in rgb)}")

    # Foot contacts over the full sequence.
    log.info(f"[{strategy}] computing foot contacts ...")
    contacts = compute_foot_contacts(env, qpos)

    # Single-body default dark-arena model (suffix -t0), camera looking at the
    # origin; we center the body's xy each frame so it stays framed while the
    # floor conveys motion.
    trail_model, base_nq = _build_body_model(
        env, camera_lookat=np.array([0.0, 0.0, 0.085]),
        camera_distance=cam_dist, camera_elevation=-18.0,
        camera_azimuth=-130.0, camera_fovy=45.0,
    )

    # Evenly sample n_frames source frames across the full sequence (phases are
    # equal-length thirds in both strategies, so this aligns phases across the
    # two side-by-side panels).
    src_idx = np.linspace(0, T - 1, n_frames).round().astype(int)

    gait_h = max(120, panel_h // 3)
    gait_pad_x = int(0.06 * panel_w)
    strip_base = _render_gait_strip(contacts, boundaries, phase_color,
                                    panel_w, gait_h)

    frames: list[np.ndarray] = []
    last_pi = -1
    for fi, t in enumerate(src_idx):
        pi = _phase_of(int(t), boundaries)
        if pi != last_pi:
            recolor_copies(trail_model, [[*phase_color[pi], 1.0]])
            last_pi = pi
        qp = qpos[int(t)].copy()
        qp[0] -= qpos[int(t), 0]
        qp[1] -= qpos[int(t), 1]
        body = render_trail_frame(trail_model, base_nq, [qp],
                                  width=panel_w, height=panel_h)
        strip = _overlay_playhead(strip_base, fi / max(n_frames - 1, 1), gait_pad_x)
        cell = np.concatenate([body, strip], axis=0)
        # Dark banner (readable over the white arena) carries strategy + phase.
        cell = add_banner(
            cell, f"{cfg.display}   |   Behavior: {phase_disp[pi]}   t={int(t)}",
            banner_h=46, bg=(25, 25, 25), fg=(245, 245, 245), font_size=26,
        )
        frames.append(cell)
        if (fi + 1) % max(1, n_frames // 6) == 0:
            log.info(f"  [{strategy}] frame {fi + 1}/{n_frames}")
    return frames


def _write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Encode frames to H.264 mp4 (even dims) via imageio's ffmpeg plugin."""
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames[0].shape[:2]
    H2, W2 = H - (H % 2), W - (W % 2)
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p", "-crf", "23"],
    )
    for f in frames:
        writer.append_data(f[:H2, :W2])
    writer.close()
    log.info(f"Saved video: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default=None,
                    help="single strategy (default: all available parade_dir).")
    ap.add_argument("--n-frames", type=int, default=180,
                    help="output frames per strategy (phases align across both).")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--panel-w", type=int, default=720)
    ap.add_argument("--panel-h", type=int, default=540)
    ap.add_argument("--cam-dist", type=float, default=0.55)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategies = ([args.strategy] if args.strategy
                  else available("parade_dir"))
    log.info(f"Strategies: {strategies}")

    rendered: dict[str, list[np.ndarray]] = {}
    for s in strategies:
        log.info(f"=== Rendering parade motion: {s} ===")
        frames = render_strategy(
            s, args.n_frames, args.fps, args.panel_w, args.panel_h, args.cam_dist,
        )
        rendered[s] = frames
        _write_video(frames, OUTPUT_DIR / f"{s}.mp4", args.fps)

    # Combined side-by-side (truncate to common frame count; all panels equal H).
    if len(rendered) >= 2:
        order = [s for s in available("parade_dir") if s in rendered]
        n = min(len(rendered[s]) for s in order)
        H = min(rendered[s][0].shape[0] for s in order)
        combined = []
        for fi in range(n):
            cells = [rendered[s][fi][:H] for s in order]
            combined.append(np.concatenate(cells, axis=1))
        _write_video(combined, OUTPUT_DIR / "parade_side.mp4", args.fps)

    log.info(f"=== Done. Outputs under {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    main()
