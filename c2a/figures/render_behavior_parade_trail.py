"""Behavior transition parade as a motion trail: walk -> groom -> rear.

Renders one body performing the consecutive walk -> groom -> rear transition
with frozen copies colored by behavior phase, trajectory trace line, and
a gait cycle timeline bar.

Data: outputs/moseq_behavior_parade/data/parade_rollouts.npz

Usage:
    cd moseq_jax/figures
    python render_behavior_parade_trail.py
    python render_behavior_parade_trail.py --n-copies-per-beh 4
"""

import re
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path


from c2a.paths import get_c2a_root, get_repo_root
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).parent
MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()
OUTPUT_DIR = get_repo_root() / "outputs" / "figures" / "behavior_parade_trail"

from vnl_playground.tasks import utils as vnl_utils
from c2a.rendering_utils import _disable_lights_recursive
from c2a.codegen.shared.label_alignment import BEHAVIOR_TO_INT
from c2a.codegen.shared.mlb_labels import MLB_SHORT
from c2a.experiments.shared.code_labels import nine_class_label_for_code
from c2a.experiments.shared.plotting import nine_class_rgb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Per-steering-strategy checkpoint + codes for the env used to build the trail
# walker (the rendered motion itself comes from the saved parade rollout, so the
# codes only seed the env; the walker XML / rescale are what matter). Override
# any field on the command line.
RUN_CONFIG: dict[str, dict[str, str]] = {
    "kpms": {
        "checkpoint": "model_checkpoints/260407_031233_484020",
        "codes": "outputs/tokenizer/kpms_sweep/best_codes.npz",
    },
    "kvae": {
        "checkpoint": "model_checkpoints/260603_180852_140475",
        "codes": ("outputs/tokenizer/trained_kinematics_vae_ckpts/vae_sweep/"
                  "kl_0p001/tokenize_k25/best_codes.npz"),
    },
    "mlb": {
        "checkpoint": "model_checkpoints/260629_212713_669229",
        "codes": "outputs/tokenizer/mlb_v1/best_codes.npz",
    },
}

# parade behavior name (as written by run_behavior_parade) -> BEHAVIORS index.
# c2a's 4-class corpus folds grooming into Immobile.
_PHASE_BEH_TO_INT: dict[str, int] = {
    "walk": BEHAVIOR_TO_INT["Walk"],
    "rear": BEHAVIOR_TO_INT["Rear"],
    "turn": BEHAVIOR_TO_INT["Turn"],
    "groom": BEHAVIOR_TO_INT["Immobile"],
    "immobile": BEHAVIOR_TO_INT["Immobile"],
}

# Behavior-parade rollouts live under the repo-root outputs/ tree (never
# c2a/outputs), one subdir per behavior sequence written by
# run_behavior_parade. Select the sequence to render here.
PARADE_SEQUENCE = "walk_groom_rear"
_PARADE_DIR = (
    get_repo_root() / "outputs" / "behavior" / "moseq_behavior_parade"
    / PARADE_SEQUENCE
)
PARADE_DATA = _PARADE_DIR / "data" / "parade_rollouts.npz"
PARADE_META = _PARADE_DIR / "code_selection.json"

BEHAVIOR_ORDER = ["rear", "walk", "groom"]
BEHAVIOR_LABELS = {"walk": "Walk", "groom": "Immobility", "rear": "Rear"}


def _quat_to_yaw(quat):
    """Yaw from a (w,x,y,z) quaternion array (..., 4)."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def phase_nine(beh_name: str, qpos_phase: np.ndarray,
               deadband: float = 0.10) -> tuple[str, tuple[float, float, float]]:
    """9-class label + RGB for one parade phase (D3).

    The phase's coarse behavior is taken from its name; the kinematic direction
    (Left / Right / Straight, or forced L/R for Turn) is derived from the
    exemplar body's net heading change over the phase, matching the mlb / kpms /
    kvae 9-class auto-labeling convention.

    Args:
        beh_name: parade behavior name (e.g. ``"walk"``, ``"groom"``).
        qpos_phase: ``(T_phase, nq)`` exemplar-body qpos for this phase.

    Returns:
        ``(label, rgb)`` where ``label`` is a 9-class category string.
    """
    b_int = _PHASE_BEH_TO_INT.get(beh_name.lower(), -1)
    yaw = np.unwrap(_quat_to_yaw(qpos_phase[:, 3:7]))
    net = float(yaw[-1] - yaw[0]) if yaw.size >= 2 else 0.0
    label, _code, _d = nine_class_label_for_code(
        b_int, [net], [float(len(qpos_phase))], deadband=deadband
    )
    return label, nine_class_rgb(label)


def select_parade_exemplar(qpos_all, boundaries):
    """Pick the body that best performs the whole parade sequence.

    Each phase contributes a z-scored behavior-discriminative kinematic (planar
    speed for walk, root height for rear, yaw rate for turn, stillness for
    groom/immobile); the body with the highest summed score across phases is the
    clearest exemplar of the full walk/groom/rear sequence.
    """
    K = qpos_all.shape[0]
    if K <= 1:
        return 0

    def _z(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-9 else a * 0.0

    total = np.zeros(K)
    for beh, s, e in boundaries:
        if e <= s:
            continue
        seg = qpos_all[:, s:e]
        speed = np.linalg.norm(np.diff(seg[:, :, :2], axis=1), axis=2).mean(1)
        height = seg[:, :, 2].mean(1)
        yaw = np.unwrap(_quat_to_yaw(seg[:, :, 3:7]), axis=1)
        yaw_rate = np.abs(np.diff(yaw, axis=1)).mean(1)
        motion = speed + 0.1 * seg[:, :, 7:].std(1).mean(1)
        b = beh.lower()
        if "walk" in b:
            total += _z(speed)
        elif "rear" in b:
            total += _z(height)
        elif "turn" in b:
            total += _z(yaw_rate)
        else:  # groom / immobile
            total += _z(-motion)
    return int(np.argmax(total))

# Behavior colors (matching experiments/shared/plotting.py)
BEHAVIOR_RGB = {
    "walk": (0.836, 0.369, 0.0),      # #D55E00 orange
    "groom": (0.0, 0.447, 0.698),     # #0072B2 blue
    "rear": (0.0, 0.620, 0.451),      # #009E73 green
}
BEHAVIOR_RGBA = {
    beh: [*rgb, 1.0] for beh, rgb in BEHAVIOR_RGB.items()
}
FPS = 50

LIMBS = ["FL", "FR", "HL", "HR"]
LIMB_FOOT_BODY = {"HL": "toe_L", "HR": "toe_R", "FL": "finger_L", "FR": "finger_R"}
STANCE_THRESHOLD = 0.005


# ── Helpers ─────────────────────────────────────────────────────────────────


def compute_foot_contacts(
    env,
    qpos_traj: np.ndarray,
    threshold: float = STANCE_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Compute per-limb stance (True) / swing (False) from FK."""
    model = env.mj_model
    data = mujoco.MjData(model)
    suffix = getattr(env, "_suffix", "")

    foot_ids = {}
    for limb, body_name in LIMB_FOOT_BODY.items():
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name + suffix
        )
        if bid < 0:
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
        foot_ids[limb] = bid

    T = len(qpos_traj)
    contacts = {limb: np.zeros(T, dtype=bool) for limb in LIMBS}
    for t in range(T):
        data.qpos[:] = qpos_traj[t]
        mujoco.mj_forward(model, data)
        for limb, bid in foot_ids.items():
            contacts[limb][t] = data.xpos[bid, 2] < threshold

    return contacts


def recolor_copies(
    model: mujoco.MjModel,
    copy_rgbas: list[list[float]],
) -> None:
    """Recolor each copy's body geoms. copy_rgbas[i] applies to suffix -t{i}.

    Matches the per-copy suffix on the geom name or its parent body name. The
    suffix must be followed by a non-digit (regex boundary) -- a plain substring
    test would match ``-t1`` inside ``-t10``/``-t11``, miscoloring copies 10/11
    as copy 1 (the orange-headed reared body bug).
    """
    pats = [re.compile(rf"-t{ci}(?!\d)") for ci in range(len(copy_rgbas))]
    for gi in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        bid = int(model.geom_bodyid[gi])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        tag = gname + "\x00" + bname
        for ci, rgba in enumerate(copy_rgbas):
            if pats[ci].search(tag):
                model.geom_rgba[gi] = rgba
                model.geom_matid[gi] = -1
                break


def timestep_to_behavior(
    t: int,
    boundaries: list[tuple[str, int, int]],
) -> str:
    """Return behavior name for a given timestep."""
    for beh, start, end in boundaries:
        if start <= t < end:
            return beh
    return boundaries[-1][0]


def _phase_of(t: int, boundaries: list[tuple[str, int, int]]) -> int:
    """Index of the parade phase containing timestep ``t`` (last on overrun)."""
    for pi, (_beh, start, end) in enumerate(boundaries):
        if start <= t < end:
            return pi
    return len(boundaries) - 1


# ── Trail model construction ────────────────────────────────────────────────


def build_parade_trail_model(
    env,
    num_copies: int,
    trace_segments: list[tuple[np.ndarray, list[float]]],
    trace_radius: float = 0.003,
    camera_lookat: np.ndarray | None = None,
    camera_distance: float = 1.0,
    camera_elevation: float = -25.0,
    camera_azimuth: float = -135.0,
    camera_fovy: float = 45.0,
) -> tuple[mujoco.MjModel, int]:
    """Build trail model with multi-colored trace line for behavior phases."""
    from vnl_playground.tasks.rodent import consts as rodent_consts

    arena_xml = str(rodent_consts.ARENA_XML_PATH)
    walker_path = str(env._walker_xml_path)
    rescale = env.reference_clips._config["model"]["SCALE_FACTOR"]

    spec = mujoco.MjSpec.from_file(arena_xml)

    for i in range(num_copies):
        walker = mujoco.MjSpec.from_file(walker_path)
        if rescale != 1.0:
            walker = vnl_utils.dm_scale_spec(walker, rescale)
        if i > 0:
            _disable_lights_recursive(walker.worldbody)
        frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
        gb = frame.attach_body(walker.body("walker"), "", suffix=f"-t{i}")
        gb.add_freejoint(name=f"root-t{i}")

    seg_idx = 0
    for positions, rgba in trace_segments:
        for j in range(len(positions) - 1):
            p1 = positions[j].tolist()
            p2 = positions[j + 1].tolist()
            if np.linalg.norm(positions[j + 1] - positions[j]) < 1e-6:
                continue
            spec.worldbody.add_geom(
                name=f"trace_{seg_idx}",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[*p1, *p2],
                size=[trace_radius, 0, 0],
                rgba=rgba,
                contype=0, conaffinity=0,
            )
            seg_idx += 1

    all_positions = np.concatenate([p for p, _ in trace_segments], axis=0)
    if camera_lookat is None:
        camera_lookat = all_positions.mean(axis=0)
        # center horizontally on the content span (not the point-density mean,
        # which is pulled toward the dense walk trace -> off-center framing)
        camera_lookat[0] = 0.5 * (all_positions[:, 0].min()
                                  + all_positions[:, 0].max())
        camera_lookat[1] = 0.5 * (all_positions[:, 1].min()
                                  + all_positions[:, 1].max())
        camera_lookat[2] = max(camera_lookat[2], 0.06)

    el_rad = np.radians(camera_elevation)
    az_rad = np.radians(camera_azimuth)
    cam_pos = camera_lookat + np.array([
        camera_distance * np.cos(el_rad) * np.cos(az_rad),
        camera_distance * np.cos(el_rad) * np.sin(az_rad),
        -camera_distance * np.sin(el_rad),
    ])

    forward = camera_lookat - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    s = 0.5 / np.sqrt(tr + 1.0)
    cam_quat = [
        0.25 / s, (R[2, 1] - R[1, 2]) * s,
        (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s,
    ]

    spec.worldbody.add_camera(
        name="trail_cam", pos=cam_pos.tolist(),
        quat=cam_quat, fovy=camera_fovy,
    )

    overhead = spec.worldbody.add_light(name="trail_overhead")
    overhead.pos = [camera_lookat[0], camera_lookat[1], 3.0]
    overhead.dir = [0, 0, -1]
    overhead.diffuse = [0.9, 0.9, 0.9]
    overhead.specular = [0.2, 0.2, 0.2]

    model = spec.compile()
    base_nq = model.nq // num_copies

    # A single overhead light leaves a near-horizontal camera (e.g. the
    # perturbation filmstrip) looking at the bodies' shadowed front faces, which
    # reads as very dark. Add a global ambient/headlight fill so every face is
    # lit regardless of camera angle.
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


def render_trail_frame(
    trail_model: mujoco.MjModel,
    base_nq: int,
    copy_qpos_list: list[np.ndarray],
    width: int = 2048,
    height: int = 768,
) -> np.ndarray:
    """Set qpos for each copy and render a single frame."""
    trail_model.vis.global_.offwidth = width
    trail_model.vis.global_.offheight = height
    data = mujoco.MjData(trail_model)
    renderer = mujoco.Renderer(trail_model, height=height, width=width)

    for i, qp in enumerate(copy_qpos_list):
        start = i * base_nq
        data.qpos[start:start + base_nq] = qp

    mujoco.mj_forward(trail_model, data)
    renderer.update_scene(data, camera="trail_cam")
    frame = renderer.render().copy()
    renderer.close()
    return frame


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Render behavior transition parade as motion trail"
    )
    parser.add_argument("--run", default="kvae",
                        help="steering strategy: kpms | kvae | mlb "
                             "(selects checkpoint + codes)")
    parser.add_argument("--checkpoint", default=None,
                        help="override the strategy's decoder checkpoint path")
    parser.add_argument("--codes", default=None,
                        help="override the strategy's best_codes.npz path")
    parser.add_argument("--parade-dir", default=None,
                        help="parade rollout dir (with data/parade_rollouts.npz "
                             "and code_selection.json); default is the legacy "
                             f"{PARADE_SEQUENCE} dir")
    parser.add_argument("--body-idx", type=int, default=-1,
                        help="Which body from the parade to use; -1 (default) "
                             "auto-selects the body that best performs the "
                             "whole sequence")
    parser.add_argument("--n-copies-per-beh", type=int, default=4,
                        help="Frozen copies per behavior phase")
    parser.add_argument("--phase-spacing", type=float, default=0.0,
                        help="If >0, re-center each behavior phase along x at "
                             "this spacing (metres) so phases read as distinct "
                             "left-to-right groups instead of overlapping. "
                             "0 = use natural trajectory positions.")
    parser.add_argument("--trace-step", type=int, default=3,
                        help="Trace line resolution: connect every N frames")
    parser.add_argument("--width", type=int, default=2048,
                        help="Render width in pixels")
    parser.add_argument("--height", type=int, default=768,
                        help="Render height in pixels")
    parser.add_argument("--cam-dist", type=float, default=0.0,
                        help="Camera distance (0 = auto)")
    parser.add_argument("--cam-elev", type=float, default=-25.0)
    parser.add_argument("--cam-azim", type=float, default=-135.0)
    parser.add_argument("--cam-fovy", type=float, default=45.0)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load env ─────────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from c2a.moseq_env_wrapper import MoSeqImitation

    rcfg = RUN_CONFIG.get(args.run, RUN_CONFIG["kvae"])
    ckpt_rel = args.checkpoint or rcfg["checkpoint"]
    codes_rel = args.codes or rcfg["codes"]
    ckpt_path = str(REPO_ROOT / ckpt_rel)
    log.info(f"run={args.run}  checkpoint={ckpt_rel}  codes={codes_rel}")
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

    codes_data = np.load(REPO_ROOT / codes_rel)
    test_codes = codes_data["test_codes"]

    test_clips = ReferenceClips(
        data_path=str(REPO_ROOT / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    env = MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes)
    log.info("Environment loaded")

    # ── Load parade data ─────────────────────────────────────────────
    if args.parade_dir:
        parade_dir = Path(args.parade_dir)
        parade_data = parade_dir / "data" / "parade_rollouts.npz"
        parade_meta = parade_dir / "code_selection.json"
    else:
        parade_data, parade_meta = PARADE_DATA, PARADE_META
    log.info(f"parade data: {parade_data}")
    d = np.load(parade_data, allow_pickle=True)
    qpos_all = np.asarray(d["qpos"], dtype=np.float64)  # (K, T, nq)

    with open(parade_meta) as f:
        selection = json.load(f)

    # Phase boundaries must follow the ACTUAL temporal order of the rollout
    # (behavior_names/starts/ends saved by run_behavior_parade), not the
    # hardcoded BEHAVIOR_ORDER -- otherwise colors/labels get scrambled.
    if "behavior_names" in d:
        names = [str(x) for x in d["behavior_names"]]
        starts = [int(x) for x in np.asarray(d["behavior_starts"]).tolist()]
        ends = [int(x) for x in np.asarray(d["behavior_ends"]).tolist()]
        boundaries = list(zip(names, starts, ends))
    else:
        boundaries = []
        offset = 0
        for beh in BEHAVIOR_ORDER:
            if beh in selection:
                beh_len = len(selection[beh]["code_sequence"])
                boundaries.append((beh, offset, offset + beh_len))
                offset += beh_len

    # Pick the body that best performs the full sequence unless overridden.
    if args.body_idx is not None and args.body_idx >= 0:
        body_idx = int(args.body_idx)
    else:
        body_idx = select_parade_exemplar(qpos_all, boundaries)
    qpos = qpos_all[body_idx]
    log.info(f"Parade body {body_idx} of {qpos_all.shape[0]}: {qpos.shape}")

    # Per-phase 9-class color + label (D3): behavior from the phase name,
    # direction from the exemplar body's net heading change over the phase.
    phase_color: dict[int, tuple[float, float, float]] = {}
    phase_label: dict[int, str] = {}
    for pi, (beh, start, end) in enumerate(boundaries):
        lab, rgb = phase_nine(beh, qpos[start:end])
        phase_color[pi] = rgb
        phase_label[pi] = MLB_SHORT.get(lab, lab)
        log.info(f"  {beh}: frames {start}-{end}  -> {lab}")

    # ── Compute copy timesteps and trace segments ────────────────────
    all_trail_indices = []
    copy_phase = []                 # phase index per copy
    raw_trace = []                  # (positions, rgba, phase_idx)

    for pi, (beh, beh_start, beh_end) in enumerate(boundaries):
        phase_indices = np.linspace(
            beh_start, beh_end - 1, args.n_copies_per_beh
        ).round().astype(int)
        all_trail_indices.extend(phase_indices.tolist())
        copy_phase.extend([pi] * len(phase_indices))

        trace_timesteps = np.arange(beh_start, beh_end, args.trace_step)
        trace_pos = qpos[trace_timesteps, :3].copy()
        trace_pos[:, 2] += 0.005
        raw_trace.append((trace_pos, [*phase_color[pi], 0.85], pi))

    total_copies = len(all_trail_indices)
    log.info(f"Total copies: {total_copies} at timesteps: {all_trail_indices}")

    # Per-copy qpos; optionally re-center each phase along x so stationary
    # phases (immobile/rear) don't pile on top of the walk's endpoint.
    copy_qpos = [qpos[t].copy() for t in all_trail_indices]
    phase_dx = {pi: 0.0 for pi in range(len(boundaries))}
    if args.phase_spacing > 0:
        for pi in range(len(boundaries)):
            xs = [copy_qpos[k][0] for k in range(len(copy_phase))
                  if copy_phase[k] == pi]
            centroid = float(np.mean(xs)) if xs else 0.0
            phase_dx[pi] = pi * args.phase_spacing - centroid
        for k in range(len(copy_qpos)):
            copy_qpos[k][0] += phase_dx[copy_phase[k]]
        for trace_pos, _rgba, pi in raw_trace:
            trace_pos[:, 0] += phase_dx[pi]
        # Bridge the gaps between offset phases so the trace reads as one
        # continuous line (walk -> immobile -> rear), not just within walk.
        # Each bridge is a colour gradient from the previous phase's colour to
        # the next phase's, drawn as several short sub-segments.
        bridges = []
        n_sub = 10
        for pi in range(len(raw_trace) - 1):
            end_pt = raw_trace[pi][0][-1].copy()
            start_pt = raw_trace[pi + 1][0][0].copy()
            c0 = np.array(phase_color[pi])
            c1 = np.array(phase_color[pi + 1])
            pts = np.linspace(end_pt, start_pt, n_sub + 1)
            for s in range(n_sub):
                frac = (s + 0.5) / n_sub
                col = ((1 - frac) * c0 + frac * c1).tolist()
                bridges.append((pts[s:s + 2], [*col, 0.85], -1))
        raw_trace.extend(bridges)
        log.info(f"  phase x-offsets: "
                 f"{[round(phase_dx[pi], 3) for pi in range(len(boundaries))]}")
    trace_segments = [(tp, rgba) for tp, rgba, _pi in raw_trace]

    # Auto-compute camera distance from the (possibly offset) copy positions
    positions = np.stack([qp[:3] for qp in copy_qpos], axis=0)
    extent = positions.max(axis=0) - positions.min(axis=0)
    max_span = max(extent[0], extent[1])
    fovy_rad = math.radians(args.cam_fovy)
    cam_dist = max(
        (max_span + 0.2) / (2 * math.tan(fovy_rad / 2)),
        0.35,
    )
    if args.cam_dist > 0:
        cam_dist = args.cam_dist
    log.info(f"  extent={max_span:.3f}m, cam_dist={cam_dist:.3f}m")

    # ── Build trail model and render ─────────────────────────────────
    trail_model, base_nq = build_parade_trail_model(
        env,
        num_copies=total_copies,
        trace_segments=trace_segments,
        camera_distance=cam_dist,
        camera_elevation=args.cam_elev,
        camera_azimuth=args.cam_azim,
        camera_fovy=args.cam_fovy,
    )

    # Recolor each copy by its phase's 9-class color
    copy_colors = []
    for t in all_trail_indices:
        copy_colors.append([*phase_color[_phase_of(t, boundaries)], 1.0])
    recolor_copies(trail_model, copy_colors)

    # copy_qpos already built above (with optional per-phase x offsets)
    frame = render_trail_frame(
        trail_model, base_nq, copy_qpos,
        width=args.width, height=args.height,
    )
    log.info(f"Rendered trail: {frame.shape}")

    # ── Compute foot contacts ────────────────────────────────────────
    log.info("Computing foot contacts...")
    contacts = compute_foot_contacts(env, qpos)

    # ── Assemble figure: behavior bar + trail + gait diagram ─────────
    total_frames = sum(end - start for _, start, end in boundaries)
    total_sec = total_frames / FPS
    n_limbs = len(LIMBS)

    fig = plt.figure(figsize=(13.6, 5.7))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1, 13, 4],
        hspace=0.05,
    )

    # Top: behavior phase bar (9-class colored, 9-class short labels)
    ax_bar_top = fig.add_subplot(gs[0])
    for pi, (beh, beh_start, beh_end) in enumerate(boundaries):
        ax_bar_top.barh(
            0, (beh_end - beh_start) / FPS, left=beh_start / FPS,
            height=1, color=phase_color[pi], edgecolor="none",
        )
        ax_bar_top.text(
            (beh_start + beh_end) / 2 / FPS, 0,
            phase_label[pi], ha="center", va="center",
            fontsize=15, fontweight="bold", color="white",
        )
    ax_bar_top.set_xlim(0, total_sec)
    ax_bar_top.set_ylim(-0.5, 0.5)
    ax_bar_top.set_yticks([])
    ax_bar_top.set_xticks([])
    for sp in ax_bar_top.spines.values():
        sp.set_visible(False)

    # Middle: trail render
    ax_img = fig.add_subplot(gs[1])
    ax_img.imshow(frame)
    ax_img.set_axis_off()

    # Bottom: real gait diagram (stance/swing per limb)
    ax_gait = fig.add_subplot(gs[2])

    def _draw_stance(li, t0, t1, pidx):
        ax_gait.fill_between(
            [t0 / FPS, t1 / FPS], li - 0.4, li + 0.4,
            color=phase_color[pidx], alpha=0.7, edgecolor="none",
        )

    for li, limb in enumerate(LIMBS):
        stance = contacts[limb]
        in_stance = False
        start_t = 0
        cur_phase = None
        for t in range(len(stance)):
            phase_t = _phase_of(t, boundaries) if stance[t] else None
            if stance[t] and not in_stance:
                start_t, cur_phase, in_stance = t, phase_t, True
            elif in_stance and (not stance[t] or phase_t != cur_phase):
                # close run at the swing onset OR the phase boundary, so a
                # foot held across a phase boundary is split and recolored
                _draw_stance(li, start_t, t, cur_phase)
                if stance[t]:               # phase changed, still planted
                    start_t, cur_phase = t, phase_t
                else:
                    in_stance = False
        if in_stance:
            _draw_stance(li, start_t, len(stance), cur_phase)

    # Behavior phase boundaries
    for _, beh_start, beh_end in boundaries:
        if beh_start > 0:
            ax_gait.axvline(
                beh_start / FPS, color="black", linewidth=0.8,
                linestyle="--", alpha=0.5,
            )

    # Copy position markers
    for t in all_trail_indices:
        ax_gait.axvline(
            t / FPS, color="black", linewidth=0.5, alpha=0.3,
            linestyle=":",
        )

    ax_gait.set_xlim(0, total_sec)
    ax_gait.set_ylim(-0.6, n_limbs - 0.4)
    ax_gait.set_yticks(range(n_limbs))
    ax_gait.set_yticklabels(LIMBS, fontsize=10)
    ax_gait.set_xlabel("Time (s)", fontsize=13)
    ax_gait.tick_params(axis="x", labelsize=10)
    ax_gait.tick_params(axis="y", length=0)
    ax_gait.spines["top"].set_visible(False)
    ax_gait.spines["right"].set_visible(False)
    ax_gait.spines["left"].set_visible(False)
    ax_gait.invert_yaxis()

    rsuf = "" if args.run == "kvae" else f"_{args.run}"
    out_base = OUTPUT_DIR / f"behavior_parade_trail{rsuf}"
    for ext in (".pdf", ".png"):
        fig.savefig(
            str(out_base) + ext, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.02,
        )
    plt.close(fig)
    log.info(f"Saved: {out_base}.pdf / .png")


if __name__ == "__main__":
    main()
