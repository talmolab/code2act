"""Paper-style trail videos for the Code2Act project page.

The manuscript's static renders (codebook atlas, perturbation filmstrip,
unobserved-pair grid) show one body as a frozen-ghost motion trail in the
dark checkerboard arena, colored by behavior. This module renders the same
look as video: a live body, a fading trail of ghost copies sampled from its
recent past, and a thin root trace, under a camera that follows the body.

The scene is compiled and the renderer created once per video, so rendering
is a per-frame qpos write plus ``mj_forward``.

Requires a MuJoCo GL backend (EGL on headless GPUs).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from vnl_playground.tasks import utils as vnl_utils

from c2a.codegen.shared.mlb_labels import MLB_COLORS, MLB_LABELS
from c2a.paths import get_repo_root
from c2a.rendering_utils import _disable_lights_recursive

#: Checkpoint used only to build the environment (walker XML and rescale);
#: every steering strategy drives the same body.
ENV_CKPT = "model_checkpoints/260629_212713_669229"
MLB_CODES = "outputs/tokenizer/mlb_v1/best_codes.npz"

#: Control rate of the saved rollouts (Hz).
CTRL_HZ = 100

_DIR_FULL = {"L": "Left", "R": "Right", "S": "Straight"}


def behavior_rgb(label: str) -> tuple[float, float, float]:
    """Paper palette color for a directional label.

    Accepts the MLB names (``Walk_Left``) and the short rollout labels
    (``Walk_L``, ``Immobile_S``). Labels with no palette entry (a straight
    turn) take the mean of their behavior's shades.
    """
    if label in MLB_COLORS:
        return tuple(MLB_COLORS[label])
    base, _, d = label.partition("_")
    if base == "Immobile":
        return tuple(MLB_COLORS["Immobile"])
    key = f"{base}_{_DIR_FULL.get(d, d)}"
    if key in MLB_COLORS:
        return tuple(MLB_COLORS[key])
    shades = [v for k, v in MLB_COLORS.items() if k.startswith(base + "_")]
    if shades:
        return tuple(np.mean(shades, axis=0))
    return (0.6, 0.6, 0.6)


def mlb_rgb(code: int) -> tuple[float, float, float]:
    """Paper palette color for an MLB code id."""
    return tuple(MLB_COLORS[MLB_LABELS[int(code)]])


def build_env():
    """Environment used only for the walker XML and its rescale factor."""
    from c2a.figures.render_codebook_atlas import build_env as _build

    root = get_repo_root()
    return _build(str(root / ENV_CKPT), str(root / MLB_CODES))


@dataclass
class Scene:
    """A compiled multi-walker scene plus its reusable renderer."""

    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: mujoco.Renderer
    base_nq: int
    n_copies: int
    copy_geoms: list[np.ndarray]
    light_id: int

    def close(self) -> None:
        self.renderer.close()


def build_scene(env, n_copies: int, width: int, height: int) -> Scene:
    """Compile the arena with ``n_copies`` walkers and a renderer.

    Copy 0 is the live body; the others are trail ghosts. Lighting matches
    ``build_parade_trail_model`` (one overhead light plus a headlight fill).
    """
    from vnl_playground.tasks.rodent import consts as rodent_consts

    spec = mujoco.MjSpec.from_file(str(rodent_consts.ARENA_XML_PATH))
    rescale = env.reference_clips._config["model"]["SCALE_FACTOR"]
    for i in range(n_copies):
        walker = mujoco.MjSpec.from_file(str(env._walker_xml_path))
        if rescale != 1.0:
            walker = vnl_utils.dm_scale_spec(walker, rescale)
        if i > 0:
            _disable_lights_recursive(walker.worldbody)
        frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
        body = frame.attach_body(walker.body("walker"), "", suffix=f"-t{i}")
        body.add_freejoint(name=f"root-t{i}")

    light = spec.worldbody.add_light(name="follow_overhead")
    light.pos = [0, 0, 3.0]
    light.dir = [0, 0, -1]
    light.diffuse = [0.9, 0.9, 0.9]
    light.specular = [0.2, 0.2, 0.2]

    model = spec.compile()
    model.vis.headlight.active = 1
    model.vis.headlight.ambient = [0.55, 0.55, 0.55]
    model.vis.headlight.diffuse = [0.45, 0.45, 0.45]
    model.vis.headlight.specular = [0.1, 0.1, 0.1]
    for i in range(model.nlight):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i) or ""
        if "tracking_light" in name:
            model.light_diffuse[i] = [0, 0, 0]
            model.light_specular[i] = [0, 0, 0]
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)

    # Geoms belonging to each copy, for per-copy recoloring.
    import re

    pats = [re.compile(rf"-t{ci}(?!\d)") for ci in range(n_copies)]
    owner = np.full(model.ngeom, -1)
    for gi in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        bname = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gi]))
            or ""
        )
        for ci, p in enumerate(pats):
            if p.search(gname + "\x00" + bname):
                owner[gi] = ci
                break
    copy_geoms = [np.flatnonzero(owner == ci) for ci in range(n_copies)]
    for g in copy_geoms:
        model.geom_matid[g] = -1

    base_nq = model.nq // n_copies
    light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "follow_overhead")
    renderer = mujoco.Renderer(model, height=height, width=width)
    return Scene(model, mujoco.MjData(model), renderer, base_nq, n_copies,
                 copy_geoms, light_id)


def smooth_track(xy: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average of a ``(T, 2)`` track (edge-padded)."""
    if window <= 1:
        return xy
    pad = window // 2
    p = np.pad(xy, ((pad, window - 1 - pad), (0, 0)), mode="edge")
    k = np.ones(window) / window
    return np.stack([np.convolve(p[:, i], k, mode="valid") for i in range(2)], 1)


@dataclass
class TrailStyle:
    """Look of a trail video.

    Attributes:
        n_ghosts: Ghost copies drawn behind the live body.
        ghost_every: Control steps between consecutive ghosts.
        ghost_alpha: Opacity of the most recent ghost; older ones fade
            linearly toward ``ghost_alpha_min``.
        trace_steps: Length of the root trace, in control steps (0 = off).
        trace_width: Root trace radius in meters.
        cam_distance: Camera distance in meters.
        cam_elevation: Camera elevation in degrees.
        cam_azimuth: Camera azimuth in degrees.
        cam_fovy: Vertical field of view in degrees.
        follow_window: Moving-average window (steps) for the camera target.
        lookat_z: Height of the camera target.
    """

    n_ghosts: int = 5
    ghost_every: int = 25
    ghost_alpha: float = 0.45
    ghost_alpha_min: float = 0.12
    trace_steps: int = 300
    trace_width: float = 0.0025
    cam_distance: float = 0.75
    cam_elevation: float = -22.0
    cam_azimuth: float = -130.0
    cam_fovy: float = 40.0
    follow_window: int = 120
    lookat_z: float = 0.05


def _add_trace(scene: mujoco.MjvScene, pts: np.ndarray, rgbas: np.ndarray,
               width: float) -> None:
    """Append a polyline of capsules to a rendered scene."""
    for j in range(len(pts) - 1):
        if scene.ngeom >= scene.maxgeom:
            return
        a, b = pts[j], pts[j + 1]
        if np.linalg.norm(b - a) < 1e-5:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.zeros(9), rgbas[j].astype(np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scene.ngeom += 1


def render_trail_video(
    scene: Scene,
    qpos: np.ndarray,
    colors: np.ndarray,
    style: TrailStyle,
    frames: range,
    alpha_override: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Render one body's trajectory as a trail video.

    Args:
        scene: Scene from :func:`build_scene` with ``style.n_ghosts + 1`` copies.
        qpos: ``(T, nq)`` trajectory of a single body.
        colors: ``(T, 3)`` RGB in [0, 1] per control step; ghosts take the
            color of the step they were sampled at, so phase changes stay
            visible along the trail.
        style: Trail and camera settings.
        frames: Control steps to render (e.g. ``range(0, T, 2)`` for 50 fps
            playback of a 100 Hz rollout).
        alpha_override: Optional ``(T,)`` opacity for the live body.

    Returns:
        List of ``(H, W, 3)`` uint8 frames.
    """
    T, nq = qpos.shape
    if nq != scene.base_nq:
        raise ValueError(f"qpos has nq={nq}, scene walker has {scene.base_nq}")
    track = smooth_track(qpos[:, :2].astype(np.float64), style.follow_window)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = style.cam_distance
    cam.elevation = style.cam_elevation
    cam.azimuth = style.cam_azimuth
    scene.model.vis.global_.fovy = style.cam_fovy
    hidden_pos = np.zeros(nq)
    hidden_pos[2] = -5.0  # park unused ghosts below the floor
    hidden_pos[3] = 1.0

    out = []
    for t in frames:
        q = scene.data.qpos
        times = [t] + [t - k * style.ghost_every for k in range(1, scene.n_copies)]
        for ci, tt in enumerate(times):
            sl = slice(ci * nq, (ci + 1) * nq)
            if tt < 0:
                q[sl] = hidden_pos
                continue
            q[sl] = qpos[tt]
            if ci == 0:
                a = 1.0 if alpha_override is None else float(alpha_override[tt])
            else:
                f = (ci - 1) / max(1, scene.n_copies - 2)
                a = style.ghost_alpha + f * (style.ghost_alpha_min - style.ghost_alpha)
            scene.model.geom_rgba[scene.copy_geoms[ci]] = [*colors[tt], a]
        mujoco.mj_forward(scene.model, scene.data)

        look = np.array([track[t, 0], track[t, 1], style.lookat_z])
        cam.lookat[:] = look
        scene.model.light_pos[scene.light_id] = [look[0], look[1], 3.0]
        scene.renderer.update_scene(scene.data, camera=cam)
        if style.trace_steps > 0:
            t0 = max(0, t - style.trace_steps)
            pts = qpos[t0:t + 1:2, :3].astype(np.float64).copy()
            pts[:, 2] = 0.003
            if len(pts) > 1:
                fade = np.linspace(0.15, 0.95, len(pts))[:, None]
                rgba = np.concatenate([colors[t0:t + 1:2][: len(pts)], fade], 1)
                _add_trace(scene.renderer.scene, pts, rgba, style.trace_width)
        out.append(scene.renderer.render().copy())
    return out


def write_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 50,
              crf: int = 23) -> Path:
    """Encode frames as web-friendly H.264 (yuv420p, faststart)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=16, quality=None,
        output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"],
    )
    for f in frames:
        w.append_data(f)
    w.close()
    return path


def write_poster(frame: np.ndarray, path: str | Path, quality: int = 85) -> Path:
    """Save a JPEG poster frame for a video."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), frame, quality=quality)
    return path
