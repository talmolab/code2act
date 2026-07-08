"""Shared overlay renderer for the presentation demo videos (``c2a/demos``).

Animates ``N`` rodent bodies overlaid in the clean **light arena** (the
``grid_gt_transitions`` aesthetic: white/grey diamond floor, soft shadows, a
3/4 angled camera), driven by precomputed per-body ``qpos`` trajectories.

The heavy lifting reuses the ghost-overlay primitive
(:func:`c2a.rendering_utils.build_ghost_model`) so the **clean light arena**
(default MuJoCo skybox + textured diamond floor, soft shadows, the
``divergent_cam`` 3/4 view) is identical to ``grid_gt_transitions`` and the
population-grid figures; this module adds the per-frame animation loop, a
per-(body, frame) color hook, and tiling helpers.

The light floor comes from rendering through ``env._spec`` (textured floor
material, ``matid >= 0``); only the walker geoms (base bodies suffixed
``-rodent``, ghost bodies suffixed ``-ghost{i}``) carry ``matid == -1`` and so
respond to per-frame ``geom_rgba`` recoloring — the floor is never touched.

Typical use::

    env = build_env("model_checkpoints/260603_180852_140475")
    panel = render_overlay_panel(env, [q0, q1, ...], color_fn)   # (T, H, W, 3)
    video = tile_horizontal([panel_a, panel_b], ["rear", "walk"])
    write_video(video, out_path, fps=20)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from c2a.paths import get_repo_root
from c2a.experiments.shared.code_grid import add_banner, stack_panels_into_grid
from c2a.figures.render_behavior_parade_trail import recolor_copies

log = logging.getLogger(__name__)

# RGBA used by the reference grid_gt_transitions figure (teal bodies):
# RGB [30, 128, 146] normalized.
TEAL_RGBA: list[float] = [0.12, 0.50, 0.57, 1.0]

ColorFn = Callable[[int, int], Sequence[float]]
"""``color_fn(body_idx, frame_idx) -> rgba`` (4 floats in [0, 1])."""


# ── Env construction ─────────────────────────────────────────────────────────


def build_env(ckpt_rel: str):
    """Build a ``MoSeqImitation`` env from a strategy's decoder checkpoint.

    Only the walker XML / rescale are used for trail rendering, so the codes
    that seed the env are immaterial; this delegates to the codebook-atlas
    builder which already resolves a valid ``best_codes.npz``.

    Args:
        ckpt_rel: checkpoint dir, relative to the repo root (or absolute).

    Returns:
        A ``MoSeqImitation`` env exposing ``mj_model`` / ``_walker_xml_path``.
    """
    from c2a.figures.render_codebook_atlas import build_env as _build_env

    root = get_repo_root()
    ckpt_path = ckpt_rel if os.path.isabs(ckpt_rel) else str(root / ckpt_rel)
    return _build_env(ckpt_path)


# ── Trajectory helpers ───────────────────────────────────────────────────────


def anchor_xy(
    qpos: np.ndarray,
    ref_frame: int | None = 0,
    ref_slice: slice | None = None,
) -> np.ndarray:
    """Recenter a ``(T, nq)`` trajectory so a reference root x/y sits at origin.

    Subtracting a per-body world-translation makes several bodies overlap at a
    common local origin so their motion reads as displacement from there (the
    population-grid convention). Height (z) and all joint angles are untouched.

    Args:
        qpos: ``(T, nq)`` trajectory.
        ref_frame: subtract this frame's x/y (used when ``ref_slice`` is None).
        ref_slice: if given, subtract the **mean** x/y over this window instead
            (centers the whole motion around the origin — handy for translating
            codes like walk so the bodies don't drift out of frame).
    """
    out = np.array(qpos, dtype=np.float64, copy=True)
    if ref_slice is not None:
        off = out[ref_slice, :2].mean(0)
    else:
        off = out[ref_frame, :2]
    out[:, 0] -= off[0]
    out[:, 1] -= off[1]
    return out


def _look_quat(cam_pos: np.ndarray, look_target: np.ndarray) -> list[float]:
    """Camera quaternion (w, x, y, z) pointing ``cam_pos`` at ``look_target``."""
    forward = look_target - cam_pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    R = np.stack([right, up, -forward], axis=1)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    s = 0.5 / np.sqrt(max(tr + 1.0, 1e-9))
    return [
        0.25 / s, (R[2, 1] - R[1, 2]) * s,
        (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s,
    ]


def build_white_overlay_model(
    env,
    n_bodies: int,
    *,
    camera_distance: float,
    camera_elevation: float,
    camera_azimuth: float,
    camera_fovy: float,
    look_z: float = 0.06,
):
    """Build an overlay model: ``n_bodies`` walkers + a 3/4 cam.

    The arena is the USUAL :data:`ARENA_XML_PATH` (the default dark-blue checker
    environment used by ``build_parade_trail_model`` and seen in the prior
    project videos), NOT the pure-white arena (which read as flat/weird). It
    ships its own floor + skybox, so no floor plane is added here. An overhead
    shadow light + headlight fill keep the (teal) bodies clearly lit. Walker
    copies are suffixed ``-t{i}`` so :func:`recolor_copies` can tint each frame.

    Returns ``(mj_model, base_nq)``.
    """
    from vnl_playground.tasks import utils as vnl_utils
    from vnl_playground.tasks.rodent import consts as rodent_consts
    from c2a.rendering_utils import _disable_lights_recursive

    # Usual environment background (dark-blue checker floor + skybox).
    spec = mujoco.MjSpec.from_file(str(rodent_consts.ARENA_XML_PATH))

    walker_path = str(env._walker_xml_path)
    rescale = env.reference_clips._config["model"]["SCALE_FACTOR"]
    for i in range(n_bodies):
        walker = mujoco.MjSpec.from_file(walker_path)
        if rescale != 1.0:
            walker = vnl_utils.dm_scale_spec(walker, rescale)
        if i > 0:
            _disable_lights_recursive(walker.worldbody)
        frame = spec.worldbody.add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
        gb = frame.attach_body(walker.body("walker"), "", suffix=f"-t{i}")
        gb.add_freejoint(name=f"root-t{i}")

    look_target = np.array([0.0, 0.0, look_z])
    el_rad = np.radians(camera_elevation)
    az_rad = np.radians(camera_azimuth)
    cam_pos = look_target + np.array([
        camera_distance * np.cos(el_rad) * np.cos(az_rad),
        camera_distance * np.cos(el_rad) * np.sin(az_rad),
        -camera_distance * np.sin(el_rad),
    ])
    spec.worldbody.add_camera(
        name="overlay_cam", pos=cam_pos.tolist(),
        quat=_look_quat(cam_pos, look_target), fovy=camera_fovy,
    )

    light = spec.worldbody.add_light()
    light.name = "overlay_overhead"
    light.pos = [0.0, 0.0, 3.0]
    light.dir = [0.0, 0.0, -1.0]
    light.diffuse = [0.5, 0.5, 0.5]
    light.specular = [0.1, 0.1, 0.1]
    light.castshadow = True

    model = spec.compile()
    base_nq = model.nq // n_bodies

    # Gentle headlight fill so body faces are lit, but keep the dark-blue arena
    # looking like the usual environment (don't wash the floor toward grey).
    model.vis.headlight.active = 1
    model.vis.headlight.ambient = [0.3, 0.3, 0.3]
    model.vis.headlight.diffuse = [0.4, 0.4, 0.4]
    model.vis.headlight.specular = [0.1, 0.1, 0.1]
    return model, base_nq


# ── Core overlay panel ───────────────────────────────────────────────────────


def render_overlay_panel(
    env,
    body_qpos: Sequence[np.ndarray],
    color_fn: ColorFn,
    *,
    camera_distance: float | None = None,
    camera_elevation: float = -25.0,
    camera_azimuth: float = -135.0,
    camera_fovy: float = 50.0,
    width: int = 1080,
    height: int = 960,
    pad: float = 0.30,
) -> np.ndarray:
    """Animate ``N`` overlaid bodies on a WHITE arena -> ``(T, H, W, 3)`` uint8.

    Bodies should be pre-anchored (see :func:`anchor_xy`) so their motion is
    centered near the origin, which is where the 3/4 camera looks.

    Args:
        env: a ``MoSeqImitation`` (for walker XML / rescale).
        body_qpos: list of ``(T, nq)`` trajectories, one per body. ``T`` is the
            min length across bodies.
        color_fn: ``color_fn(body_idx, frame_idx) -> rgba`` recolors each body
            every frame (constant colors are fine — ignore ``frame_idx``).
        camera_distance: explicit distance; auto-derived from the bodies'
            spatial extent when ``None``.
        camera_elevation / camera_azimuth / camera_fovy: the 3/4 view.
        width / height: per-panel render size.
        pad: extra metres around the body extent for the auto camera.

    Returns:
        ``(T, height, width, 3)`` uint8 video frames (white background).
    """
    n_bodies = len(body_qpos)
    T = min(int(q.shape[0]) for q in body_qpos)

    pos = np.stack([q[:T, :3] for q in body_qpos], axis=0)  # (N, T, 3)
    if camera_distance is None:
        flat = pos.reshape(-1, 3)
        # robust span about the origin: the 92nd pct planar radius (x2) so a
        # single body that drifts far (e.g. a walk code) doesn't shrink the view.
        radius = np.linalg.norm(flat[:, :2], axis=1)
        reach = float(np.percentile(radius, 92)) * 2.0
        fovy_rad = np.radians(camera_fovy)
        camera_distance = max((reach + pad) / (2 * np.tan(fovy_rad / 2)), 0.5)

    model, base_nq = build_white_overlay_model(
        env, n_bodies,
        camera_distance=float(camera_distance),
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
        camera_fovy=camera_fovy,
    )
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = np.empty((T, height, width, 3), dtype=np.uint8)
    try:
        for t in range(T):
            recolor_copies(
                model, [list(color_fn(i, t)) for i in range(n_bodies)]
            )
            for i, q in enumerate(body_qpos):
                data.qpos[i * base_nq:(i + 1) * base_nq] = q[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="overlay_cam")
            frames[t] = renderer.render()
    finally:
        renderer.close()
    return frames


# ── Tiling ───────────────────────────────────────────────────────────────────


def _banner_panels(
    panels: Sequence[np.ndarray],
    titles: Sequence[str] | None,
    banner_h: int,
    font_size: int,
) -> list[np.ndarray]:
    if not titles:
        return list(panels)
    out = []
    for panel, title in zip(panels, titles):
        T = panel.shape[0]
        framed = np.stack(
            [add_banner(panel[t], title, banner_h=banner_h,
                        bg=(245, 245, 245), fg=(20, 20, 20),
                        font_size=font_size)
             for t in range(T)],
            axis=0,
        )
        out.append(framed)
    return out


def tile_horizontal(
    panels: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    *,
    banner_h: int = 44,
    font_size: int = 30,
    gap: int = 8,
    gap_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Tile per-body panels side by side (each ``(T, H, W, 3)``) -> one video.

    Optional ``titles`` are drawn as a banner above each panel. A thin
    background gap separates panels.
    """
    panels = _banner_panels(panels, titles, banner_h, font_size)
    T = panels[0].shape[0]
    H = max(p.shape[1] for p in panels)
    panels = [_pad_to_height(p, H, gap_color) for p in panels]
    sep = np.empty((T, H, gap, 3), dtype=np.uint8)
    sep[:] = np.array(gap_color, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            pieces.append(sep)
        pieces.append(p)
    return np.concatenate(pieces, axis=2)


def tile_grid(
    panels: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    cols: int,
    grid_title: str = "",
    banner_h: int = 38,
    font_size: int = 24,
) -> np.ndarray:
    """Tile per-cell panels into a ``ceil(N/cols) x cols`` grid video.

    Each cell gets its own ``titles[i]`` banner; ``grid_title`` is drawn once at
    the top. Panels are padded to a common size and blank cells fill the last
    row. Returns ``(T, gridH, gridW, 3)`` uint8.
    """
    panels = _banner_panels(panels, titles, banner_h, font_size)
    T = panels[0].shape[0]
    H = max(p.shape[1] for p in panels)
    W = max(p.shape[2] for p in panels)
    panels = [_pad_to_size(p, H, W) for p in panels]
    rows = int(np.ceil(len(panels) / cols))
    blank = np.full((T, H, W, 3), 255, dtype=np.uint8)
    while len(panels) < rows * cols:
        panels.append(blank)
    return stack_panels_into_grid(
        list(panels), rows=rows, cols=cols, grid_title=grid_title,
        grid_banner_h=64 if grid_title else 0,
    )


def _pad_to_height(p: np.ndarray, H: int, color) -> np.ndarray:
    if p.shape[1] == H:
        return p
    T, h, W, _ = p.shape
    out = np.full((T, H, W, 3), 255, dtype=np.uint8)
    out[:] = np.array(color, dtype=np.uint8)
    top = (H - h) // 2
    out[:, top:top + h] = p
    return out


def _pad_to_size(p: np.ndarray, H: int, W: int) -> np.ndarray:
    if p.shape[1] == H and p.shape[2] == W:
        return p
    T = p.shape[0]
    out = np.full((T, H, W, 3), 255, dtype=np.uint8)
    top = (H - p.shape[1]) // 2
    left = (W - p.shape[2]) // 2
    out[:, top:top + p.shape[1], left:left + p.shape[2]] = p
    return out


# ── Video writing ────────────────────────────────────────────────────────────


def write_video(frames: np.ndarray, out_path: str | Path, fps: int = 20) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames to an MP4 (yuv420p, even dims)."""
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 + yuv420p needs even dimensions; pad if necessary.
    T, H, W, _ = frames.shape
    if H % 2 or W % 2:
        Hp, Wp = H + (H % 2), W + (W % 2)
        padded = np.full((T, Hp, Wp, 3), 255, dtype=np.uint8)
        padded[:, :H, :W] = frames
        frames = padded
    writer = imageio.get_writer(
        str(out_path), fps=fps, codec="libx264",
        macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    log.info("wrote %s  (%d frames, %dx%d)", out_path, frames.shape[0],
             frames.shape[2], frames.shape[1])
