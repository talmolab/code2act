"""Experiment: 2-panel videos comparing real RL rollouts under live vs fixed codes.

For each of 5 selected ``Rear → Walk → Rear`` clips:

* **LEFT** — env rollout where the e2e policy uses its live VQ-VAE to
  encode each step's reference-future window into a code (so the codes
  evolve with the policy's behavior, and reflect what the e2e training
  has learned the codebook to *mean*).
* **RIGHT** — env rollout where the policy's e2e VQ-VAE encoder is
  bypassed and the policy is forced to act on the fixed code sequence
  from the pretrained tokenizer (``best_codes.npz``). Same MuJoCo env,
  same physics; only the code feeding the action decoder differs.

Both rollouts share the same env config (with reference-deviation
terminations disabled — see ``_build_pose_env(disable_far_termination=True)``)
so the walker is free to drift without being killed.

Below each walker panel a per-frame code strip with a vertical cursor
shows which code the policy was operating on at each step.

Usage:
    cd code2act
    python -m c2a.experiments.run_e2e_live_vs_fixed_codes
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import hydra
import imageio
import jax
import matplotlib
matplotlib.use("Agg")
import numpy as np
import mujoco
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

from c2a.codegen.shared.label_alignment import load_label_frames
from c2a.experiments.shared.e2e_live_codes import load_e2e_moseq_checkpoint
from c2a.experiments.shared.e2e_rl_rollout import run_e2e_rollout
from c2a.experiments.shared.walk_rear_walk import find_walk_rear_walk_clips
from c2a.experiments.shared.code_grid import build_uniform_walkers
from c2a.experiments.shared.checkpoint_utils import make_inference_fn
from c2a.e2e_vqvae import build_motion_norm_padded, load_pretrained_vqvae
from c2a.moseq_env_wrapper import MoSeqImitation
from c2a.paths import get_repo_root

log = logging.getLogger(__name__)
REPO_ROOT = get_repo_root()


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _render_code_strip(
    codes: np.ndarray,
    *,
    height: int = 44,
    width: int = 384,
    cmap_name: str = "tab20",
    label: str = "",
    title_color: tuple = (40, 40, 40),
) -> np.ndarray:
    """Render a (T,) code sequence as a wide RGB strip image."""
    cmap = matplotlib.colormaps[cmap_name]
    T = int(codes.shape[0])
    n_colors = cmap.N
    color_strip = np.zeros((1, T, 3), dtype=np.uint8)
    for t in range(T):
        c = int(codes[t]) % n_colors
        rgba = cmap(c / max(n_colors - 1, 1))
        color_strip[0, t] = np.array([int(rgba[0] * 255), int(rgba[1] * 255),
                                       int(rgba[2] * 255)], dtype=np.uint8)
    img = Image.fromarray(color_strip).resize(
        (width, height), Image.Resampling.NEAREST,
    )
    if label:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13,
            )
        except Exception:
            font = ImageFont.load_default()
        x, y = 6, 3
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), label, fill=(255, 255, 255), font=font)
        draw.text((x, y), label, fill=title_color, font=font)
    return np.asarray(img)


def _draw_cursor(strip_rgb: np.ndarray, t_norm: float,
                 color: tuple = (255, 0, 0), width: int = 2) -> np.ndarray:
    img = strip_rgb.copy()
    W = img.shape[1]
    x = int(round(t_norm * (W - 1)))
    x0 = max(0, x - width // 2)
    x1 = min(W, x + width // 2 + 1)
    img[:, x0:x1, :] = np.array(color, dtype=np.uint8)
    return img


def _render_walker_panel(
    env: Any,
    qpos_trajectory: np.ndarray,
    *,
    walker_color: tuple = (0.4, 0.6, 0.95, 1.0),
    camera_distance: float = 0.8,
    camera_elevation: float = -30.0,
    camera_azimuth: float = 135.0,
    width: int = 384,
    height: int = 288,
    recenter_xy: bool = True,
) -> np.ndarray:
    """Render a 1-body walker over a qpos trajectory.

    With ``recenter_xy=True`` the walker's (x, y) is anchored to origin at
    t=0, preserving relative motion (so we still see drift between live
    and fixed) but keeping the body inside the fixed ``divergent_cam``
    framing instead of wandering off into the skybox.
    """
    if recenter_xy and qpos_trajectory.shape[0] > 0:
        qpos_trajectory = qpos_trajectory.copy()
        qpos_trajectory[:, 0] -= qpos_trajectory[0, 0]
        qpos_trajectory[:, 1] -= qpos_trajectory[0, 1]

    ghost_model, base_nq = build_uniform_walkers(
        env, walker_colors=[list(walker_color)],
        camera_distance=camera_distance,
        camera_elevation=camera_elevation,
        camera_azimuth=camera_azimuth,
    )
    ghost_model.vis.global_.offwidth = width
    ghost_model.vis.global_.offheight = height
    data = mujoco.MjData(ghost_model)
    renderer = mujoco.Renderer(ghost_model, height=height, width=width)
    T = qpos_trajectory.shape[0]
    out = np.zeros((T, height, width, 3), dtype=np.uint8)
    try:
        for t in range(T):
            data.qpos[:base_nq] = qpos_trajectory[t]
            mujoco.mj_forward(ghost_model, data)
            renderer.update_scene(data, camera="divergent_cam")
            out[t] = renderer.render()
    finally:
        renderer.close()
    return out


def _build_e2e_pose_env(
    ckpt_cfg: DictConfig,
    test_clips_data_path: str,
    clip_len: int,
    keep_clips_idx: np.ndarray,
    test_codes: np.ndarray,
    code_stack_size: int,
    probe_frame: int,
    e2e_ctx: Any,
    disable_far_termination: bool = True,
):
    """Build a MoSeqImitation env in e2e mode for the given clip subset.

    Mirrors ``_build_pose_env`` from the controlled hidden-dynamics
    experiment but additionally constructs the per-clip padded motion
    buffer required by the e2e ``motion_window_stack`` obs entry — without
    it the env runs in legacy mode and the policy raises
    ``KeyError: 'motion_window_stack'``.
    """
    from track_mjx.config import utils
    from vnl_playground.tasks.rodent.imitation import ReferenceClips

    clips = ReferenceClips(
        data_path=test_clips_data_path,
        n_frames_per_clip=clip_len,
        keep_clips_idx=np.asarray(keep_clips_idx, dtype=np.int64),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [probe_frame, probe_frame]
    env_cfg.domain_randomization.use_domain_randomization = False
    if disable_far_termination:
        if hasattr(env_cfg.termination_criteria, "root_too_far"):
            env_cfg.termination_criteria.root_too_far.max_distance = 1e6
        if hasattr(env_cfg.termination_criteria, "root_too_rotated"):
            env_cfg.termination_criteria.root_too_rotated.max_degrees = 1e6
        if hasattr(env_cfg.termination_criteria, "pose_error"):
            env_cfg.termination_criteria.pose_error.max_l2_error = 1e6

    qpos = np.asarray(clips.qpos, dtype=np.float32)
    qvel = (
        np.asarray(clips.qvel, dtype=np.float32) if e2e_ctx.use_qvel else None
    )
    motion_padded = build_motion_norm_padded(qpos, qvel, e2e_ctx)
    env = MoSeqImitation(
        config=env_cfg, clips=clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
        e2e_motion_norm_padded=motion_padded,
        e2e_window_len=int(e2e_ctx.window_len),
    )
    return env


_BEHAVIOR_COLORS_RGB = {
    "Immobile": (200, 200, 200),
    "Rear":     (220, 140, 70),
    "Turn":     (130, 90, 200),
    "Walk":     (90, 180, 110),
}


def _render_behavior_strip(
    behavior_per_frame: np.ndarray,
    *,
    height: int = 30,
    width: int = 384,
    label: str = "",
) -> np.ndarray:
    """Render a (T,) behavior id sequence as a colored strip.

    behavior ids follow :data:`label_alignment.BEHAVIORS` (0=Immobile,
    1=Rear, 2=Turn, 3=Walk) with -1 = unlabeled (rendered light gray).
    """
    from c2a.codegen.shared.label_alignment import BEHAVIORS
    T = int(behavior_per_frame.shape[0])
    color_strip = np.zeros((1, T, 3), dtype=np.uint8)
    for t in range(T):
        b = int(behavior_per_frame[t])
        if 0 <= b < len(BEHAVIORS):
            color_strip[0, t] = _BEHAVIOR_COLORS_RGB[BEHAVIORS[b]]
        else:
            color_strip[0, t] = (235, 235, 235)
    img = Image.fromarray(color_strip).resize(
        (width, height), Image.Resampling.NEAREST,
    )
    if label:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12,
            )
        except Exception:
            font = ImageFont.load_default()
        x, y = 6, 2
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), label, fill=(255, 255, 255), font=font)
        draw.text((x, y), label, fill=(30, 30, 30), font=font)
    return np.asarray(img)


def _composite_frame(
    walker_panels: list[np.ndarray],
    strip_panels: list[np.ndarray],
    panel_labels: list[str],
    panel_label_colors: list[tuple],
    title_text: str = "",
    panel_gap: int = 24,
    strip_gap_px: int = 4,
) -> np.ndarray:
    """Compose N walker panels in a row, with their N strips beneath.

    ``walker_panels`` and ``strip_panels`` must be same-length lists. Each
    walker is ``(H_w, W_w, 3)`` uint8; each strip is ``(H_s, W_w, 3)`` uint8
    (same W as its walker).
    """
    assert len(walker_panels) == len(strip_panels) == len(panel_labels)
    n = len(walker_panels)
    H_w, W_w = walker_panels[0].shape[:2]
    H_s = strip_panels[0].shape[0]
    sub_h = 20
    total_w = n * W_w + (n - 1) * panel_gap

    sub = np.full((sub_h, total_w, 3), 250, dtype=np.uint8)
    sub_img = Image.fromarray(sub)
    sub_draw = ImageDraw.Draw(sub_img)
    try:
        sub_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13,
        )
    except Exception:
        sub_font = ImageFont.load_default()
    for i, (lab, col) in enumerate(zip(panel_labels, panel_label_colors)):
        x0 = i * (W_w + panel_gap)
        try:
            tw, _ = sub_draw.textsize(lab, font=sub_font)
        except AttributeError:
            bbox = sub_draw.textbbox((0, 0), lab, font=sub_font)
            tw = bbox[2] - bbox[0]
        sub_draw.text((x0 + (W_w - tw) // 2, 2), lab, fill=col, font=sub_font)
    sub_arr = np.asarray(sub_img)

    gap_col_w = np.full((H_w, panel_gap, 3), 250, dtype=np.uint8)
    gap_col_s = np.full((H_s, panel_gap, 3), 250, dtype=np.uint8)
    top_parts = []
    bot_parts = []
    for i in range(n):
        top_parts.append(walker_panels[i])
        bot_parts.append(strip_panels[i])
        if i < n - 1:
            top_parts.append(gap_col_w)
            bot_parts.append(gap_col_s)
    top = np.concatenate(top_parts, axis=1)
    bot = np.concatenate(bot_parts, axis=1)
    sep = np.full((strip_gap_px, top.shape[1], 3), 250, dtype=np.uint8)
    composite = np.concatenate([sub_arr, top, sep, bot], axis=0)

    if title_text:
        banner_h = 30
        out = np.full((composite.shape[0] + banner_h, composite.shape[1], 3),
                       250, dtype=np.uint8)
        out[banner_h:] = composite
        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15,
            )
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 6), title_text, fill=(30, 30, 30), font=font)
        return np.asarray(img)
    return composite


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="e2e_live_vs_fixed_codes",
)
def main(cfg: DictConfig) -> None:
    log.info("=== E2E live-vs-fixed-codes 2-panel videos (real RL rollouts) ===")
    out_dir = Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    e2e_ckpt = _abs(cfg.checkpoint.path)
    pretrained_vqvae = _abs(cfg.checkpoint.pretrained_vqvae_dir)
    ref_h5 = _abs(cfg.data.reference_data_path)
    splits_path = _abs(cfg.data.balanced_split_path)

    # ---- Step 1: find 5 Rear → Walk → Rear clips inside the balanced split ----
    log.info("Loading labels and finding Rear→Walk→Rear clips ...")
    la = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=int(cfg.data.n_h5_frames_per_clip),
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    with open(splits_path) as f:
        splits_payload = json.load(f)
    balanced_ids = set(
        int(c) for c in splits_payload["balanced"]["train_indices"]
    ) | set(
        int(c) for c in splits_payload["balanced"]["test_indices"]
    )
    keep_mask = np.array(
        [int(c) in balanced_ids for c in la.clip_indices], dtype=bool,
    )
    filt_behavior = la.behavior[keep_mask]
    filt_clip_ids = la.clip_indices[keep_mask]
    log.info(
        f"  pre-filter: {keep_mask.sum()}/{la.clip_indices.shape[0]} "
        "labeled clips are in the balanced split"
    )
    wrw = find_walk_rear_walk_clips(
        filt_behavior, filt_clip_ids,
        outer=str(cfg.selection.outer),
        inner=str(cfg.selection.inner),
        strict=bool(cfg.selection.strict),
        bridge_max_frames=int(cfg.selection.bridge_max_frames),
        min_outer_frames=int(cfg.selection.min_outer_frames),
        min_inner_frames=int(cfg.selection.min_inner_frames),
        k=int(cfg.selection.k),
    )
    if not wrw:
        raise RuntimeError(
            f"No {cfg.selection.outer}→{cfg.selection.inner}→{cfg.selection.outer} "
            "clips found; relax selection criteria."
        )
    log.info(f"Selected {len(wrw)} clips:")
    for c in wrw:
        log.info(
            f"  clip {c.clip_id}: {cfg.selection.outer}[{c.walk1_start}:{c.walk1_end}] "
            f"{cfg.selection.inner}[{c.rear_start}:{c.rear_end}] "
            f"{cfg.selection.outer}[{c.walk2_start}:{c.walk2_end}]"
        )

    # ---- Step 2: load e2e ckpt + build inference fn ----
    log.info("Loading e2e ckpt and building inference fn ...")
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_e2e_moseq_checkpoint(
        e2e_ckpt, pretrained_vqvae_dir=pretrained_vqvae,
    )
    e2e_ctx = load_pretrained_vqvae(pretrained_vqvae)
    z_e_scale = float(cfg.get("z_e_scale", 0.0))
    inf_fn = make_inference_fn(
        ppo_networks, use_rnn=True,
        deterministic=True, z_e_scale=z_e_scale,
    )
    params = (norm_state, policy_params)
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 5))

    # ---- Step 3: fixed codes (pretrained tokenizer) keyed by clip id ----
    fixed_payload = np.load(_abs(cfg.checkpoint.fixed_codes_npz))
    all_fixed = fixed_payload["all_codes"]
    fixed_ids = (
        fixed_payload["all_indices"]
        if "all_indices" in fixed_payload.files
        else np.sort(np.unique(
            np.concatenate([fixed_payload["train_indices"],
                            fixed_payload["test_indices"]])
        ))
    )
    fixed_id_to_pos = {int(c): i for i, c in enumerate(fixed_ids)}

    # ---- Step 4: per-clip rollouts + render ----
    n_frames = int(cfg.data.n_h5_frames_per_clip)
    panel_w = int(cfg.render.panel_width)
    panel_h = int(cfg.render.panel_height)
    strip_h = int(cfg.render.strip_height)
    panel_gap = int(cfg.render.get("panel_gap", 24))
    fps_out = int(cfg.render.fps)
    seed = int(cfg.get("seed", 0))

    # GT qpos for the reference walker.
    with h5py.File(ref_h5, "r") as f:
        gt_qpos_all = np.asarray(f["qpos"][:], dtype=np.float32)
    # Per-clip behavior id rows for the reference strip.
    label_row_for_clip = {int(c): i for i, c in enumerate(la.clip_indices)}

    for wi, clip in enumerate(wrw):
        cid = int(clip.clip_id)
        log.info(f"\n--- Clip {cid} ({wi+1}/{len(wrw)}) ---")
        if cid not in fixed_id_to_pos:
            log.warning(f"  clip {cid} not in fixed encoding; skipping")
            continue
        fixed_codes = all_fixed[fixed_id_to_pos[cid]].astype(np.int32)

        # Build env restricted to this one clip; reset_clip_idx=0 is the only
        # available clip after subsetting. start_frame_range is fixed at
        # probe_frame=0 inside _build_pose_env, so each rollout starts at the
        # clip's frame 0. ``test_codes`` carries the pretrained-tokenizer code
        # sequence — the env exposes it via ``kpms_code`` but in e2e mode the
        # policy ignores that (unless we explicitly pass ``code_override``).
        log.info("  building env (e2e mode, termination disabled) ...")
        env = _build_e2e_pose_env(
            ckpt_cfg, ref_h5, n_frames,
            keep_clips_idx=np.array([cid], dtype=np.int64),
            test_codes=fixed_codes[None, :],   # (1, T)
            code_stack_size=code_stack_size,
            probe_frame=0,
            e2e_ctx=e2e_ctx,
            disable_far_termination=True,
        )
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        # LIVE rollout: no code_override; policy uses its live VQ-VAE.
        log.info("  rolling out LIVE (real RL, policy uses live VQ-VAE) ...")
        live_key = jax.random.PRNGKey(seed + 1000 * cid)
        live_out = run_e2e_rollout(
            env, inf_fn, params, ppo_networks, live_key,
            max_steps=n_frames, code_override=None,
            reset_clip_idx=0, jit_reset=jit_reset, jit_step=jit_step,
        )
        log.info(
            f"    survived {live_out['survival']}/{n_frames}, "
            f"mean reward {live_out['rewards'].mean():.3f}, "
            f"unique live codes {len(np.unique(live_out['live_code_idx']))}"
        )

        # FIXED rollout: feed the pretrained-tokenizer code sequence as override.
        log.info("  rolling out FIXED (policy forced onto pretrained codes) ...")
        fixed_key = jax.random.PRNGKey(seed + 1000 * cid + 1)
        fixed_out = run_e2e_rollout(
            env, inf_fn, params, ppo_networks, fixed_key,
            max_steps=n_frames, code_override=fixed_codes,
            reset_clip_idx=0, jit_reset=jit_reset, jit_step=jit_step,
        )
        log.info(
            f"    survived {fixed_out['survival']}/{n_frames}, "
            f"mean reward {fixed_out['rewards'].mean():.3f}, "
            f"unique fixed codes {len(np.unique(fixed_codes))}"
        )

        qpos_live = live_out["qpos"]
        qpos_fixed = fixed_out["qpos"]
        live_codes_seq = live_out["live_code_idx"]
        fixed_codes_seq = fixed_codes  # what we forced

        # GT reference qpos for this clip.
        qpos_ref = gt_qpos_all[cid * n_frames:(cid + 1) * n_frames]

        # Render walker panels (GT reference, live rollout, fixed-code rollout).
        log.info("  rendering walker panel (reference) ...")
        frames_ref = _render_walker_panel(
            env, qpos_ref,
            walker_color=tuple(cfg.render.reference_color),
            camera_distance=float(cfg.render.camera_distance),
            camera_elevation=float(cfg.render.camera_elevation),
            camera_azimuth=float(cfg.render.camera_azimuth),
            width=panel_w, height=panel_h,
            recenter_xy=True,
        )
        log.info("  rendering walker panel (live) ...")
        frames_live = _render_walker_panel(
            env, qpos_live,
            walker_color=tuple(cfg.render.live_color),
            camera_distance=float(cfg.render.camera_distance),
            camera_elevation=float(cfg.render.camera_elevation),
            camera_azimuth=float(cfg.render.camera_azimuth),
            width=panel_w, height=panel_h,
            recenter_xy=True,
        )
        log.info("  rendering walker panel (fixed) ...")
        frames_fixed = _render_walker_panel(
            env, qpos_fixed,
            walker_color=tuple(cfg.render.fixed_color),
            camera_distance=float(cfg.render.camera_distance),
            camera_elevation=float(cfg.render.camera_elevation),
            camera_azimuth=float(cfg.render.camera_azimuth),
            width=panel_w, height=panel_h,
            recenter_xy=True,
        )

        # GT behavior id row for this clip (default to all-unlabeled if the
        # clip is somehow absent from la — shouldn't happen since we
        # filtered to la's clips upstream).
        if cid in label_row_for_clip:
            ref_behavior = la.behavior[label_row_for_clip[cid]]
        else:
            ref_behavior = np.full(n_frames, -1, dtype=np.int32)
        strip_ref_base = _render_behavior_strip(
            ref_behavior, height=strip_h, width=panel_w,
            label="GT behavior",
        )
        strip_live_base = _render_code_strip(
            live_codes_seq, height=strip_h, width=panel_w,
            label=f"{len(np.unique(live_codes_seq))} unique codes",
        )
        strip_fixed_base = _render_code_strip(
            fixed_codes_seq, height=strip_h, width=panel_w,
            label=f"{len(np.unique(fixed_codes_seq))} unique codes",
        )

        out_path = out_dir / f"clip_{cid:04d}_rwr_live_vs_fixed.mp4"
        log.info(f"  writing {out_path}")
        T_render = min(
            len(frames_ref), len(frames_live), len(frames_fixed),
            len(live_codes_seq), len(fixed_codes_seq),
        )
        writer = imageio.get_writer(str(out_path), fps=fps_out,
                                     codec="libx264", quality=8)
        disagree = (live_codes_seq[:T_render] != fixed_codes_seq[:T_render]).mean()
        panel_labels = [
            "GT reference",
            "live (e2e codebook)",
            "fixed (pretrained codebook)",
        ]
        panel_label_colors = [
            (180, 110, 20),    # warm amber to match reference walker
            (20, 60, 140),     # deep blue to match live walker
            (80, 80, 80),      # gray to match fixed walker
        ]
        try:
            for t in range(T_render):
                t_norm = t / max(T_render - 1, 1)
                strip_ref_t = _draw_cursor(strip_ref_base, t_norm)
                strip_live_t = _draw_cursor(strip_live_base, t_norm)
                strip_fixed_t = _draw_cursor(strip_fixed_base, t_norm)
                composite = _composite_frame(
                    walker_panels=[frames_ref[t], frames_live[t], frames_fixed[t]],
                    strip_panels=[strip_ref_t, strip_live_t, strip_fixed_t],
                    panel_labels=panel_labels,
                    panel_label_colors=panel_label_colors,
                    title_text=(
                        f"clip {cid}  |  Rear-Walk-Rear  |  "
                        f"live-vs-fixed code disagree {disagree*100:.0f}%  |  "
                        f"surv live {live_out['survival']}/{n_frames}, "
                        f"fixed {fixed_out['survival']}/{n_frames}"
                    ),
                    panel_gap=panel_gap,
                )
                writer.append_data(composite)
        finally:
            writer.close()
        log.info(f"  done: {out_path}")

    log.info(f"\n=== Done. Outputs under {out_dir}/ ===")


if __name__ == "__main__":
    main()
