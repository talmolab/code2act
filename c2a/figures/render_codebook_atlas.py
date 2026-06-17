"""Codebook atlas for the hand-designed multi-scale codebook.

For every designed code, hold it constant for ``--hold`` frames (default 1000)
and render its motion as a frozen-ghost-copy trail (the same style as the
behavior parade), in a single color = the code's defined behavior. The 25 trails
are tiled and grouped by compositional scale (fine / p3 / d3 / n4 / n5 / long),
giving an at-a-glance catalog of what every code does.

Reuses the precomputed held-code rollouts (no new rollout): one trajectory per
code is taken from the best-reward body of
``outputs/code_grid/code_grid_rollouts/<run>/data/rollout_<pose>_code*.npz``.

Usage:
    python -m c2a.figures.render_codebook_atlas \
        --run multiscale --pose random \
        --codes outputs/tokenizer/designed_v1/best_codes.npz \
        --checkpoint model_checkpoints/260606_083446_639090
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

from c2a.paths import get_repo_root

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from c2a.figures.render_behavior_parade_trail import (
    build_parade_trail_model,
    recolor_copies,
    render_trail_frame,
)
from c2a.figures.paper.render_paper_figures import behavior_sequence

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# behavior colors consistent with the paper matrices (render_paper_figures)
GROUP_COLORS_RGB = {
    "Immobile": (0.298, 0.447, 0.690),
    "Rear": (0.867, 0.518, 0.322),
    "Turn": (0.506, 0.447, 0.702),
    "Walk": (0.333, 0.659, 0.408),
}
SCALE_ORDER = ["fine", "p3", "d3", "n4", "n5", "long"]
SCALE_TITLE = {
    "fine": "Fine (single behavior)",
    "p3": "Palindromic 3-gram",
    "d3": "Directed 3-gram",
    "n4": "4-gram",
    "n5": "5-gram",
    "long": "Long (sustained)",
}
_TOK = {"Walk": "Walk", "Rear": "Rear", "Turn": "Turn", "Imm": "Immobile"}


def designed_meta(codes_path: Path) -> dict[int, dict]:
    """code_id -> {name, scale, behavior} from the designed provenance."""
    d = np.load(codes_path, allow_pickle=True)
    prov = json.loads(str(d["provenance_json"]))
    meta = {}
    for pc in prov["per_code"]:
        # long (sustained) hold codes are excluded from the multi-scale writeup
        if pc.get("type") in ("sentinel", "long"):
            continue
        cid = int(pc["id"])
        body = pc["name"].split(".", 1)[1]
        meta[cid] = {
            "name": pc["name"],
            "scale": pc["type"],
            "behavior": _TOK.get(body.split("_")[0], body.split("_")[0]),
        }
    return meta


def build_env(ckpt_path: str):
    """Minimal env (for the walker XML + rescale used to build trail models)."""
    from omegaconf import OmegaConf
    from track_mjx.config import utils
    from track_mjx.agent import checkpointing
    from vnl_playground.tasks.rodent.imitation import ReferenceClips
    from c2a.moseq_env_wrapper import MoSeqImitation

    root = get_repo_root()
    cfg = checkpointing.load_config_from_checkpoint(
        ckpt_path, step_prefix="MoSeqPPONetwork"
    )
    cfg = OmegaConf.create(cfg)
    _, _, env_cfg = utils.prepare_config(cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    splits = json.load(open(root / "data" / "rodent" / "rodent_balanced_splits.json"))
    test_indices = splits["balanced"]["test_indices"]
    codes = np.load(
        root / "outputs" / "tokenizer" / "designed_v1" / "best_codes.npz"
    )["test_codes"]
    clips = ReferenceClips(
        data_path=str(root / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    return MoSeqImitation(config=env_cfg, clips=clips, kpms_codes=codes)


def render_one_code(env, qpos, n_copies, color_rgb, elev, azim, fovy,
                    width, height):
    """Render a single-color motion trail for one code's trajectory."""
    qpos = np.asarray(qpos, dtype=np.float64)
    idxs = np.linspace(0, len(qpos) - 1, n_copies).round().astype(int)
    trace_ts = np.arange(0, len(qpos), 3)
    trace_pos = qpos[trace_ts, :3].copy()
    trace_pos[:, 2] += 0.005
    trace_segments = [(trace_pos, [*color_rgb, 0.8])]
    copy_qpos = [qpos[t].copy() for t in idxs]
    positions = np.stack([qp[:3] for qp in copy_qpos], axis=0)
    extent = positions.max(axis=0) - positions.min(axis=0)
    max_span = max(extent[0], extent[1])
    cam_dist = max((max_span + 0.25) / (2 * math.tan(math.radians(fovy) / 2)), 0.4)
    model, base_nq = build_parade_trail_model(
        env, num_copies=len(idxs), trace_segments=trace_segments,
        camera_distance=cam_dist, camera_elevation=elev,
        camera_azimuth=azim, camera_fovy=fovy,
    )
    recolor_copies(model, [[*color_rgb, 1.0]] * len(idxs))
    return render_trail_frame(model, base_nq, copy_qpos, width=width, height=height)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="multiscale")
    ap.add_argument("--pose", default="random")
    ap.add_argument("--codes",
                    default="outputs/tokenizer/designed_v1/best_codes.npz")
    ap.add_argument("--checkpoint",
                    default="model_checkpoints/260606_083446_639090")
    ap.add_argument("--hold", type=int, default=1000)
    ap.add_argument("--n-copies", type=int, default=7)
    ap.add_argument("--cam-elev", type=float, default=-22.0)
    ap.add_argument("--cam-azim", type=float, default=-130.0)
    ap.add_argument("--cam-fovy", type=float, default=45.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    root = get_repo_root()
    codes_path = root / args.codes
    meta = designed_meta(codes_path)
    data_dir = (root / "outputs" / "code_grid" / "code_grid_rollouts"
                / args.run / "data")

    env = build_env(str(root / args.checkpoint))
    log.info("Environment loaded")

    # render one trail per code (best-reward body, first --hold frames)
    frames: dict[int, np.ndarray] = {}
    for cid in sorted(meta):
        f = data_dir / f"rollout_{args.pose}_code{cid:03d}.npz"
        if not f.exists():
            log.warning(f"[skip] no rollout for code {cid} at {f}")
            continue
        d = np.load(f, allow_pickle=True)
        q = np.asarray(d["qpos"])           # (bodies, T, nq)
        r = np.asarray(d["rewards"])        # (bodies, T)
        hold = min(args.hold, q.shape[1])
        best = int(np.argmax(r[:, :hold].mean(axis=1)))
        color = GROUP_COLORS_RGB.get(meta[cid]["behavior"], (0.5, 0.5, 0.5))
        frames[cid] = render_one_code(
            env, q[best, :hold], args.n_copies, color,
            args.cam_elev, args.cam_azim, args.cam_fovy, args.width, args.height,
        )
        log.info(f"  code {cid:2d} {meta[cid]['name']:28s} "
                 f"[{meta[cid]['scale']}/{meta[cid]['behavior']}] body {best}")

    # ---- montage: rows = scale, columns = codes within that scale ----
    by_scale = {s: [c for c in sorted(meta) if meta[c]["scale"] == s and c in frames]
                for s in SCALE_ORDER}
    by_scale = {s: cs for s, cs in by_scale.items() if cs}
    ncol = max(len(cs) for cs in by_scale.values())
    nrow = len(by_scale)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.2 * nrow),
                             squeeze=False)
    for ri, (scale, cs) in enumerate(by_scale.items()):
        for ci in range(ncol):
            ax = axes[ri][ci]
            ax.set_axis_off()
            if ci >= len(cs):
                continue
            cid = cs[ci]
            ax.imshow(frames[cid])
            beh = meta[cid]["behavior"]
            ax.set_title(meta[cid]["name"].split(".", 1)[1],
                         fontsize=9, color=GROUP_COLORS_RGB.get(beh, "black"),
                         pad=2)
            # defined behavior-composition bar beneath the trail
            from matplotlib.patches import Rectangle
            seq = behavior_sequence(meta[cid]["name"]) or [beh]
            w = 1.0 / len(seq)
            for k, b in enumerate(seq):
                ax.add_patch(Rectangle(
                    (k * w, -0.10), w, 0.06, transform=ax.transAxes,
                    color=GROUP_COLORS_RGB.get(b, (0.5, 0.5, 0.5)),
                    ec="white", lw=0.4, clip_on=False))
        # scale row label on the far left
        axes[ri][0].text(-0.06, 0.5, SCALE_TITLE.get(scale, scale),
                         transform=axes[ri][0].transAxes, rotation=90,
                         ha="right", va="center", fontsize=11, fontweight="bold")
    fig.suptitle("Designed codebook atlas: each code held for "
                 f"{args.hold} frames (color = defined behavior)", fontsize=13)
    fig.tight_layout(rect=[0.03, 0, 1, 0.98])

    out_dir = root / "outputs" / "codebook_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"codebook_atlas.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_dir}/codebook_atlas.pdf / .png "
             f"({len(frames)} codes, {nrow} scales)")


if __name__ == "__main__":
    main()
