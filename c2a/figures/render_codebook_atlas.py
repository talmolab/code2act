"""Codebook atlas for the mlb (Manually Labeled Behaviors) steering strategy.

For each of the 9 flat mlb codes, hold it constant for ``--hold`` frames
(default 1000) and render its motion as a frozen-ghost-copy trail (the same
style as the behavior parade), colored by the canonical 9-class behavior palette
(behavior = hue family, direction = shade). The 9 trails are tiled in the
fixed 3x3 :data:`c2a.codegen.shared.mlb_labels.MLB_GRID` layout (rows = behavior
Walk / Rear / Turn; columns = Left / center / Right; Immobile in the Turn-row
center), giving an at-a-glance catalog of what every mlb code does.

Reuses the precomputed held-code rollouts (no new rollout): one trajectory per
code is taken from the medoid body of
``outputs/code_grid/code_grid_rollouts/<run>/data/rollout_<pose>_code*.npz``.

Usage:
    python -m c2a.figures.render_codebook_atlas \
        --run mlb --pose random \
        --codes outputs/tokenizer/mlb_v1/best_codes.npz \
        --checkpoint model_checkpoints/<mlb_ckpt>
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
from c2a.codegen.shared.mlb_labels import (
    MLB_COLORS,
    MLB_GRID,
    MLB_LABELS,
    MLB_LABEL_TO_INT,
    MLB_SHORT,
)
from c2a.experiments.ic_contraction.run_ic_contraction import pose_features

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Behavior colors consistent with the paper matrices (render_paper_figures);
# kept for the data-discovered atlas (render_code_atlas_grouped imports this).
GROUP_COLORS_RGB = {
    "Immobile": (0.298, 0.447, 0.690),
    "Rear": (0.867, 0.518, 0.322),
    "Turn": (0.506, 0.447, 0.702),
    "Walk": (0.333, 0.659, 0.408),
}


def mlb_meta() -> dict[int, dict]:
    """code_id -> {label, short, behavior, color} for the 9 flat mlb codes.

    Code id == index into :data:`MLB_LABELS` (the mlb segmenter writes codes in
    that canonical order), so no provenance parsing is needed: the codebook is a
    fixed behavior x direction map.
    """
    meta: dict[int, dict] = {}
    for cid, label in enumerate(MLB_LABELS):
        meta[cid] = {
            "label": label,
            "short": MLB_SHORT[label],
            "behavior": label.split("_")[0],
            "color": tuple(MLB_COLORS[label]),
        }
    return meta


def select_representative_body(
    qpos: np.ndarray, hold: int, late_frac: float = 0.33,
    knn: int = 5, subsample: int = 2500, pca_dim: int = 10, seed: int = 0,
) -> int:
    """Pick the body whose motion best *represents* the code's attractor.

    The held-code rollout drives ``B`` bodies from diverse initial poses; they
    converge onto a shared per-code goal pose (the one-to-many / IC-contraction
    result). The old atlas rendered the single best-*reward* body, which can be
    an idiosyncratic outlier that does not look like what the code actually does.
    Instead we render the *medoid*: the body whose late-window poses sit most
    densely inside the pooled late-window attractor cloud (root-stripped,
    yaw-removed pose space — the same features used for IC-contraction). This is,
    by construction, the body that "actually resembles the walker" the code
    produces.

    Args:
        qpos: ``(B, T, nq)`` held-code rollout.
        hold: number of leading frames considered (matches what is rendered).
        late_frac: trailing fraction defining the converged window.

    Returns:
        Index of the representative body.
    """
    B = qpos.shape[0]
    hold = min(hold, qpos.shape[1])
    if B <= 1:
        return 0
    feat = pose_features(qpos[:, :hold].astype(np.float32))   # (B, hold, F)
    flat = feat.reshape(-1, feat.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    featz = (feat - mean) / std
    n_late = max(2, int(round(late_frac * hold)))
    late = np.arange(hold - n_late, hold)
    rng = np.random.RandomState(seed)
    M = featz[:, late].reshape(-1, featz.shape[-1])
    if M.shape[0] > subsample:
        M = M[rng.choice(M.shape[0], subsample, replace=False)]
    dim = min(pca_dim, M.shape[1], M.shape[0] - 1)
    pca = PCA(n_components=dim).fit(M)
    nn = NearestNeighbors(n_neighbors=min(knn, M.shape[0])).fit(pca.transform(M))
    scores = np.empty(B)
    for b in range(B):
        dist, _ = nn.kneighbors(pca.transform(featz[b, late]))
        scores[b] = float(dist.mean())
    return int(np.argmin(scores))


def build_env(ckpt_path: str, codes_path: str | None = None):
    """Minimal env (for the walker XML + rescale used to build trail models).

    Args:
        ckpt_path: decoder checkpoint (only its env config is read).
        codes_path: ``best_codes.npz`` whose ``test_codes`` seed the env's code
            stream. Only the walker XML / rescale matter for trail rendering, so
            the exact codes are immaterial; defaults to the designed_v3 codes
            for backward compatibility with the data-discovered atlas caller.
    """
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
    if codes_path is None:
        codes_path = str(
            root / "outputs" / "tokenizer" / "designed_v3" / "best_codes.npz"
        )
    codes = np.load(codes_path)["test_codes"]
    clips = ReferenceClips(
        data_path=str(root / "data" / "rodent" / "rodent_reference_clips.h5"),
        n_frames_per_clip=int(cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    return MoSeqImitation(config=env_cfg, clips=clips, kpms_codes=codes)


def render_one_code(env, qpos, n_copies, color_rgb, elev, azim, fovy,
                    width, height, anchor=False):
    """Render a single-color motion trail for one code's trajectory.

    ``anchor=True`` removes world translation (zeros each copy's root x/y) so a
    *translating* behavior (walking) overlays into a single in-place striding
    form, reading as a clean gait rather than a spread caterpillar of bodies.
    Stationary behaviors (rear/immobile) are unaffected by anchoring."""
    qpos = np.asarray(qpos, dtype=np.float64)
    idxs = np.linspace(0, len(qpos) - 1, n_copies).round().astype(int)
    copy_qpos = [qpos[t].copy() for t in idxs]
    if anchor:
        for qp in copy_qpos:
            qp[0:2] = 0.0                    # overlay all copies at the origin
        # invisible 1-segment trace (build_parade_trail_model needs >=1 segment)
        trace_segments = [(np.zeros((2, 3), dtype=np.float64), [*color_rgb, 0.0])]
        max_span = 0.36                      # frame the body + limb spread
    else:
        trace_ts = np.arange(0, len(qpos), 3)
        trace_pos = qpos[trace_ts, :3].copy()
        trace_pos[:, 2] += 0.005
        trace_segments = [(trace_pos, [*color_rgb, 0.8])]
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
    ap.add_argument("--run", default="mlb")
    ap.add_argument("--pose", default="random")
    ap.add_argument("--codes",
                    default="outputs/tokenizer/mlb_v1/best_codes.npz")
    ap.add_argument("--checkpoint",
                    default="model_checkpoints/MLB_CKPT_PLACEHOLDER")
    ap.add_argument("--hold", type=int, default=1000)
    ap.add_argument("--select", default="medoid",
                    choices=["medoid", "best_reward"],
                    help="how to pick the body to render per code")
    ap.add_argument("--select-overrides", nargs="*", default=None,
                    help="per-code body-selection overrides, e.g. '0=medoid' to "
                         "render code 0 with the medoid picker while every other "
                         "code uses --select. Repeatable.")
    ap.add_argument("--n-copies", type=int, default=7)
    ap.add_argument("--hold-overrides", nargs="*", default=None,
                    help="per-code hold overrides, e.g. '2=160' to render code 2 "
                         "over a shorter trajectory (fewer frames -> tighter camera "
                         "fit -> larger body). Repeatable.")
    ap.add_argument("--copies-overrides", nargs="*", default=None,
                    help="per-code n_copies overrides, e.g. '2=4'. Repeatable.")
    ap.add_argument("--anchor-mode", default="auto",
                    choices=["auto", "on", "off"],
                    help="overlay translating (walk) copies in place: auto|on|off")
    ap.add_argument("--grid", default="behavior3x3",
                    choices=["behavior3x3", "compact2x5"],
                    help="montage layout: behavior3x3 (rows=behavior, cols=L/S/R) or "
                         "compact2x5 (2 rows x 5 cols, moving then postural, with a "
                         "blocky behavior legend in the 10th cell)")
    ap.add_argument("--cam-elev", type=float, default=-22.0)
    ap.add_argument("--cam-azim", type=float, default=-130.0)
    ap.add_argument("--cam-fovy", type=float, default=45.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    root = get_repo_root()
    meta = mlb_meta()  # 9 flat codes, id == MLB_LABELS index
    data_dir = (root / "outputs" / "code_grid" / "code_grid_rollouts"
                / args.run / "data")

    # per-code selection overrides ("0=medoid") -> {code_id: method}
    select_overrides: dict[int, str] = {}
    for spec in (args.select_overrides or []):
        cid_s, _, method = spec.partition("=")
        select_overrides[int(cid_s)] = method
    hold_overrides: dict[int, int] = {}
    for spec in (args.hold_overrides or []):
        cid_s, _, val = spec.partition("=")
        hold_overrides[int(cid_s)] = int(val)
    copies_overrides: dict[int, int] = {}
    for spec in (args.copies_overrides or []):
        cid_s, _, val = spec.partition("=")
        copies_overrides[int(cid_s)] = int(val)

    env = build_env(str(root / args.checkpoint), codes_path=str(root / args.codes))
    log.info("Environment loaded")

    # render one trail per code (medoid body, first --hold frames)
    frames: dict[int, np.ndarray] = {}
    for cid in sorted(meta):
        f = data_dir / f"rollout_{args.pose}_code{cid:03d}.npz"
        if not f.exists():
            log.warning(f"[skip] no rollout for code {cid} at {f}")
            continue
        d = np.load(f, allow_pickle=True)
        q = np.asarray(d["qpos"])           # (bodies, T, nq)
        r = np.asarray(d["rewards"])        # (bodies, T)
        hold = min(hold_overrides.get(cid, args.hold), q.shape[1])
        n_copies = copies_overrides.get(cid, args.n_copies)
        method = select_overrides.get(cid, args.select)
        if method == "medoid":
            best = select_representative_body(q, hold)
        elif method in ("farthest", "far_left", "far_right", "far_straight"):
            # clearest walker: largest steady forward displacement, optionally
            # matching the code's labeled heading (net yaw; L=+, R=-).
            disp = np.linalg.norm(
                q[:, hold - 1, :2].astype(np.float64)
                - q[:, 0, :2].astype(np.float64), axis=1)
            if method == "farthest":
                best = int(np.argmax(disp))
            else:
                quat = q[:, :hold, 3:7].astype(np.float64)
                w, x, y, z = (quat[..., 0], quat[..., 1],
                              quat[..., 2], quat[..., 3])
                yaw = np.unwrap(np.arctan2(2 * (w * z + x * y),
                                           1 - 2 * (y * y + z * z)), axis=1)
                ny = yaw[:, -1] - yaw[:, 0]
                mask = ({"far_left": ny > 0.2, "far_right": ny < -0.2,
                         "far_straight": np.abs(ny) < 0.3})[method]
                score = np.where(mask, disp, -1.0)
                best = int(np.argmax(score)) if mask.any() else int(np.argmax(disp))
        else:
            best = int(np.argmax(r[:, :hold].mean(axis=1)))
        color = meta[cid]["color"]
        traj = q[best, :hold]
        # Auto-anchor codes that translate a lot (walking): overlay the copies at
        # the origin so they read as an in-place striding gait instead of a spread
        # caterpillar. Stationary codes (rear/immobile/turn) render as trails.
        rxy = np.asarray(traj[:, :2], dtype=np.float64)
        world_span = float(np.linalg.norm(rxy.max(0) - rxy.min(0)))
        if args.anchor_mode == "on":
            anchor = True
        elif args.anchor_mode == "off":
            anchor = False
        else:  # auto: overlay only strongly-translating (walk) codes
            anchor = world_span > 0.15
        frames[cid] = render_one_code(
            env, traj, n_copies, color,
            args.cam_elev, args.cam_azim, args.cam_fovy, args.width, args.height,
            anchor=anchor,
        )
        log.info(f"  code {cid:2d} {meta[cid]['label']:14s} body {best} "
                 f"span={world_span:.2f} anchor={anchor}")

    # ---- montage layout ----
    if args.grid == "compact2x5":
        # 2 rows x 5 cols: row 1 = moving (walk, turn), row 2 = postural
        # (rear, immobile) + a blocky behavior legend in the last (10th) cell,
        # filling the whitespace that a 9-into-10 grid leaves.
        from matplotlib.patches import Patch
        layout = [
            ["Walk_Left", "Walk_Straight", "Walk_Right", "Turn_Left", "Turn_Right"],
            ["Rear_Left", "Rear_Straight", "Rear_Right", "Immobile", "__LEGEND__"],
        ]
        nrow, ncol = 2, 5
    else:  # behavior3x3
        layout = [list(row) for row in MLB_GRID]
        nrow, ncol = len(MLB_GRID), len(MLB_GRID[0])

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ri, row in enumerate(layout):
        for ci, label in enumerate(row):
            ax = axes[ri][ci]
            ax.set_axis_off()
            if label == "__LEGEND__":
                handles = [Patch(facecolor=tuple(MLB_COLORS[l]), edgecolor="none",
                                 label=MLB_SHORT[l]) for l in MLB_LABELS]
                ax.legend(handles=handles, title="behavior", loc="center",
                          ncol=2, fontsize=17, title_fontsize=19, frameon=False,
                          handlelength=1.5, handleheight=1.5, handletextpad=0.5,
                          columnspacing=1.0, labelspacing=0.7)
                continue
            cid = MLB_LABEL_TO_INT[label]
            color = tuple(MLB_COLORS[label])
            if cid not in frames:
                # placeholder cell (rollout not yet rendered, e.g. Phase B)
                ax.text(0.5, 0.5, f"{MLB_SHORT[label]}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=24, color=color, fontweight="bold")
                continue
            ax.imshow(frames[cid])
            ax.set_title(f"{MLB_SHORT[label]}",
                         fontsize=26, color=color, pad=6)
    fig.tight_layout()

    out_dir = root / "outputs" / "figures" / "codebook_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"codebook_atlas_{args.run}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_dir}/codebook_atlas_{args.run}.pdf / .png "
             f"({len(frames)}/9 codes rendered, {nrow}x{ncol} {args.grid})")


if __name__ == "__main__":
    main()
