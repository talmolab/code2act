"""Behavior-grouped code atlas for data-discovered tokenizers.

Companion to ``render_codebook_atlas`` (the hand-designed codebook). For a
data-discovered tokenizer (Keypoint-MoSeq or the kinematics VAE + k-means) this
renders the top-N most-used codes as frozen-ghost motion trails, **grouped by
modal behavior** — the same grouping used for the behavior→code spread of the
code-distinctness figure (Fig. 6b). It makes two points the distinctness section
argues numerically:

* one behavior is realized by *several* codes (the redundancy quantified by the
  ``6.7/100`` effective-code count for Keypoint-MoSeq): within a behavior row,
  multiple codes drive visibly similar — yet distinct — motor programs;
* each panel shows the *representative* (medoid) body across the 100 held-code
  walkers, i.e. what the code actually produces, not a high-reward outlier.

No new rollouts: reuses the held-code rollouts and per-code labels already
produced by ``run_code_grid_rollouts`` under
``outputs/code_grid/code_grid_rollouts/<run>/``.

Usage::

    python -m c2a.figures.render_code_atlas_grouped --run kpms --top 15
    python -m c2a.figures.render_code_atlas_grouped --run best_vae --top 15
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from c2a.paths import get_repo_root

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from c2a.figures.render_codebook_atlas import (
    GROUP_COLORS_RGB,
    build_env,
    render_one_code,
    select_representative_body,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Behavior row order (matches the paper's matrices / Fig 6b ordering).
BEHAVIOR_ORDER = ["Immobile", "Walk", "Rear", "Turn"]
RUN_TITLE = {
    "kpms": "Keypoint-MoSeq (K=100)",
    "best_vae": "Kinematics VAE + k-means (K=25)",
}


def _base_behavior(label: str) -> str:
    if not label or label == "unlabeled":
        return "Other"
    return label.split("_")[0]


def load_run_meta(run_dir: Path, top: int) -> tuple[list[int], dict[int, str], np.ndarray]:
    """Return (top_codes_by_usage, code->behavior, usage array)."""
    usage = np.load(run_dir / "data" / "usage_rank.npz")
    ranked = np.asarray(usage["ranked_codes"])
    usage_counts = np.asarray(usage["usage"])
    labels = json.loads((run_dir / "labels" / "code_labels.json").read_text())
    code_beh = {
        int(cid): _base_behavior(str(rec.get("label", "")))
        for cid, rec in labels.get("codes", {}).items()
    }
    # top-N codes that have a behavior label and a rollout on disk
    top_codes: list[int] = []
    for c in ranked.tolist():
        c = int(c)
        f = run_dir / "data" / f"rollout_random_code{c:03d}.npz"
        if f.exists() and code_beh.get(c, "Other") != "Other":
            top_codes.append(c)
        if len(top_codes) >= top:
            break
    return top_codes, code_beh, usage_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="kpms", help="code_grid run name")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--pose", default="random")
    ap.add_argument("--checkpoint",
                    default="model_checkpoints/260407_031233_484020")
    ap.add_argument("--hold", type=int, default=1000)
    ap.add_argument("--n-copies", type=int, default=7)
    ap.add_argument("--cam-elev", type=float, default=-22.0)
    ap.add_argument("--cam-azim", type=float, default=-130.0)
    ap.add_argument("--cam-fovy", type=float, default=45.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    root = get_repo_root()
    run_dir = root / "outputs" / "code_grid" / "code_grid_rollouts" / args.run
    data_dir = run_dir / "data"
    top_codes, code_beh, usage = load_run_meta(run_dir, args.top)
    log.info(f"[{args.run}] top {len(top_codes)} codes: {top_codes}")

    env = build_env(str(root / args.checkpoint))
    log.info("Environment loaded")

    frames: dict[int, np.ndarray] = {}
    for cid in top_codes:
        f = data_dir / f"rollout_{args.pose}_code{cid:03d}.npz"
        d = np.load(f, allow_pickle=True)
        q = np.asarray(d["qpos"])
        hold = min(args.hold, q.shape[1])
        body = select_representative_body(q, hold)
        beh = code_beh.get(cid, "Other")
        color = GROUP_COLORS_RGB.get(beh, (0.5, 0.5, 0.5))
        frames[cid] = render_one_code(
            env, q[body, :hold], args.n_copies, color,
            args.cam_elev, args.cam_azim, args.cam_fovy, args.width, args.height,
        )
        log.info(f"  code {cid:>3d} [{beh:>8s}] body {body} use={int(usage[cid])}")

    # rows = behavior; codes within a behavior wrap to the next line after
    # MAXCOL so panels (and their text) stay large. Title lives in the LaTeX
    # caption, not the image.
    MAXCOL = 5
    by_beh = {
        b: sorted([c for c in top_codes if code_beh.get(c) == b],
                  key=lambda c: -usage[c])
        for b in BEHAVIOR_ORDER
    }
    by_beh = {b: cs for b, cs in by_beh.items() if cs}
    layout: list[tuple[str, list[int]]] = []
    for beh, cs in by_beh.items():
        for i in range(0, len(cs), MAXCOL):
            layout.append((beh if i == 0 else "", cs[i:i + MAXCOL]))
    nrow, ncol = len(layout), MAXCOL
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow),
                             squeeze=False)
    for ri, (label, chunk) in enumerate(layout):
        for ci in range(ncol):
            ax = axes[ri][ci]
            ax.set_axis_off()
            if ci >= len(chunk):
                continue
            cid = chunk[ci]
            beh = code_beh.get(cid, "Other")
            ax.imshow(frames[cid])
            ax.set_title(f"code {cid}  (use {int(usage[cid])})",
                         fontsize=14, color=GROUP_COLORS_RGB.get(beh, "black"),
                         pad=4)
        if label:
            axes[ri][0].text(-0.07, 0.5, label, transform=axes[ri][0].transAxes,
                             rotation=90, ha="right", va="center",
                             fontsize=17, fontweight="bold",
                             color=GROUP_COLORS_RGB.get(label, "black"))
    fig.tight_layout()

    out_dir = root / "outputs" / "codebook_atlas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"code_atlas_{args.run}.{ext}", dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Saved {out_dir}/code_atlas_{args.run}.pdf / .png "
             f"({len(frames)} codes, {nrow} behaviors)")


if __name__ == "__main__":
    main()
