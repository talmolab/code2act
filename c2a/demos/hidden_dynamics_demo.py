"""Demo 3 — within-syllable GRU hidden dynamics: bodies + shared directional PCA.

For one steering strategy this builds a single video that combines, per
**9-class directional category** (Walk_L/R/S, Rear_*, Turn_L/R, Immobile):

* a **20-body white-arena overlay panel** — 20 bodies executing that code from
  20 different initial conditions (the held-code rollout's K bodies), overlaid
  in the clean ``grid_gt_transitions`` light arena, tinted by the code's 9-class
  directional color; and
* a single **shared rotating hidden-PCA panel** — every selected code's
  within-syllable GRU hidden trajectory in ONE joint PCA basis, colored by the
  same 9-class directional scheme, so directional separation in hidden space is
  directly visible (shade encodes Left / Right / Straight).

The body panels are tiled into a 2-row grid (one cell per populated directional
category, so possibly more than 4); the shared hidden-PCA panel is stacked
beneath. This mirrors the reference ``hidden_dynamics_triptych.mp4`` (per-behavior
body panels + a joint hidden-PCA) generalized to the directional categories.

Reuses :mod:`c2a.demos.demo_render` (white-arena overlay, tiling, video) and
:mod:`c2a.experiments.shared.rotating_pca_video` (joint PCA), and the code-grid
label/usage machinery + 9-class color map.

Data: per-strategy ``rollouts_dir`` (held-code rollouts
``data/rollout_<pose>_code<NNN>.npz`` with ``hidden`` + ``qpos`` for K bodies,
``labels/code_labels.json``, ``data/usage_rank.npz``) and ``policy_ckpt`` (env)
from :mod:`c2a.demos.strategies`. Built for kpms + kvae; ``mlb`` drops in once
its paths are wired in ``strategies.py``.

Output:
    outputs/demos/hidden_dynamics/<strategy>.mp4

Usage:
    C2A_ROOT="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python3.12 bash_scripts/_pyrun.py c2a.demos.hidden_dynamics_demo
    # smoke:
    ... c2a.demos.hidden_dynamics_demo --strategy kpms --n-frames 8 \
        --n-bodies 4
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from c2a.codegen.shared.mlb_labels import MLB_LABELS, MLB_SHORT
from c2a.demos import demo_render
from c2a.demos.strategies import STRATEGIES, available
from c2a.experiments.shared.code_labels import load_code_labels_json
from c2a.experiments.shared.plotting import nine_class_rgb
from c2a.experiments.shared.rotating_pca_video import (
    fit_joint_pca,
    _compute_pca_limits,
    _style_pca_axes,
)
from c2a.paths import get_repo_root

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
REPO_ROOT = get_repo_root()
OUTPUT_DIR = REPO_ROOT / "outputs" / "demos" / "hidden_dynamics"
POSE = "random"

# Fallback Okabe-Ito per-behavior color for any non-canonical 9-class label
# (e.g. the auto-labeler's off-scheme "Turn_S", which has no canonical hue).
_BEHAVIOR_FALLBACK = {
    "Immobile": (0.0, 0.447, 0.698),
    "Rear": (0.0, 0.620, 0.451),
    "Turn": (0.800, 0.475, 0.655),
    "Walk": (0.835, 0.369, 0.0),
}
_DIR = {"L": "Left", "R": "Right", "S": "Straight"}


def _canonical(label: str) -> str | None:
    """Map a code-label short form (``"Walk_S"``) to a canonical MLB label.

    ``code_labels.json`` writes ``Walk_S`` / ``Turn_L`` etc., but the 9-class
    color map keys on ``Walk_Straight`` / ``Turn_Left``. Returns the canonical
    label, or ``None`` if off-scheme (e.g. ``"Turn_S"`` — Turn is L/R only).
    """
    beh, _, d = label.partition("_")
    if beh == "Immobile":
        return "Immobile"
    cand = f"{beh}_{_DIR.get(d, '')}"
    return cand if cand in MLB_LABELS else None


def _color(label: str) -> tuple[float, float, float]:
    """9-class directional color for a code label (shade encodes L/R/S)."""
    canon = _canonical(label)
    if canon is not None:
        return nine_class_rgb(canon)
    return _BEHAVIOR_FALLBACK.get(label.partition("_")[0], (0.2, 0.2, 0.2))


def _display(label: str) -> str:
    """Short display name, e.g. ``"Turn L"`` (falls back to the raw label)."""
    canon = _canonical(label)
    return MLB_SHORT.get(canon, label) if canon else label


def _order_key(label: str) -> int:
    """Canonical sort key (behavior-major, direction-minor); others last."""
    canon = _canonical(label)
    return MLB_LABELS.index(canon) if canon in MLB_LABELS else 99


def _select_codes(rollouts_dir: Path) -> list[tuple[int, str]]:
    """Most-used available code per **9-class directional** category.

    Returns ``[(code_id, nine_class_label), ...]`` in canonical 9-class order,
    one per populated category with a cached rollout npz; ``"unlabeled"`` skipped.
    """
    labels = load_code_labels_json(rollouts_dir / "labels" / "code_labels.json")
    usage = np.load(rollouts_dir / "data" / "usage_rank.npz")["usage"]
    data_dir = rollouts_dir / "data"
    avail = {
        int(p.stem.split("code")[1])
        for p in data_dir.glob(f"rollout_{POSE}_code*.npz")
    }
    # mlb: show ALL 9 codes by their definitional category so every trained
    # behavior is represented (not one-per-executed-label, which collapses).
    if "mlb" in str(rollouts_dir).lower():
        from c2a.codegen.shared.mlb_labels import MLB_LABELS
        sel = [(cid, MLB_LABELS[cid]) for cid in range(len(MLB_LABELS))
               if cid in avail]
        for cid, lab in sel:
            log.info(f"  mlb code {cid}: {_display(lab)}")
        return sel
    by_label: dict[str, list[int]] = {}
    for cid, info in labels.items():
        lab = str(info["label"])
        if lab == "unlabeled":
            continue
        by_label.setdefault(lab, []).append(int(cid))

    selected: list[tuple[int, str]] = []
    for lab in sorted(by_label, key=_order_key):
        cand = [c for c in by_label[lab] if c in avail]
        if not cand:
            continue
        best = max(cand, key=lambda c: float(usage[c]))
        selected.append((best, lab))
        log.info(f"  {_display(lab):10s}: code {best} "
                 f"(usage={float(usage[best]):.4f}, {len(cand)} avail)")
    return selected


def _load_rollout(rollouts_dir: Path, code: int):
    """Load one code's held-code rollout npz (``hidden`` + ``qpos``)."""
    p = rollouts_dir / "data" / f"rollout_{POSE}_code{code:03d}.npz"
    d = np.load(p)
    return d["hidden"], d["qpos"]  # (K, T, H), (K, T, nq)


def _render_shared_pca_panel(
    emb_list: list[np.ndarray],
    colors: list[tuple[float, float, float]],
    names: list[str],
    *,
    var: np.ndarray,
    xlim: tuple,
    ylim: tuple,
    zlim: tuple,
    n_frames: int,
    width: int,
    height: int,
) -> np.ndarray:
    """One shared rotating 3-D hidden-PCA panel for all codes -> ``(N,H,W,3)``.

    ``emb_list[i]`` is ``(K_i, T_i, 3)`` the PCA-projected hidden trajectories of
    code ``i``; all share the joint basis. Each code is drawn in its 9-class
    directional ``colors[i]`` with a legend, so directional separation in the
    shared hidden space is visible. The azimuth rotates over the clip.
    """
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                     facecolor="white")
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.92)

    out = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[i],
               markersize=5, linestyle="None", label=names[i])
        for i in range(len(names))
    ]
    for fi in range(n_frames):
        ax.cla()
        azim = 140.0 + 180.0 * fi / max(n_frames - 1, 1)
        for emb, col in zip(emb_list, colors):
            K, T, _ = emb.shape
            t = int(round((T - 1) * fi / max(n_frames - 1, 1)))
            for ki in range(K):
                traj = emb[ki, : t + 1]
                if traj.shape[0] >= 2:
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=col,
                            linewidth=0.8, alpha=0.55)
                ax.scatter([traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                           color=col, s=14, edgecolors="black", linewidths=0.3,
                           depthshade=False, zorder=10)
        _style_pca_axes(ax, var, xlim, ylim, zlim)
        ax.view_init(elev=20, azim=azim)
        ax.set_title("Shared GRU hidden-state PCA (color = 9-class direction)",
                     fontsize=15, fontweight="bold", pad=-1)
        ax.legend(handles=legend_handles, loc="upper left", fontsize=11,
                  frameon=True, framealpha=0.9, edgecolor="none",
                  handletextpad=0.2, borderpad=0.3, ncol=2)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        out[fi] = img
    plt.close(fig)
    return out


def render_strategy(strategy: str, *, n_frames: int, fps: int, n_bodies: int,
                    t_use: int, body_w: int, body_h: int) -> None:
    """Render one strategy's combined 20-body + shared-PCA hidden-dynamics video."""
    cfg = STRATEGIES[strategy]
    rollouts_dir = REPO_ROOT / cfg.rollouts_dir
    log.info(f"=== {strategy}: selecting directional codes ===")
    selected = _select_codes(rollouts_dir)
    if not selected:
        log.warning(f"  {strategy}: no labeled+available codes; skipping.")
        return

    env = demo_render.build_env(cfg.policy_ckpt)

    # Per-code: 20-body qpos (subsampled to n_frames) + hidden (for joint PCA).
    body_panels, titles, hidden_list, colors, names = [], [], [], [], []
    for code, label in selected:
        hidden, qpos = _load_rollout(rollouts_dir, code)
        K = hidden.shape[0]
        bsel = np.linspace(0, K - 1, min(n_bodies, K)).round().astype(int)
        color = _color(label)

        # Body overlay: subsample to n_frames frames, anchor each body to origin.
        fsel = np.linspace(0, min(t_use, qpos.shape[1]) - 1, n_frames)
        fsel = fsel.round().astype(int)
        body_q = [demo_render.anchor_xy(qpos[b, fsel], ref_slice=slice(None))
                  for b in bsel]
        rgba = [*color, 1.0]
        log.info(f"  [{_display(label)}] code {code}: rendering {len(body_q)} "
                 f"bodies ...")
        panel = demo_render.render_overlay_panel(
            env, body_q, color_fn=lambda i, t, _c=rgba: _c,
            width=body_w, height=body_h,
        )
        body_panels.append(panel)
        titles.append(f"{_display(label)}  (code {code})")

        hidden_list.append(hidden[bsel, : min(t_use, hidden.shape[1])])
        colors.append(color)
        names.append(_display(label))

    # Shared joint hidden-PCA across all selected codes.
    pca, var = fit_joint_pca(hidden_list, n_components=3)
    emb_all = pca.transform(
        np.concatenate([h.reshape(-1, h.shape[-1]) for h in hidden_list], axis=0)
    )
    xlim, ylim, zlim = _compute_pca_limits(emb_all)
    emb_list = [pca.transform(h.reshape(-1, h.shape[-1])).reshape(h.shape[0], -1, 3)
                for h in hidden_list]
    log.info(f"  joint hidden-PCA var explained: {var[:3]}")

    # Tile body panels into a 2-row grid (one cell per directional category).
    cols = (len(body_panels) + 1) // 2
    body_grid = demo_render.tile_grid(
        body_panels, titles, cols=cols,
        grid_title=f"{cfg.display} — codes executed by 20 bodies",
    )
    grid_h = body_grid.shape[1]

    # Shared rotating hidden-PCA panel on the RIGHT, matched to the grid height.
    pca_w = int(grid_h * 0.95)
    pca_panel = _render_shared_pca_panel(
        emb_list, colors, names, var=var, xlim=xlim, ylim=ylim, zlim=zlim,
        n_frames=n_frames, width=pca_w, height=grid_h,
    )

    T = min(body_grid.shape[0], pca_panel.shape[0])
    combined = np.concatenate([body_grid[:T], pca_panel[:T]], axis=2)
    demo_render.write_video(combined, OUTPUT_DIR / f"{strategy}.mp4", fps=fps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default=None,
                    help="single strategy (default: all available rollouts_dir).")
    ap.add_argument("--n-frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--n-bodies", type=int, default=20)
    ap.add_argument("--t-use", type=int, default=600,
                    help="frames of held-code rollout to visualize.")
    ap.add_argument("--body-w", type=int, default=440)
    ap.add_argument("--body-h", type=int, default=380)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategies = [args.strategy] if args.strategy else available("rollouts_dir")
    log.info(f"Strategies: {strategies}")
    for s in strategies:
        render_strategy(
            s, n_frames=args.n_frames, fps=args.fps, n_bodies=args.n_bodies,
            t_use=args.t_use, body_w=args.body_w, body_h=args.body_h,
        )
    log.info(f"=== Done. Outputs under {OUTPUT_DIR}/ ===")


if __name__ == "__main__":
    main()
