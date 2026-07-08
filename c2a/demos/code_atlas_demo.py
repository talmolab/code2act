"""Demo video: the code atlas as motion (video version of Figs 14/15/16).

Per steering strategy, a behavior-grouped GRID of codes. Each cell shows **5
bodies holding that single code simultaneously**, animated on a clean WHITE
arena, so a coherent code reads as five walkers moving together. Rows are
behavior groups (Immobile / Rear / Turn / Walk); within a group the most-used
codes are shown. Each cell is titled with the code id and its 9-class category.

Data: the per-strategy ``rollouts_dir`` written by
``c2a.experiments.code_grid.run_code_grid_rollouts`` — held-code rollouts
``data/rollout_random_code<NNN>.npz`` (``qpos`` of shape ``(K, T, nq)``) and
``labels/code_labels.json`` (per-code behavior / direction / usage). Only codes
that have a rollout file are used.

``mlb`` is skipped until its checkpoint / data exist (filtered out by
``strategies.available("rollouts_dir")``).

Usage::

    C2A_ROOT="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
      python3.12 bash_scripts/_pyrun.py c2a.demos.code_atlas_demo
    # ... --strategy kvae --per-group 4
"""
from __future__ import annotations

import argparse
import glob
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


def _codes_with_rollouts(data_dir: Path) -> dict[int, Path]:
    out = {}
    for f in sorted(glob.glob(str(data_dir / "rollout_random_code*.npz"))):
        cid = int(os.path.basename(f).split("code")[1].split(".")[0])
        out[cid] = Path(f)
    return out


def _select_codes(labels: dict, have: dict[int, Path], per_group: int
                  ) -> list[tuple[int | None, str]]:
    """Behavior-grouped, usage-ranked code selection.

    Returns ``(cid, title)`` rows in behavior order, ``per_group`` per behavior
    (short groups padded with ``(None, "")`` blanks so each behavior fills one
    grid row). ``title`` is e.g. ``"code 60 · Immobile_S"``.
    """
    behs = labels["behaviors"]
    rows: list[tuple[int | None, str]] = []
    for b in range(len(behs)):
        grp = [
            (cid, info) for cid, info in (
                (c, labels["codes"].get(str(c))) for c in have
            )
            if info is not None and info["behavior"] == b
        ]
        if not grp:
            continue  # behavior absent for this strategy -> no empty row
        grp.sort(key=lambda ci: -ci[1].get("frames_total", 0))
        chosen = grp[:per_group]
        for cid, info in chosen:
            rows.append((cid, f"code {cid} · {info['label']}"))
        rows += [(None, "")] * (per_group - len(chosen))  # pad to a full row
    return rows


def _pick_bodies(npz: Path, window: slice, n: int = 5) -> list[int]:
    """``n`` bodies that execute the held code cleanest (highest mean reward)."""
    with np.load(npz) as d:
        r = d["rewards"][:, window].mean(axis=1)
    return [int(b) for b in np.argsort(-r)[:n]]


def _render_cell(env, npz: Path, *, window: slice, step: int, n_bodies: int,
                 width: int, height: int) -> np.ndarray:
    bodies_idx = _pick_bodies(npz, window, n_bodies)
    with np.load(npz) as d:
        q = np.asarray(d["qpos"], dtype=np.float64)  # (K, T, nq)
    frames = np.arange(window.start, window.stop, step)
    bodies = [
        DR.anchor_xy(q[b, frames], ref_slice=slice(0, len(frames)))
        for b in bodies_idx
    ]
    teal = list(DR.TEAL_RGBA)

    def color_fn(i: int, t: int):
        return teal

    return DR.render_overlay_panel(
        env, bodies, color_fn, width=width, height=height,
        camera_elevation=-26.0, camera_azimuth=-120.0, camera_fovy=50.0,
    )


def run_strategy(name: str, per_group: int, fps: int) -> Path | None:
    cfg = STRATEGIES[name]
    root = get_repo_root()
    rdir = root / cfg.rollouts_dir
    have = _codes_with_rollouts(rdir / "data")
    labels = json.loads((rdir / "labels" / "code_labels.json").read_text())
    if name == "mlb":
        # mlb is a fixed 9-code codebook — show ALL 9 codes, one per cell, in
        # code-id order with their *definitional* labels (the 3x3 codebook
        # layout), rather than the behavior-grouped/capped view used for the
        # many-code data-discovered strategies.
        from c2a.codegen.shared.mlb_labels import MLB_LABELS, MLB_SHORT
        rows = [(cid, f"code {cid} · {MLB_SHORT[MLB_LABELS[cid]]}")
                for cid in range(len(MLB_LABELS)) if cid in have]
        cols = 3
    else:
        rows = _select_codes(labels, have, per_group)
        cols = per_group
    log.info("[%s] %d cells (%d/row): %s", name, len(rows), cols,
             [r[0] for r in rows])

    env = DR.build_env(cfg.policy_ckpt)
    window = slice(400, 1000)
    step = 8
    cell_w, cell_h = 680, 640

    panels, titles = [], []
    blank = None
    for cid, title in rows:
        if cid is None:
            if blank is None:
                # reuse the previous cell's shape for a white placeholder.
                ref = panels[-1] if panels else None
                if ref is not None:
                    blank = np.full_like(ref, 255)
            panels.append(blank if blank is not None
                          else np.full((1, cell_h, cell_w, 3), 255, np.uint8))
            titles.append("")
            continue
        log.info("[%s]   rendering %s", name, title)
        panel = _render_cell(env, have[cid], window=window, step=step,
                             n_bodies=5, width=cell_w, height=cell_h)
        panels.append(panel)
        titles.append(title)
        if blank is None:
            blank = np.full_like(panel, 255)

    video = DR.tile_grid(
        panels, titles, cols=cols,
        grid_title=f"{cfg.display} — code atlas (5 bodies hold each code)",
        banner_h=36, font_size=24,
    )
    out = root / "outputs" / "demos" / "code_atlas" / f"{name}.mp4"
    DR.write_video(video, out, fps=fps)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default=None,
                   help="one strategy (kpms|kvae); default = all available")
    p.add_argument("--per-group", type=int, default=4,
                   help="codes per behavior row (grid columns)")
    p.add_argument("--fps", type=int, default=18)
    args = p.parse_args()

    names = [args.strategy] if args.strategy else available("rollouts_dir")
    log.info("strategies: %s", names)
    for name in names:
        out = run_strategy(name, args.per_group, args.fps)
        if out:
            log.info("[%s] wrote %s", name, out)


if __name__ == "__main__":
    main()
