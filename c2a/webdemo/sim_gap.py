"""Simulator-gap rollouts: the website presets in MuJoCo C + ONNX.

Runs in ``.venv-webdemo``. Replays the three sequence-viewer presets from the
same 48 start poses as ``c2a.website.run_sequences`` (which used MJX), under
both code-window conventions, and saves qpos/hidden for scoring with
``c2a.webdemo.score_sim_gap`` (project venv, which owns the labeler).

Usage::

    .venv-webdemo/bin/python -m c2a.webdemo.sim_gap [--workers 6] [--set demo]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import mujoco
import numpy as np

from c2a.webdemo.build_onnx import OUT
from c2a.webdemo.c_sim import Controller, code_windows, rollout

GAP_DIR = OUT / "sim_gap"


def commanded_stream(blocks: list[list], names: list[str]) -> np.ndarray:
    ids = {n: i for i, n in enumerate(names)}
    return np.concatenate([np.full(h, ids[s], np.int64) for s, h in blocks])


def run_one(job: tuple[str, str, str]) -> str:
    preset_set, preset, convention = job
    meta = json.loads((OUT / "symbols.json").read_text())
    names = [s["name"] for s in meta["symbols"]]
    starts = np.asarray(json.loads((OUT / "starts.json").read_text())["qpos"])
    m = mujoco.MjModel.from_xml_path(str(OUT / "rodent_c2a.xml"))
    ctrl = Controller(str(OUT / "controller.onnx"), threads=2)
    cmd = commanded_stream(meta[preset_set][preset]["blocks"], names)
    stack = int(ctrl.sess.get_modelmeta().custom_metadata_map["code_stack_size"])
    out = rollout(m, ctrl, starts, code_windows(cmd, stack, convention))
    prefix = "demo_" if preset_set == "presets" else ""
    path = GAP_DIR / f"{prefix}{preset}_{convention}.npz"
    np.savez_compressed(path, **out, commanded=cmd)
    return f"{path.name}: finite={np.isfinite(out['qpos']).all()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", choices=["website", "demo"], default="website",
                    help="website: the project page's presets (MJX reference exists), "
                         "both window conventions; demo: the demo's own presets")
    args = ap.parse_args()
    GAP_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "symbols.json").read_text())
    if args.set == "website":
        jobs = [("website_presets", p, c) for p in meta["website_presets"]
                for c in ("ctrl", "mocap")]
    else:
        jobs = [("presets", p, "ctrl") for p in meta["presets"]]
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(run_one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
