"""Fixtures for the web app's TypeScript tests (closed loop, MuJoCo C + ONNX).

Runs in ``.venv-webdemo``. Rolls one body from start pose 0 through the
'tour' preset with ``c_sim`` and records, per control step, the code window,
the proprioception, the action and qpos. The TypeScript runner must reproduce
these (WASM MuJoCo is bit-identical to native C; ORT kernels differ by ~1e-6).

Usage::

    .venv-webdemo/bin/python -m c2a.webdemo.export_fixtures [--steps 300]
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from c2a.webdemo.build_onnx import OUT
from c2a.webdemo.c_sim import Body, Controller, ObsIndex, code_windows
from c2a.webdemo.sim_gap import commanded_stream


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    meta = json.loads((OUT / "symbols.json").read_text())
    names = [s["name"] for s in meta["symbols"]]
    q0 = np.asarray(json.loads((OUT / "starts.json").read_text())["qpos"][0])
    m = mujoco.MjModel.from_xml_path(str(OUT / "rodent_c2a.xml"))
    ctrl = Controller(str(OUT / "controller.onnx"))
    win = code_windows(commanded_stream(meta["presets"]["tour"]["blocks"], names),
                       5, "ctrl")[:args.steps]
    body, hidden = Body(m, ObsIndex.from_model(m), q0), ctrl.zeros(1)
    rec = {"window": [], "proprio": [], "action": [], "qpos": []}
    for t in range(args.steps):
        p = body.obs()
        action, hidden, _ = ctrl(win[t][None], p[None], hidden)
        rec["window"].append(win[t].tolist())
        rec["proprio"].append(np.round(p, 7).tolist())
        rec["action"].append(np.round(action[0], 7).tolist())
        rec["qpos"].append(np.round(body.d.qpos, 9).tolist())
        body.step(action[0])
    (OUT / "fixture_tour.json").write_text(json.dumps({"start": 0, **rec}))
    print(f"wrote fixture_tour.json ({args.steps} steps)")


if __name__ == "__main__":
    main()
