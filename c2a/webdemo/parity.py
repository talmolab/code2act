"""Parity of the ONNX + MuJoCo-C pipeline against the recorded JAX/MJX rollout.

Runs in ``.venv-webdemo``. Uses ``outputs/webdemo/golden_rollout.npz`` from
``export_assets`` and checks, in increasing scope:

1. **network**: ONNX(window, proprio, hidden) == JAX inference (action, carry);
2. **proprio at reset**: ``c_sim.proprioception`` on the reset state == env obs;
3. **one control step**: from the MJX state at ``t-1``, one C step with the
   recorded action reproduces the env obs at ``t`` (per-block max error);
4. **closed loop**: C rollout from the same start with the same code windows,
   reporting how long root position stays within 1 cm / 5 cm of MJX (chaotic
   divergence is expected eventually; behavior is compared in ``sim_gap``).

Usage::

    .venv-webdemo/bin/python -m c2a.webdemo.parity
"""

from __future__ import annotations

import mujoco
import numpy as np

from c2a.webdemo.c_sim import Body, Controller, ObsIndex, rollout
from c2a.webdemo.build_onnx import OUT

BLOCKS = [("joint_angles", 67), ("joint_vels", 67), ("qfrc_actuator", 73),
          ("body_height", 1), ("world_zaxis", 3), ("appendages", 15),
          ("kinematic", 9), ("touch", 4), ("prev_action", 38)]


def block_errors(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    out, i = {}, 0
    for name, n in BLOCKS:
        scale = max(np.abs(b[..., i:i + n]).max(), 1e-8)
        out[name] = float(np.abs(a[..., i:i + n] - b[..., i:i + n]).max() / scale)
        i += n
    return out


def main() -> None:
    g = dict(np.load(OUT / "golden_rollout.npz"))
    m = mujoco.MjModel.from_xml_path(str(OUT / "rodent_c2a.xml"))
    ctrl = Controller(str(OUT / "controller.onnx"))
    idx = ObsIndex.from_model(m)
    B, T = g["action"].shape[:2]
    sizes = ctrl.hidden_sizes
    split = np.cumsum(sizes)[:-1]

    # 1. Network parity on every recorded step.
    win = g["window"].reshape(B * T, -1)
    prop = g["proprio"].reshape(B * T, -1)
    hin = np.split(g["hidden_in"].reshape(B * T, -1), split, axis=1)
    action, hout, _ = ctrl(np.rint(win), prop, hin)
    hout = np.concatenate(hout, axis=1)
    ea = np.abs(action - g["action"].reshape(B * T, -1)).max()
    eh = np.abs(hout - g["hidden_out"].reshape(B * T, -1)).max()
    print(f"[1] network   max|d action| = {ea:.2e}   max|d hidden| = {eh:.2e}")

    # 2. Proprio at reset.
    p0 = np.stack([Body(m, idx, g["qpos"][b, 0]).obs() for b in range(B)])
    e0 = block_errors(p0, g["proprio"][:, 0])
    print("[2] reset obs  rel err per block: "
          + ", ".join(f"{k}={v:.1e}" for k, v in e0.items()))

    # 3. One control step from each recorded MJX state.
    errs = []
    for b in range(B):
        body = Body(m, idx, g["qpos"][b, 0])
        for t in range(2, T, 25):
            body.reset(g["qpos"][b, t - 1])
            body.d.qvel[:] = g["qvel"][b, t - 1]
            body.d.act[:] = g["act"][b, t - 1]  # actuators have filter dynamics
            mujoco.mj_forward(m, body.d)
            body.action = g["action"][b, t - 2]
            body.step(g["action"][b, t - 1])
            errs.append(block_errors(body.obs()[None], g["proprio"][b, t][None]))
    print("[3] one step   median rel err per block: "
          + ", ".join(f"{k}={np.median([e[k] for e in errs]):.1e}" for k, _ in BLOCKS))

    # 4. Closed loop from the same starts with the same windows.
    out = rollout(m, ctrl, g["qpos"][:, 0], np.rint(g["window"][0]).astype(np.int64))
    d = np.linalg.norm(out["qpos"][:, :, :3] - g["qpos"][:, :, :3], axis=-1)
    for b in range(B):
        t1 = np.argmax(d[b] > 0.01) if (d[b] > 0.01).any() else T
        t5 = np.argmax(d[b] > 0.05) if (d[b] > 0.05).any() else T
        print(f"[4] body {b}: root within 1 cm for {t1 / 100:.2f} s, "
              f"within 5 cm for {t5 / 100:.2f} s (of {T / 100:.1f} s)")


if __name__ == "__main__":
    main()
