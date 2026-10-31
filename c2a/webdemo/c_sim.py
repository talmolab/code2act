"""MuJoCo-C + ONNX reimplementation of the paper's decoder-only rollout.

This is the executable spec for the browser: it uses only ``mujoco`` (the C
engine that the WASM build compiles), ``onnxruntime`` and numpy, and it builds
the 277-d proprioception from plain ``MjData`` fields in exactly the order
``vnl_playground`` ``_get_proprioception`` does. The TypeScript port mirrors
this file line for line.

Two quirks of the training env are reproduced on purpose:

* the "actuator_ctrl" slot is ``data.qfrc_actuator`` (nv = 73 values);
* the "prev_action" slot lags two steps: after stepping with ``a_t`` the env
  sets ``prev_action = a_{t-1}``, so the observation at step ``t`` carries
  ``a_{t-2}`` (zeros for the first two steps).

Observation fields are read straight after ``mj_step`` (no extra
``mj_forward``), as MJX does, so positions and sensors reflect the start of the
last physics substep.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import onnxruntime as ort

SUFFIX = "-rodent"
END_EFFECTORS = ["lower_arm_R", "lower_arm_L", "foot_R", "foot_L", "skull"]
TOUCH_SENSORS = ["palm_L", "palm_R", "sole_L", "sole_R"]
KINEMATIC_SENSORS = ["accelerometer", "velocimeter", "gyro"]
CTRL_DT = 0.01
PROPRIO_SIZE = 277


@dataclass
class ObsIndex:
    """Model ids needed to assemble the proprioception vector."""

    torso: int
    walker: int
    end_effectors: list[int]
    kinematic: list[tuple[int, int]]  # (adr, dim) per sensor
    touch: list[tuple[int, int]]

    @classmethod
    def from_model(cls, m: mujoco.MjModel) -> "ObsIndex":
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n + SUFFIX)

        def sensor(n):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n + SUFFIX)
            assert i >= 0, n
            return int(m.sensor_adr[i]), int(m.sensor_dim[i])

        idx = cls(torso=body("torso"), walker=body("walker"),
                  end_effectors=[body(n) for n in END_EFFECTORS],
                  kinematic=[sensor(n) for n in KINEMATIC_SENSORS],
                  touch=[sensor(n) for n in TOUCH_SENSORS])
        assert min(idx.torso, idx.walker, *idx.end_effectors) >= 0
        return idx


def proprioception(m: mujoco.MjModel, d: mujoco.MjData, idx: ObsIndex,
                   prev_action: np.ndarray) -> np.ndarray:
    """The 277-d proprioception, in ``_get_proprioception`` order."""
    torso_pos = d.xpos[idx.torso]
    torso_mat = d.xmat[idx.torso].reshape(3, 3)
    parts = [
        d.qpos[7:],                                   # joint_angles (67)
        d.qvel[6:],                                   # joint_ang_vels (67)
        d.qfrc_actuator,                              # actuator_ctrl (nv=73)
        torso_pos[2:3],                               # body_height (1)
        d.xmat[idx.walker][6:9],                      # world_zaxis (3)
    ]
    parts += [(d.xpos[b] - torso_pos) @ torso_mat for b in idx.end_effectors]  # 15
    parts += [d.sensordata[a:a + n] for a, n in idx.kinematic]                 # 9
    parts += [d.sensordata[a:a + n] for a, n in idx.touch]                     # 4
    parts.append(prev_action)                                                  # 38
    out = np.nan_to_num(np.concatenate(parts).astype(np.float32))
    assert out.shape == (PROPRIO_SIZE,), out.shape
    return out


class Controller:
    """``controller.onnx`` with the GRU carry held on the Python side."""

    def __init__(self, onnx_path: str, threads: int = 0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.hidden_names = [i.name for i in self.sess.get_inputs() if i.name.startswith("h")]
        self.hidden_sizes = [i.shape[1] for i in self.sess.get_inputs()
                             if i.name.startswith("h")]

    def zeros(self, batch: int) -> list[np.ndarray]:
        return [np.zeros((batch, s), np.float32) for s in self.hidden_sizes]

    def __call__(self, codes: np.ndarray, proprio: np.ndarray, hidden: list[np.ndarray]):
        feeds = {"codes": codes.astype(np.int64), "proprio": proprio.astype(np.float32)}
        feeds.update(dict(zip(self.hidden_names, hidden)))
        outs = self.sess.run(None, feeds)
        return outs[0], outs[1:1 + len(hidden)], outs[-1]


class Body:
    """One simulated rat: MjData plus the env's action bookkeeping."""

    def __init__(self, m: mujoco.MjModel, idx: ObsIndex, qpos0: np.ndarray):
        self.m, self.idx = m, idx
        self.d = mujoco.MjData(m)
        self.n_sub = int(round(CTRL_DT / m.opt.timestep))
        self.reset(qpos0)

    def reset(self, qpos0: np.ndarray) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos0
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.action = np.zeros(self.m.nu, np.float32)       # info["action"]
        self.prev_action = np.zeros(self.m.nu, np.float32)  # info["prev_action"]

    def obs(self) -> np.ndarray:
        return proprioception(self.m, self.d, self.idx, self.prev_action)

    def step(self, action: np.ndarray) -> None:
        self.d.ctrl[:] = action
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
        self.prev_action, self.action = self.action, np.asarray(action, np.float32)


def code_windows(commanded: np.ndarray, stack: int, convention: str) -> np.ndarray:
    """Per-control-step forward code windows ``[T, stack]``.

    ``"ctrl"``: next ``stack`` control steps, padded with the last symbol
    (``run_sequences.windows_for`` / ``checkpoint_utils.run_rollout``; every
    paper and website rollout).
    ``"mocap"``: next ``stack`` 50 Hz reference frames, two control steps per
    frame (``MoSeqImitation._get_obs`` during training).
    """
    T = len(commanded)
    if convention == "ctrl":
        padded = np.concatenate([commanded, np.full(stack, commanded[-1])])
        return np.stack([padded[t:t + stack] for t in range(T)])
    if convention == "mocap":
        frames = commanded[::2]
        padded = np.concatenate([frames, np.full(stack, frames[-1])])
        return np.stack([padded[t // 2:t // 2 + stack] for t in range(T)])
    raise ValueError(convention)


def rollout(m: mujoco.MjModel, ctrl: Controller, starts: np.ndarray,
            windows: np.ndarray) -> dict[str, np.ndarray]:
    """Batched closed-loop rollout of ``len(starts)`` bodies through ``windows``."""
    idx = ObsIndex.from_model(m)
    bodies = [Body(m, idx, q) for q in starts]
    B, T = len(bodies), len(windows)
    hidden = ctrl.zeros(B)
    qpos = np.zeros((B, T, m.nq), np.float32)
    last_h = np.zeros((B, T, ctrl.hidden_sizes[-1]), np.float32)
    for t in range(T):
        proprio = np.stack([b.obs() for b in bodies])
        codes = np.repeat(windows[t][None], B, axis=0)
        action, hidden, _ = ctrl(codes, proprio, hidden)
        for i, b in enumerate(bodies):
            qpos[i, t] = b.d.qpos
            b.step(action[i])
        last_h[:, t] = hidden[-1]
    return {"qpos": qpos, "hidden": last_h}
