"""Export the paper's MLB controller and its scene for the browser demo.

Runs in the project venv (JAX + track-mjx). Everything downstream (ONNX graph
construction, the MuJoCo-C simulator-gap test, the web app) consumes only the
plain files written here, so it never needs JAX.

Writes to ``outputs/webdemo/``:

* ``rodent_c2a.xml``        compiled scene exactly as the env builds it
                            (rescaled rat attached to the arena, torque
                            actuators) with the env's solver options baked in.
* ``controller_weights.npz`` policy params the browser needs (code embedding,
                            GRU stack, VAE head, frozen decoder) + proprio
                            normalizer; the Stage-1 encoder is dropped.
* ``starts.json``           the 48 start poses of the website sequence viewer
                            (same seed, same test clips, frame 20).
* ``golden_rollout.npz``    a short deterministic JAX/MJX rollout recording
                            everything the parity tests compare against.
* ``pca.json``              the 2-D GRU-state PCA basis of the sequence viewer.
* ``symbols.json``          MLB names, palette, and per-symbol round-trip recall.

Usage::

    C2A_ROOT=$PWD python -m c2a.webdemo.export_assets
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from c2a.codegen.code_sources import load_codes
from c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled import (
    _build_pose_env,
)
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint, make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.decoder_only import assert_decoder_only_settings
from c2a.paths import get_repo_root
from c2a.website.render_lib import mlb_rgb
from c2a.website.render_videos import MLB_SHORT
from c2a.website.run_sequences import PRESETS, hidden_basis, windows_for

log = logging.getLogger(__name__)
ROOT = get_repo_root()
#: The paper's MLB controller (perturbation / pairs / round-trip results).
CKPT = "model_checkpoints/260629_212713_669229_step208"
CODES = "outputs/tokenizer/mlb_v1/best_codes.npz"
OUT = ROOT / "outputs/webdemo"
PROBE_FRAME = 20

#: Symbols offered in the demo. Walk_S and Rear_S are dropped because the
#: labeler reads them as Walk_L / Rear_R (exact-symbol recall 0.00 / 0.07),
#: and Turn_L because it is almost never performed (0.02); see
#: outputs/ic_contraction/code_roundtrip/mlb/roundtrip_summary.json.
PALETTE = ["Walk_L", "Walk_R", "Rear_L", "Rear_R", "Turn_R", "Immobile"]

#: Demo presets (symbol, hold steps at 100 Hz), built only from PALETTE and
#: from switches the pair experiment reaches reliably
#: (outputs/behavior/impossible_transitions/mlb_s208_len500*/qualitative.csv).
DEMO_PRESETS = {
    "tour": ("A short tour",
             "Each switch here lands in at least three of four bodies in the "
             "paper's pair test.",
             [("Rear_R", 300), ("Turn_R", 300), ("Walk_L", 300), ("Immobile", 300),
              ("Walk_R", 300), ("Walk_L", 300)]),
    "unseen": ("Transitions the corpus never contains",
               "None of these four switches occurs in the training data.",
               [("Rear_L", 250), ("Rear_R", 250), ("Rear_L", 250), ("Walk_R", 250),
                ("Walk_L", 250)]),
    "quick": ("Rapid switching",
              "One second per symbol.",
              [("Rear_R", 100), ("Turn_R", 100), ("Walk_L", 100), ("Immobile", 100),
               ("Walk_R", 100), ("Walk_L", 100), ("Immobile", 100), ("Walk_R", 100)]),
}


def build_env_and_starts(cfg, n_starts: int = 48):
    """Rebuild the website sequence viewer's env and start clips (seed 0)."""
    clip_len = int(cfg.env_config.clip_length)
    stack = int(cfg.network_config.get("code_stack_size", 1))
    splits = load_balanced_splits(str(ROOT / "data/rodent/rodent_balanced_splits.json"))
    train_idx = np.asarray(splits["balanced"]["train_indices"], np.int64)
    test_idx = np.asarray(splits["balanced"]["test_indices"], np.int64)
    payload = load_codes(cfg, train_idx, test_idx, clip_len)
    rng = np.random.RandomState(0)
    starts = rng.choice(test_idx, size=n_starts, replace=False)
    test_pos = {int(c): i for i, c in enumerate(test_idx)}
    codes_for = np.stack([payload.test_codes[test_pos[int(c)]] for c in starts])
    env = _build_pose_env(cfg, str(ROOT / "data/rodent/rodent_reference_clips.h5"),
                          clip_len, keep_clips_idx=starts, test_codes=codes_for,
                          code_stack_size=stack, probe_frame=PROBE_FRAME)
    return env, starts, stack


def export_scene(env) -> mujoco.MjModel:
    """Write the compiled scene with the env's runtime solver options baked in."""
    m = env.mj_model
    spec = env._spec.copy()
    spec.option.timestep = float(m.opt.timestep)
    spec.option.solver = int(m.opt.solver)
    spec.option.iterations = int(m.opt.iterations)
    spec.option.ls_iterations = int(m.opt.ls_iterations)
    spec.option.noslip_iterations = int(m.opt.noslip_iterations)
    xml = spec.to_xml()
    m2 = mujoco.MjModel.from_xml_string(xml)
    # The re-loaded model must be the model the controller was trained on.
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "nsensordata"):
        assert getattr(m2, name) == getattr(m, name), name
    for field in ("body_mass", "body_pos", "body_quat", "body_inertia", "geom_pos",
                  "geom_quat", "geom_rbound", "geom_friction", "geom_solref",
                  "geom_solimp", "geom_contype", "geom_conaffinity",
                  "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
                  "actuator_gear", "jnt_range", "dof_damping", "dof_armature",
                  "dof_frictionloss", "sensor_adr"):
        np.testing.assert_allclose(getattr(m2, field), getattr(m, field),
                                   rtol=1e-5, atol=1e-6, err_msg=field)
    # XML only stores the size slots a geom type uses (capsule: 2, sphere: 1).
    used = {mujoco.mjtGeom.mjGEOM_SPHERE: 1, mujoco.mjtGeom.mjGEOM_CAPSULE: 2,
            mujoco.mjtGeom.mjGEOM_CYLINDER: 2}
    for g in range(m.ngeom):
        k = used.get(mujoco.mjtGeom(m.geom_type[g]), 3)
        np.testing.assert_allclose(m2.geom_size[g, :k], m.geom_size[g, :k],
                                   rtol=1e-5, atol=1e-7, err_msg=f"geom_size[{g}]")
    for name in ("timestep", "iterations", "ls_iterations", "solver", "noslip_iterations",
                 "cone", "impratio", "tolerance", "ls_tolerance", "disableflags",
                 "enableflags", "integrator"):
        assert np.all(getattr(m2.opt, name) == getattr(m.opt, name)), f"opt.{name}"
    (OUT / "rodent_c2a.xml").write_text(xml)
    log.info(f"scene: nq={m.nq} nv={m.nv} nu={m.nu} timestep={m.opt.timestep} "
             f"solver={m.opt.solver} it={m.opt.iterations}/{m.opt.ls_iterations} "
             f"integrator={m.opt.integrator} cone={m.opt.cone}")
    return m


def export_weights(norm_state, policy_params) -> None:
    """Flatten the action-path params (no encoder) + proprio normalizer."""
    p = policy_params["params"]
    flat: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(p)[0]:
        key = "/".join(k.key for k in path)
        if key.startswith("encoder_module"):
            continue
        flat[key] = np.asarray(leaf, np.float32)
    ps = norm_state.proprioception
    flat["norm/proprio_mean"] = np.asarray(ps.mean, np.float32)
    flat["norm/proprio_std"] = np.asarray(ps.std, np.float32)
    np.savez(OUT / "controller_weights.npz", **flat)
    n = sum(v.size for k, v in flat.items() if not k.startswith("norm/"))
    log.info(f"weights: {len(flat)} arrays, {n / 1e6:.2f}M params")


def export_starts(env, starts: np.ndarray) -> None:
    """Reset each start clip at the probe frame and record the initial pose."""
    reset = jax.jit(jax.vmap(
        lambda i: env.reset(jax.random.PRNGKey(0), clip_idx=i).data.qpos))
    poses = np.asarray(reset(jnp.arange(len(starts), dtype=jnp.int32)), np.float64)
    poses = poses.round(6).tolist()
    (OUT / "starts.json").write_text(json.dumps({
        "probe_frame": PROBE_FRAME, "clip_ids": [int(c) for c in starts],
        "qpos": poses}))


def export_golden(env, inf_fn, params, nets, stack: int, n_bodies: int, n_steps: int):
    """Deterministic MJX rollout of the 'tour' preset, recording every input."""
    win, _ = windows_for(PRESETS["tour"][2], stack)
    win = jnp.asarray(win[:n_steps])

    def single_body(clip_idx):
        state0 = env.reset(jax.random.PRNGKey(0), clip_idx=clip_idx)
        hidden0 = nets.policy_network.init_hidden(1)

        def step(carry, window):
            state, hidden = carry
            obs = OrderedDict(state.obs)
            obs["kpms_code"] = window
            state = state.replace(obs=obs)
            batched = jax.tree.map(lambda x: x[None], state.obs)
            action, _, new_hidden = inf_fn(params, batched, hidden, jax.random.PRNGKey(0))
            action = action[0]
            out = {
                "qpos": state.data.qpos, "qvel": state.data.qvel, "act": state.data.act,
                "proprio": state.obs["proprioception"], "window": window,
                "action": action,
                "hidden_in": jnp.concatenate([h[0] for h in hidden]),
                "hidden_out": jnp.concatenate([h[0] for h in new_hidden]),
            }
            return (env.step(state, action), new_hidden), out

        _, outs = jax.lax.scan(step, (state0, hidden0), win)
        return outs

    outs = jax.jit(jax.vmap(single_body))(jnp.arange(n_bodies, dtype=jnp.int32))
    agg = {k: np.asarray(v) for k, v in outs.items()}
    np.savez_compressed(OUT / "golden_rollout.npz", **agg)
    log.info(f"golden: {n_bodies} bodies x {n_steps} steps, "
             f"finite={np.isfinite(agg['qpos']).all()}")


def export_meta() -> None:
    """PCA basis of the sequence viewer and per-symbol metadata."""
    pca, clouds = hidden_basis()
    (OUT / "pca.json").write_text(json.dumps({
        "mean": np.round(pca.mean_, 6).tolist(),
        "components": np.round(pca.components_[:2], 6).tolist(),
        "explained": np.round(pca.explained_variance_ratio_[:2], 3).tolist(),
        "clouds": clouds}))
    rt = json.loads((ROOT / "outputs/ic_contraction/code_roundtrip/mlb/"
                     "roundtrip_summary.json").read_text())["conditions"]["mlb"]["codes"]
    symbols = []
    for c, name in enumerate(MLB_SHORT):
        r = rt[str(c)]
        symbols.append({
            "code": c, "name": name, "color": list(mlb_rgb(c)),
            "behavior": r["driving_behavior"],
            "behavior_recall": round(float(r["behavior_recall"]), 3),
            "symbol_recall": round(float(r["category_recall"]), 3),
        })
    as_json = lambda presets: {k: {"title": t, "blurb": b, "blocks": [[s, h] for s, h in seq]}
                               for k, (t, b, seq) in presets.items()}
    (OUT / "symbols.json").write_text(json.dumps({
        "symbols": symbols,
        "palette": PALETTE,
        "presets": as_json(DEMO_PRESETS),
        "website_presets": as_json(PRESETS),
        "source": "outputs/ic_contraction/code_roundtrip/mlb/roundtrip_summary.json"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden-bodies", type=int, default=4)
    ap.add_argument("--golden-steps", type=int, default=600)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg, norm_state, policy_params, nets = load_moseq_checkpoint(str(ROOT / CKPT))
    cfg.kpms_config.codes_path = str(ROOT / CODES)
    assert_decoder_only_settings(0.0, deterministic=True)
    inf_fn = make_inference_fn(nets, use_rnn=True, deterministic=True, z_e_scale=0.0)
    params = (norm_state, policy_params)

    env, starts, stack = build_env_and_starts(cfg)
    export_scene(env)
    export_weights(norm_state, policy_params)
    export_starts(env, starts)
    export_golden(env, inf_fn, params, nets, stack, args.golden_bodies, args.golden_steps)
    export_meta()
    log.info(f"wrote {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
