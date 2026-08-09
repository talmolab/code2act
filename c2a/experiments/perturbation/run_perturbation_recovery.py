"""Experiment: perturbation–recovery (codes as *defended* attractors).

IC-contraction (``run_ic_contraction``) showed that a held code pulls bodies
started from diverse poses onto a common goal pose. That is a statement about
*initial conditions*. This experiment makes the stronger, interventional claim:
a held code is an attractor the controller *actively defends*. We let bodies
converge under a held code, then INJECT an external velocity impulse mid-rollout
(a "shove") and ask whether the controller drives the body back onto the
code-specific attractor.

Two properties make this the experiment the paper's thesis needs:

* It is a genuine *intervention*. A descriptive behavioral token cannot be
  perturbed; an executable one can. Holding a code and shoving the body is a
  causal manipulation only possible because the code drives a physical
  controller.
* It is *non-circular*. The goal pose is defined from the UNPERTURBED (amp=0)
  rollouts, independently of the perturbed trajectory. The body is pushed AWAY
  from that pose and must return to it. IC-contraction, by contrast, defines the
  goal from the same rollouts it scores.

Decoder-only (``z_e_scale=0``, deterministic): the discrete code is the only
time-varying input, so any recovery is attributable to the code's learned
dynamics, not the continuous encoder.

Per (code, amplitude) we roll out ``K`` bodies, apply the impulse at
``t_perturb`` (after convergence), and measure, in pose space (root-stripped,
yaw-removed qpos) and GRU-hidden space, the phase-invariant distance to the
amp=0 attractor cloud over time, ``d(t)``. Recovery = ``d`` jumps at the kick
then decays back toward its pre-kick floor. Controls:

* amplitude sweep — recovery vs. shove magnitude maps the basin.
* wrong-code control — distance of the perturbed bodies to a *different* code's
  attractor cloud stays high (recovery is code-specific, not a generic
  righting reflex).

Run::

    cd code2act
    python -m c2a.experiments.perturbation.run_perturbation_recovery
    # quick smoke test:
    python -m c2a.experiments.perturbation.run_perturbation_recovery \
        rollout.K=4 rollout.hold_frames=200 rollout.t_perturb=120 \
        amplitudes='[0.0,2.4]' codes='[2,9]'
"""
from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from c2a.paths import get_repo_root  # noqa: E402
from c2a.codegen.code_sources import load_codes  # noqa: E402
from c2a.codegen.shared.label_alignment import BEHAVIORS  # noqa: E402
from c2a.experiments.shared.checkpoint_utils import (  # noqa: E402
    load_moseq_checkpoint, make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits  # noqa: E402
from c2a.experiments.shared.code_grid import rank_codes_by_usage  # noqa: E402
from c2a.experiments.shared.decoder_only import (  # noqa: E402
    assert_decoder_only_settings, describe_decoder_only_mode,
)
from c2a.experiments.code_grid.run_code_grid_rollouts import (  # noqa: E402
    _build_pose_env, _pick_random_clips,
)
from c2a.experiments.ic_contraction.run_ic_contraction import pose_features  # noqa: E402

log = logging.getLogger(__name__)
REPO_ROOT = get_repo_root()

# Coarse-behavior line colors, matched to the 9-class directional hues of the
# confusion matrix (Fig. contraction a): Walk=blue, Rear=green, Turn=orange,
# Immobile=grey (see c2a.codegen.shared.mlb_labels.MLB_COLORS).
BEHAVIOR_COLORS = {
    "Immobile": "#8c8c8c", "Rear": "#40a659",
    "Turn": "#d9731f", "Walk": "#4d80d9", "Other": "#cccccc",
}


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Perturbed rollout (mirror of code_grid._batched_rollout_for_code + impulse)
# ---------------------------------------------------------------------------


def _perturbed_rollout(
    env, inf_fn, params, ppo_networks,
    code, amp, clip_indices, base_keys,
    hold_frames: int, t_perturb: int, perturb_dur: int, code_stack_size: int,
) -> dict[str, jax.Array]:
    """Hold ``code`` for ``hold_frames``; add an impulse ``amp`` to qvel over
    ``[t_perturb, t_perturb+perturb_dur)``.

    Identical PRNG threading / per-step ops to the code-grid held-code rollout,
    plus a per-body fixed random impulse *direction* (so ``amp`` is the only
    thing that varies across amplitudes — a paired design). The impulse is added
    to the generalized velocity *after* the controller has chosen its action for
    that step, so the body first deflects and the controller reacts on the next
    step (a one-step sensory delay, as in a real shove).

    Returns ``{hidden, qpos, reward}`` each ``(K, hold_frames, ...)``.
    """
    held_code = jnp.full(
        (code_stack_size,), jnp.asarray(code, dtype=jnp.float32),
    )

    def single_body(base_key, clip_idx):
        key, reset_key, dir_key = jax.random.split(base_key, 3)
        state0 = env.reset(reset_key, clip_idx=clip_idx)
        hidden0 = ppo_networks.policy_network.init_hidden(1)
        nv = state0.data.qvel.shape[-1]
        # Per-body fixed unit-variance impulse direction (reused across amps).
        pdir = jax.random.normal(dir_key, (nv,))

        def step(carry, t):
            state, hidden, key = carry
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = held_code
            state = state.replace(obs=new_obs)
            key, subkey = jax.random.split(key)
            batched_obs = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inf_fn(params, batched_obs, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)
            qpos_t = state.data.qpos
            hid_t = hidden[-1][0]
            # Inject the velocity impulse for the perturbation window.
            in_window = (t >= t_perturb) & (t < t_perturb + perturb_dur)
            dqvel = jnp.where(in_window, amp * pdir, 0.0)
            state = state.replace(
                data=state.data.replace(qvel=state.data.qvel + dqvel)
            )
            new_state = env.step(state, action)
            out = {"hidden": hid_t, "qpos": qpos_t, "reward": new_state.reward}
            return (new_state, hidden, key), out

        _, outs = jax.lax.scan(
            step, (state0, hidden0, key), jnp.arange(hold_frames),
        )
        return outs

    return jax.vmap(single_body)(base_keys, clip_indices)


# ---------------------------------------------------------------------------
# Code selection
# ---------------------------------------------------------------------------


def _base_behavior(label: str) -> str:
    """'Walk_L' -> 'Walk'; 'unlabeled' -> 'Other'."""
    if not label or label == "unlabeled":
        return "Other"
    return label.split("_")[0]


def _auto_pick_codes(rollouts_base: Path, run: str) -> tuple[list[int], dict[int, str]]:
    """Highest-usage code per rollout-derived behavior (Immobile/Rear/Walk/Turn)."""
    run_dir = rollouts_base / run
    labels = json.loads((run_dir / "labels" / "code_labels.json").read_text())
    usage = np.load(run_dir / "data" / "usage_rank.npz")["usage"]
    by_beh: dict[str, list[int]] = {}
    for cid, rec in labels.get("codes", {}).items():
        beh = _base_behavior(str(rec.get("label", "")))
        if beh == "Other":
            continue
        by_beh.setdefault(beh, []).append(int(cid))
    picked: list[int] = []
    code_beh: dict[int, str] = {}
    for beh in ["Immobile", "Rear", "Walk", "Turn"]:
        cands = by_beh.get(beh, [])
        if not cands:
            continue
        best = max(cands, key=lambda c: usage[c] if c < len(usage) else 0)
        picked.append(best)
        code_beh[best] = beh
    return picked, code_beh


# ---------------------------------------------------------------------------
# Recovery metrics
# ---------------------------------------------------------------------------


def _features(qpos_or_hidden: np.ndarray, space: str) -> np.ndarray:
    if space == "pose":
        return pose_features(qpos_or_hidden.astype(np.float32))
    return qpos_or_hidden.astype(np.float32)


def _build_cloud(feat0_z: np.ndarray, cloud_idx: np.ndarray,
                 pca_dim: int, knn: int, subsample: int,
                 rng: np.random.RandomState):
    """PCA + kNN over the amp=0 attractor cloud (pooled late-window states)."""
    M = feat0_z[:, cloud_idx].reshape(-1, feat0_z.shape[-1])
    if M.shape[0] > subsample:
        M = M[rng.choice(M.shape[0], subsample, replace=False)]
    dim = min(pca_dim, M.shape[1], M.shape[0] - 1)
    pca = PCA(n_components=dim).fit(M)
    nn = NearestNeighbors(n_neighbors=min(knn, M.shape[0])).fit(pca.transform(M))
    return pca, nn


def _dist_curve(feat_z: np.ndarray, pca, nn, grid: np.ndarray) -> np.ndarray:
    """Mean over bodies of the mean-kNN distance to the cloud, per grid time."""
    d = np.empty(grid.shape[0], dtype=np.float32)
    for i, t in enumerate(grid):
        dist, _ = nn.kneighbors(pca.transform(feat_z[:, t]))
        d[i] = float(dist.mean())
    return d


def _recovery_metrics(d: np.ndarray, grid: np.ndarray, t_perturb: int,
                      a: DictConfig) -> dict:
    """Floor / peak / final / recovery fraction + time from a d(t) curve."""
    pre = grid < t_perturb
    post = grid >= t_perturb
    peak_mask = post & (grid < t_perturb + int(a.peak_window))
    final_mask = grid >= grid[-1] - max(2, int(0.1 * len(grid)))
    floor = float(np.median(d[pre])) if pre.any() else float(d[0])
    if not peak_mask.any():
        return {"floor": floor, "peak": floor, "final": floor,
                "recovery_frac": float("nan"), "recovery_time": float("nan"),
                "recovered": False}
    peak = float(np.max(d[peak_mask]))
    final = float(np.median(d[final_mask]))
    span = peak - floor
    if span <= 1e-9:
        rec_frac = float("nan")
        rec_time = 0.0
        recovered = True
    else:
        rec_frac = float(np.clip((peak - final) / span, 0.0, 1.0))
        thresh = floor + float(a.recovery_tol) * span
        peak_i = int(np.argmax(np.where(peak_mask, d, -np.inf)))
        after = grid[peak_i:]
        below = np.where(d[peak_i:] <= thresh)[0]
        rec_time = float(after[below[0]] - t_perturb) if below.size else float("nan")
        recovered = bool(final <= thresh)
    return {"floor": floor, "peak": peak, "final": final,
            "recovery_frac": rec_frac, "recovery_time": rec_time,
            "recovered": recovered}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_curves(per_code: dict, space: str, t_perturb: int, out: Path) -> None:
    codes = list(per_code.keys())
    ncol = min(len(codes), 4)
    nrow = int(np.ceil(len(codes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for k, cid in enumerate(codes):
        ax = axes[k // ncol][k % ncol]
        rec = per_code[cid][space]
        g = rec["grid_frac"]
        amps = rec["amplitudes"]
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(amps)))
        for ai, amp in enumerate(amps):
            ax.plot(g, rec["d_norm"][ai], color=cmap[ai], lw=2.0,
                    label=f"amp={amp:g}")
        if rec.get("wrong_d_norm") is not None:
            ax.plot(g, rec["wrong_d_norm"], color="k", ls="--", lw=1.6,
                    label="wrong-code (max amp)")
        ax.axvline(t_perturb / rec["T"], color="red", lw=1.0, ls=":")
        ax.axhline(1.0, color="grey", lw=0.7, ls=":")
        beh = per_code[cid]["behavior"]
        ax.set_title(f"code {cid} [{beh}]",
                     color=BEHAVIOR_COLORS.get(beh, "k"), fontsize=10)
        ax.set_xlabel("rollout fraction")
        ax.set_ylabel(r"$d(t)\,/\,d_{\mathrm{floor}}$")
        ax.legend(fontsize=6, loc="upper right")
    for k in range(len(codes), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"Perturbation–recovery ({space} space): impulse at the red line; "
        "curves return toward 1 = the controller defends the code's attractor",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_basin(per_code: dict, space: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for cid, rec in per_code.items():
        r = rec[space]
        amps = r["amplitudes"]
        fracs = [r["metrics"][ai]["recovery_frac"] for ai in range(len(amps))]
        beh = rec["behavior"]
        ax.plot(amps, fracs, marker="o", color=BEHAVIOR_COLORS.get(beh, "k"),
                label=f"code {cid} [{beh}]")
    ax.set_xlabel("impulse amplitude (qvel std, rad/s | m/s)")
    ax.set_ylabel("recovery fraction  (1 = full return to attractor)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Basin: recovery vs. shove magnitude ({space})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs",
            config_name="perturbation_recovery")
def main(cfg: DictConfig) -> None:
    log.info("=== Perturbation–recovery experiment ===")
    out_dir = _abs(cfg.output.base_dir) / str(cfg.output.run_name)
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    rng = np.random.RandomState(int(cfg.seed))
    rb = _abs(cfg.rollouts_base)

    # --- codes ---
    if cfg.codes is None:
        codes, code_beh = _auto_pick_codes(rb, str(cfg.labels_run))
    else:
        codes = [int(c) for c in cfg.codes]
        # behavior lookup from labels run if available
        try:
            labels = json.loads(
                (rb / str(cfg.labels_run) / "labels" / "code_labels.json").read_text())
            code_beh = {c: _base_behavior(str(labels["codes"].get(str(c), {})
                                          .get("label", ""))) for c in codes}
        except Exception:
            code_beh = {c: "Other" for c in codes}
    log.info(f"Codes to perturb: {codes}  ({code_beh})")

    # --- checkpoint + decoder-only inference ---
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path))
    if not bool(ckpt_cfg.network_config.get("use_rnn_decoder", False)):
        raise ValueError("Perturbation experiment requires an RNN decoder.")
    codes_override = cfg.checkpoint.get("codes_path_override", None)
    if codes_override is not None:
        ckpt_cfg.kpms_config.codes_path = _abs(codes_override)
    z_e_scale = float(cfg.z_e_scale)
    assert_decoder_only_settings(z_e_scale, deterministic=True)
    log.info(describe_decoder_only_mode(ckpt_cfg.network_config))
    inf_fn = make_inference_fn(ppo_networks, use_rnn=True,
                               deterministic=True, z_e_scale=z_e_scale)
    params = (norm_state, policy_params)
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

    # --- diverse-IC start clips + env ---
    splits = load_balanced_splits(_abs(cfg.data.balanced_split_path))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    clip_len = int(ckpt_cfg.env_config.clip_length)
    codes_payload = load_codes(ckpt_cfg, train_indices, test_indices, clip_len)
    num_codes = int(codes_payload.num_codes)
    _, usage = rank_codes_by_usage(
        codes_payload.train_codes, codes_payload.test_codes, num_codes)

    K = int(cfg.rollout.K)
    hold_frames = int(cfg.rollout.hold_frames)
    t_perturb = int(cfg.rollout.t_perturb)
    perturb_dur = int(cfg.rollout.perturb_dur)
    probe_frame = int(cfg.rollout.probe_frame)
    clips_p, probe_p = _pick_random_clips(
        test_indices, K=K, seed=int(cfg.seed) + 4000, probe_frame=probe_frame)
    test_pos = {int(c): i for i, c in enumerate(test_indices)}
    codes_for_clips = np.stack(
        [codes_payload.test_codes[test_pos[int(c)]] for c in clips_p], axis=0)
    log.info(f"Building env over {K} diverse-IC clips: {clips_p.tolist()}")
    env = _build_pose_env(
        ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
        keep_clips_idx=clips_p, test_codes=codes_for_clips,
        code_stack_size=code_stack_size, probe_frame=probe_p,
    )
    clip_indices = jnp.arange(K, dtype=jnp.int32)
    base_keys = jnp.stack([jax.random.PRNGKey(int(cfg.seed) + 17 * (k + 1))
                           for k in range(K)])

    amplitudes = [float(a) for a in cfg.amplitudes]

    @partial(jax.jit, static_argnums=())
    def _run(code, amp, keys):
        return _perturbed_rollout(
            env, inf_fn, params, ppo_networks, code, amp,
            clip_indices, keys, hold_frames, t_perturb, perturb_dur,
            code_stack_size,
        )

    # --- rollouts: (code, amp) ---
    reuse = bool(cfg.get("reuse_rollouts", False))
    raw: dict[int, dict[float, dict[str, np.ndarray]]] = {}
    for cid in codes:
        raw[cid] = {}
        for amp in amplitudes:
            npz = data_dir / f"rollout_code{cid:03d}_amp{amp:g}.npz"
            if reuse and npz.exists():
                d = np.load(npz)
                raw[cid][amp] = {"qpos": d["qpos"], "hidden": d["hidden"],
                                 "reward": d["reward"]}
                continue
            outs = _run(jnp.int32(cid), jnp.float32(amp), base_keys)
            agg = {"qpos": np.asarray(outs["qpos"]),
                   "hidden": np.asarray(outs["hidden"]),
                   "reward": np.asarray(outs["reward"])}
            np.savez_compressed(npz, **agg)
            raw[cid][amp] = agg
            log.info(f"  code {cid:>3d} amp {amp:<4g}: "
                     f"qpos {agg['qpos'].shape} mean_rew "
                     f"{float(agg['reward'].mean()):.3f}")

    # --- sanitize physically-diverged bodies ---
    # At the strongest shoves a small fraction of bodies can blow the simulator
    # up and produce non-finite qpos for the rest of the episode. Forward-fill
    # the last finite frame per body (a diverged body "freezes" at its last
    # valid pose, so it correctly stays FAR from the attractor and counts as
    # *not* recovered rather than poisoning the PCA/kNN with NaN).
    def _ffill_nan(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).copy()
        bad = ~np.isfinite(x)
        if not bad.any():
            return x
        # x: (K, T, D). Forward-fill along T per (body, dim).
        K, Tt, D = x.shape
        for k in range(K):
            for dd in range(D):
                col = x[k, :, dd]
                m = ~np.isfinite(col)
                if not m.any():
                    continue
                last = 0.0
                for t in range(Tt):
                    if np.isfinite(col[t]):
                        last = col[t]
                    else:
                        col[t] = last
                x[k, :, dd] = col
        return x

    for cid in raw:
        for amp in raw[cid]:
            for key in ("qpos", "hidden"):
                if not np.isfinite(raw[cid][amp][key]).all():
                    n_bad = int((~np.isfinite(raw[cid][amp][key])).sum())
                    raw[cid][amp][key] = _ffill_nan(raw[cid][amp][key])
                    log.warning(
                        f"  sanitized {n_bad} non-finite {key} values "
                        f"(code {cid}, amp {amp:g}) by forward-fill")

    # --- analysis ---
    a = cfg.analysis
    spaces = list(a.spaces)
    T = hold_frames
    grid = np.arange(T)
    cloud_idx = np.arange(t_perturb + int(a.settle_margin), T)

    per_code: dict[int, dict] = {}
    for cid in codes:
        per_code[cid] = {"behavior": code_beh.get(cid, "Other")}
        for space in spaces:
            key = "qpos" if space == "pose" else "hidden"
            # z-score stats from amp=0 (the unperturbed attractor)
            f0 = _features(raw[cid][amplitudes[0]][key], space)
            mean = f0.reshape(-1, f0.shape[-1]).mean(0)
            std = f0.reshape(-1, f0.shape[-1]).std(0)
            std = np.where(std < 1e-6, 1.0, std)
            f0z = (f0 - mean) / std
            pca, nn = _build_cloud(f0z, cloud_idx, int(a.pca_dim),
                                   int(a.knn), int(a.manifold_subsample), rng)
            d_curves, metrics = [], []
            floor_ref = None
            for amp in amplitudes:
                fz = (_features(raw[cid][amp][key], space) - mean) / std
                d = _dist_curve(fz, pca, nn, grid)
                if floor_ref is None:
                    floor_ref = float(np.median(d[grid < t_perturb]))
                    floor_ref = max(floor_ref, 1e-9)
                d_curves.append(d)
            d_arr = np.stack(d_curves, axis=0)
            d_norm = d_arr / floor_ref
            for ai, amp in enumerate(amplitudes):
                metrics.append(_recovery_metrics(d_norm[ai], grid, t_perturb, a))
            # wrong-code control: largest amp bodies vs a different code's cloud
            wrong_d_norm = None
            others = [c for c in codes if c != cid]
            if others:
                wc = others[0]
                fw = _features(raw[wc][amplitudes[0]][key], space)
                mw = fw.reshape(-1, fw.shape[-1]).mean(0)
                sw = fw.reshape(-1, fw.shape[-1]).std(0)
                sw = np.where(sw < 1e-6, 1.0, sw)
                fwz = (fw - mw) / sw
                pcaw, nnw = _build_cloud(fwz, cloud_idx, int(a.pca_dim),
                                         int(a.knn), int(a.manifold_subsample), rng)
                fz_max = (_features(raw[cid][amplitudes[-1]][key], space) - mw) / sw
                dw = _dist_curve(fz_max, pcaw, nnw, grid)
                # normalize wrong-code curve by the *correct* code's own
                # body-to-wrong-cloud floor at convergence for comparability
                wrong_floor = max(float(np.median(dw[grid < t_perturb])), 1e-9)
                wrong_d_norm = dw / wrong_floor
            per_code[cid][space] = {
                "amplitudes": amplitudes,
                "d_norm": d_norm,
                "wrong_d_norm": wrong_d_norm,
                "metrics": metrics,
                "grid_frac": grid / (T - 1),
                "T": T,
            }
            mfracs = [m["recovery_frac"] for m in metrics]
            log.info(f"  [code {cid} {space}] recovery_frac by amp: "
                     + ", ".join(f"{amp:g}:{fr:.2f}"
                                 for amp, fr in zip(amplitudes, mfracs)))

    # --- figures ---
    for space in spaces:
        _plot_curves(per_code, space, t_perturb,
                     fig_dir / f"recovery_curves_{space}.png")
        _plot_basin(per_code, space, fig_dir / f"basin_{space}.png")

    # --- summary json ---
    summary = {
        "checkpoint": str(cfg.checkpoint.path),
        "K": K, "hold_frames": hold_frames, "t_perturb": t_perturb,
        "perturb_dur": perturb_dur, "amplitudes": amplitudes,
        "codes": {},
    }
    for cid in codes:
        summary["codes"][str(cid)] = {
            "behavior": per_code[cid]["behavior"],
            "spaces": {
                space: {
                    "amplitudes": amplitudes,
                    "metrics": [
                        {k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in m.items()}
                        for m in per_code[cid][space]["metrics"]
                    ],
                }
                for space in spaces
            },
        }
    (out_dir / "perturbation_summary.json").write_text(json.dumps(summary, indent=2))

    log.info("\n=== Recovery summary (pose space; frac=1 → full return) ===")
    for cid in codes:
        m = per_code[cid]["pose"]["metrics"]
        log.info(
            f"  code {cid:>3d} [{per_code[cid]['behavior']:>8s}] "
            + " ".join(f"a{amp:g}={mm['recovery_frac']:.2f}"
                       for amp, mm in zip(amplitudes, m)))
    log.info(f"\n=== Done. Artifacts under {out_dir} ===")


if __name__ == "__main__":
    main()
