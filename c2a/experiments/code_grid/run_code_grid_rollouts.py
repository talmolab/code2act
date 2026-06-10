"""Experiment: per-code rollouts from controlled poses, rendered as 6x6 grids.

For each tokenizer code (top-N by usage, zero-usage codes dropped) and each
starting pose (walk / rear), roll out K bodies that hold that single code
for ``hold_frames`` steps in decoder-only mode. Output:

* Raw npz rollouts per (pose, code).
* Raw 6x6 grid videos per pose — each panel shows the K walkers under one
  code. Panels are colored by the code's behavior label (all ``Rear_S``
  codes share one color, etc.) so color identity is stable across videos.
* Combined 6x6 grid videos overlaying the K walk-pose bodies (opaque) and
  the K rear-pose bodies (translucent) in each panel.
* Per-pose MMD merge of codes by behavioral similarity (UPGMA + elbow cut).
* Per-pose "merged" grid videos: one panel per cluster, walkers colored
  by their original code id.

Companion script: ``run_code_grid_dynamics.py`` runs DSA + rSLDS on the
merged-cluster hidden state trajectories. See its module docstring.

Usage:
    cd code2act
    python -m c2a.experiments.code_grid.run_code_grid_rollouts
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

# Workaround for venv's orbax-checkpoint (0.11.24) calling jax.monitoring.record_scalar,
# which was removed from JAX. Same patch as c2a/train_moseq_decoder.py — needed before
# any Orbax restore happens. Pure telemetry call; stubbing is safe.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

from c2a.paths import get_c2a_root, get_repo_root

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.codegen.code_sources import load_codes
from c2a.codegen.shared.label_alignment import (
    BEHAVIORS, BEHAVIOR_TO_INT, UNLABELED, load_label_frames,
)
from c2a.experiments.dynamics.run_hidden_dynamics import run_rollout_with_hidden
from c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled import (
    _pick_start_clips, _build_pose_env,
)
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint, make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.code_grid import (
    make_code_colors, make_label_colors, rank_codes_by_usage,
    compute_mmd_distance_matrix, cluster_codes,
    render_panel_frames, add_banner, stack_panels_into_grid, write_mp4,
    save_clusters_json, build_uniform_walkers,
)
from c2a.experiments.shared.code_labels_from_rollouts import (
    cluster_majority_label_rollout as cluster_majority_label,
    label_based_clusters_rollout as label_based_clusters,
    label_codes_from_rollouts_dir,
    load_code_labels_json,
    save_code_labels_json,
)

log = logging.getLogger(__name__)


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Rollout phase
# ---------------------------------------------------------------------------


def _batched_rollout_for_code(
    env, inf_fn, params, ppo_networks,
    code, clip_indices, base_keys,
    hold_frames: int, code_stack_size: int,
) -> dict[str, jax.Array]:
    """Roll out K bodies holding ``code``, vmapped over bodies + scanned over t.

    Replaces the ``for k: for t:`` Python loops with one ``vmap`` over the K
    bodies and one ``lax.scan`` over the ``hold_frames`` timesteps, so the
    whole batch compiles into a single XLA program instead of K*T dispatches.

    Reproduces ``run_rollout_with_hidden`` exactly for the held-code case:
    same PRNG threading (``key, reset_key = split(base_key)`` then per-step
    ``key, subkey = split(key)``), same per-step ops (override ``kpms_code``,
    inference, env step), same recorded quantities. Valid because for these
    checkpoints ``reinit_hidden_on_code=False`` and ``run_rollout_with_hidden``
    never resets hidden on ``done`` — so the GRU carry is just threaded.

    Args:
        code: scalar (traced) code id held fixed for every body/timestep.
        clip_indices: ``(K,)`` env-internal reset clip index per body.
        base_keys: ``(K, 2)`` PRNG key per body.

    Returns:
        ``{hidden, qpos, proprio, code, reward}`` each ``(K, hold_frames, ...)``.
    """
    held_code = jnp.full(
        (code_stack_size,), jnp.asarray(code, dtype=jnp.float32),
    )

    def single_body(base_key, clip_idx):
        key, reset_key = jax.random.split(base_key)
        state0 = env.reset(reset_key, clip_idx=clip_idx)
        hidden0 = ppo_networks.policy_network.init_hidden(1)

        def step(carry, _):
            state, hidden, key = carry
            # Hold the code fixed by overwriting the obs's kpms_code window.
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = held_code
            state = state.replace(obs=new_obs)
            key, subkey = jax.random.split(key)
            batched_obs = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inf_fn(params, batched_obs, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)
            out = {
                "hidden": hidden[-1][0],
                "qpos": state.data.qpos,
                "proprio": state.obs["proprioception"],
                "code": jnp.asarray(extras["code_idx"]).reshape(-1)[0].astype(
                    jnp.int32
                ),
            }
            new_state = env.step(state, action)
            out["reward"] = new_state.reward
            return (new_state, hidden, key), out

        _, outs = jax.lax.scan(
            step, (state0, hidden0, key), None, length=hold_frames,
        )
        return outs

    return jax.vmap(single_body)(base_keys, clip_indices)


def _rollouts_for_pose(
    env, inf_fn, params, ppo_networks,
    pose_clips: np.ndarray, code_ids: list[int],
    hold_frames: int, seed: int,
    jit_reset, jit_step,
    pose_name: str, data_dir: Path,
    reuse: bool, code_stack_size: int,
) -> dict[int, dict[str, np.ndarray]]:
    """For each code id, K rollouts holding that code for ``hold_frames``.

    Uses the vmap+scan batched rollout (``_batched_rollout_for_code``): the
    K bodies and the timestep loop are inside one compiled XLA program, so
    only the (short) loop over code ids stays in Python.

    Returns ``{code_id: agg_dict}`` with keys
    ``hidden, qpos, proprio, codes_actual, rewards, start_clips, code, pose``.
    """
    K = int(pose_clips.shape[0])
    clip_indices = jnp.arange(K, dtype=jnp.int32)
    results: dict[int, dict[str, np.ndarray]] = {}

    # Compiled once on the first code, reused for every subsequent code.
    @jax.jit
    def _run(code, base_keys):
        return _batched_rollout_for_code(
            env, inf_fn, params, ppo_networks, code, clip_indices, base_keys,
            hold_frames, code_stack_size,
        )

    for ci, code in enumerate(code_ids):
        npz_path = data_dir / f"rollout_{pose_name}_code{int(code):03d}.npz"
        if reuse and npz_path.exists():
            d = np.load(npz_path)
            results[int(code)] = {k: d[k] for k in d.files}
            continue

        # Per-body PRNG keys — identical derivation to the prior Python loop
        # (PRNGKey(seed + 1_000_003*(ci+1) + k)), so results match.
        base_keys = jnp.stack([
            jax.random.PRNGKey(seed + 1_000_003 * (ci + 1) + k)
            for k in range(K)
        ])
        outs = _run(jnp.int32(int(code)), base_keys)

        agg = {
            "hidden": np.asarray(outs["hidden"]),
            "qpos": np.asarray(outs["qpos"]),
            "proprio": np.asarray(outs["proprio"]),
            "codes_actual": np.asarray(outs["code"]),
            "rewards": np.asarray(outs["reward"]),
            "start_clips": np.asarray(pose_clips, dtype=np.int64),
            "code": np.int32(int(code)),
            "pose": pose_name,
        }
        np.savez_compressed(npz_path, **agg)
        log.info(
            f"  [{pose_name}] code {int(code):>3d}: saved {npz_path.name} "
            f"(K={K}, hold={hold_frames}, mean_reward="
            f"{float(np.mean(agg['rewards'])):.3f})"
        )
        results[int(code)] = agg
    return results


# ---------------------------------------------------------------------------
# Raw grid render: one ghost_model per pose, K=K_per_panel ghosts.
# ---------------------------------------------------------------------------


def _solid_ghost_colors(rgba: np.ndarray, n: int) -> list[list[float]]:
    return [rgba.tolist() for _ in range(n)]


def _render_raw_grid_for_pose(
    env, pose_name: str, top_codes_in_pose: list[int],
    rollouts: dict[int, dict[str, np.ndarray]],
    code_colors: np.ndarray, usage: np.ndarray,
    cfg: DictConfig, video_dir: Path,
    code_labels: dict[int, dict[str, Any]] | None = None,
) -> None:
    """One grid video per (pose, chunk-of-codes). Each panel = one code."""
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    panels_per_grid = rows * cols
    K = int(cfg.K_per_panel)
    panel_w, panel_h = int(cfg.panel_w), int(cfg.panel_h)
    panel_banner_h = int(cfg.panel_banner_h)
    grid_banner_h = int(cfg.grid_banner_h)
    fps = int(cfg.render_fps)

    chunks = [
        top_codes_in_pose[i:i + panels_per_grid]
        for i in range(0, len(top_codes_in_pose), panels_per_grid)
    ]
    for gi, chunk in enumerate(chunks):
        log.info(
            f"  [{pose_name}] raw grid {gi}: codes {chunk[0]}..{chunk[-1]} "
            f"({len(chunk)} panels)"
        )
        panel_frames_list: list[np.ndarray] = []
        for rank_idx, code in enumerate(chunk):
            agg = rollouts[int(code)]
            qpos = agg["qpos"]  # (K, T, nq)
            traj = [qpos[k] for k in range(min(K, qpos.shape[0]))]
            # Per-panel ghost model: K walkers all in the same code color
            # (base body recolored to match ghosts via build_uniform_walkers).
            color = code_colors[int(code)]
            walker_colors = [color.tolist()] * len(traj)
            ghost_model, base_nq = build_uniform_walkers(
                env, walker_colors=walker_colors,
                camera_distance=float(cfg.camera_distance),
                camera_elevation=float(cfg.camera_elevation),
                camera_azimuth=float(cfg.camera_azimuth),
            )
            raw_frames = render_panel_frames(
                ghost_model, base_nq, traj,
                camera_name="divergent_cam",
                width=panel_w, height=panel_h,
            )
            color_dot = "● "
            lab = (
                str(code_labels[int(code)]["label"])
                if code_labels is not None and int(code) in code_labels
                else None
            )
            lab_str = f" [{lab}]" if lab and lab != "unlabeled" else ""
            title = (
                f"{color_dot}code {int(code):>3d}{lab_str} "
                f"rank={gi * panels_per_grid + rank_idx + 1} "
                f"use={int(usage[int(code)])}"
            )
            # Per-panel banner stacked above the rendered frames.
            framed = np.stack([
                add_banner(
                    f, title, banner_h=panel_banner_h,
                    bg=(245, 245, 245),
                    fg=tuple(int(255 * c * 0.65) for c in color[:3]),
                    font_size=12,
                )
                for f in raw_frames
            ], axis=0)
            panel_frames_list.append(framed)

        grid = stack_panels_into_grid(
            panel_frames_list, rows=rows, cols=cols,
            grid_title=f"{pose_name.upper()} pose - raw codes "
                       f"{chunk[0]}..{chunk[-1]} (z_e=0)",
            grid_banner_h=grid_banner_h,
        )
        out_path = video_dir / f"{pose_name}_raw_codes_{chunk[0]:03d}-{chunk[-1]:03d}.mp4"
        write_mp4(grid, out_path, fps=fps)
        log.info(f"    Wrote {out_path}")


# ---------------------------------------------------------------------------
# Combined grid render: walk + rear bodies overlaid in each panel.
# ---------------------------------------------------------------------------


def _render_combined_grid(
    env, top_codes: list[int],
    rollouts_walk: dict[int, dict[str, np.ndarray]],
    rollouts_rear: dict[int, dict[str, np.ndarray]],
    code_colors: np.ndarray, usage: np.ndarray,
    cfg: DictConfig, video_dir: Path,
    code_labels: dict[int, dict[str, Any]] | None = None,
) -> None:
    """One grid video per code-chunk overlaying walk + rear bodies.

    Each panel = one code. The K walk-pose bodies render fully opaque; the
    K rear-pose bodies render translucent (``combined_rear_alpha``) so the
    two start poses stay visually separable within a single panel. Both
    sets share the code's behavior-label color.
    """
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    panels_per_grid = rows * cols
    K = int(cfg.K_per_panel)
    panel_w, panel_h = int(cfg.panel_w), int(cfg.panel_h)
    panel_banner_h = int(cfg.panel_banner_h)
    grid_banner_h = int(cfg.grid_banner_h)
    fps = int(cfg.render_fps)
    rear_alpha = float(cfg.get("combined_rear_alpha", 0.4))

    chunks = [
        top_codes[i:i + panels_per_grid]
        for i in range(0, len(top_codes), panels_per_grid)
    ]
    for gi, chunk in enumerate(chunks):
        log.info(
            f"  [combined] grid {gi}: codes {chunk[0]}..{chunk[-1]} "
            f"({len(chunk)} panels)"
        )
        panel_frames_list: list[np.ndarray] = []
        for rank_idx, code in enumerate(chunk):
            qpos_w = rollouts_walk[int(code)]["qpos"]  # (K, T, nq)
            qpos_r = rollouts_rear[int(code)]["qpos"]  # (K, T, nq)
            n_walk = min(K, qpos_w.shape[0])
            n_rear = min(K, qpos_r.shape[0])
            # Walk bodies opaque, rear bodies translucent — same hue.
            rgb = code_colors[int(code)][:3].tolist()
            walk_rgba = [*rgb, 1.0]
            rear_rgba = [*rgb, rear_alpha]
            walker_colors = (
                [walk_rgba] * n_walk + [rear_rgba] * n_rear
            )
            ghost_model, base_nq = build_uniform_walkers(
                env, walker_colors=walker_colors,
                camera_distance=float(cfg.camera_distance),
                camera_elevation=float(cfg.camera_elevation),
                camera_azimuth=float(cfg.camera_azimuth),
            )
            traj = (
                [qpos_w[k] for k in range(n_walk)]
                + [qpos_r[k] for k in range(n_rear)]
            )
            raw_frames = render_panel_frames(
                ghost_model, base_nq, traj,
                camera_name="divergent_cam",
                width=panel_w, height=panel_h,
            )
            lab = (
                str(code_labels[int(code)]["label"])
                if code_labels is not None and int(code) in code_labels
                else None
            )
            lab_str = f" [{lab}]" if lab and lab != "unlabeled" else ""
            title = (
                f"● code {int(code):>3d}{lab_str} "
                f"rank={gi * panels_per_grid + rank_idx + 1} "
                f"use={int(usage[int(code)])}"
            )
            framed = np.stack([
                add_banner(
                    f, title, banner_h=panel_banner_h,
                    bg=(245, 245, 245),
                    fg=tuple(int(255 * c * 0.65) for c in rgb),
                    font_size=12,
                )
                for f in raw_frames
            ], axis=0)
            panel_frames_list.append(framed)

        grid = stack_panels_into_grid(
            panel_frames_list, rows=rows, cols=cols,
            grid_title=f"WALK (opaque) + REAR (faint) - codes "
                       f"{chunk[0]}..{chunk[-1]} (z_e=0)",
            grid_banner_h=grid_banner_h,
        )
        out_path = (
            video_dir
            / f"combined_codes_{chunk[0]:03d}-{chunk[-1]:03d}.mp4"
        )
        write_mp4(grid, out_path, fps=fps)
        log.info(f"    Wrote {out_path}")


# ---------------------------------------------------------------------------
# Merged grid render: one ghost_model per (pose, cluster), colored by code.
# ---------------------------------------------------------------------------


def _render_merged_grid_for_pose(
    env, pose_name: str,
    clusters: list[list[int]],
    rollouts: dict[int, dict[str, np.ndarray]],
    code_colors: np.ndarray, usage: np.ndarray,
    cfg: DictConfig, video_dir: Path,
    code_labels: dict[int, dict[str, Any]] | None = None,
    filename_prefix: str = "merged_clusters",
    title_kind: str = "merged clusters",
) -> None:
    """One merged-grid video per chunk of clusters. Each panel = one cluster.

    Within a panel: K_per_panel walkers per sub-code in the cluster, each
    walker colored by its sub-code (so a 3-code cluster shows 3 colors).
    """
    rows = int(cfg.grid_rows)
    cols = int(cfg.grid_cols)
    panels_per_grid = rows * cols
    K = int(cfg.K_per_panel)
    panel_w, panel_h = int(cfg.panel_w), int(cfg.panel_h)
    panel_banner_h = int(cfg.panel_banner_h)
    grid_banner_h = int(cfg.grid_banner_h)
    fps = int(cfg.render_fps)

    chunks = [
        clusters[i:i + panels_per_grid]
        for i in range(0, len(clusters), panels_per_grid)
    ]

    for gi, chunk in enumerate(chunks):
        panel_frames_list: list[np.ndarray] = []
        for ci, cluster in enumerate(chunk):
            # Per-cluster ghost model: K * |cluster| walkers; each walker
            # is colored by its originating sub-code. The base body is
            # recolored to match the first sub-code so all walkers in the
            # cluster panel are uniformly visible.
            walker_colors: list[list[float]] = []
            for sub_code in cluster:
                walker_colors += [code_colors[int(sub_code)].tolist()] * K
            ghost_model, base_nq = build_uniform_walkers(
                env, walker_colors=walker_colors,
                camera_distance=float(cfg.camera_distance),
                camera_elevation=float(cfg.camera_elevation),
                camera_azimuth=float(cfg.camera_azimuth),
            )
            traj: list[np.ndarray] = []
            for sub_code in cluster:
                qpos = rollouts[int(sub_code)]["qpos"]
                for k in range(K):
                    traj.append(qpos[k % qpos.shape[0]])
            raw_frames = render_panel_frames(
                ghost_model, base_nq, traj,
                camera_name="divergent_cam",
                width=panel_w, height=panel_h,
            )

            code_str = ",".join(f"{c}" for c in cluster[:6])
            if len(cluster) > 6:
                code_str += "+"
            cluster_lab = (
                cluster_majority_label(cluster, code_labels)
                if code_labels is not None else "unlabeled"
            )
            lab_str = f" [{cluster_lab}]" if cluster_lab and cluster_lab != "unlabeled" else ""
            title = (
                f"cluster {gi * panels_per_grid + ci}{lab_str} [{code_str}]"
            )
            framed = np.stack([
                add_banner(
                    f, title, banner_h=panel_banner_h,
                    bg=(245, 245, 245),
                    fg=(35, 35, 35),
                    font_size=12,
                )
                for f in raw_frames
            ], axis=0)
            panel_frames_list.append(framed)

        if not panel_frames_list:
            continue
        # Force all panels to the same T (clip to min).
        T_min = min(p.shape[0] for p in panel_frames_list)
        panel_frames_list = [p[:T_min] for p in panel_frames_list]

        grid = stack_panels_into_grid(
            panel_frames_list, rows=rows, cols=cols,
            grid_title=f"{pose_name.upper()} pose - {title_kind} "
                       f"{gi * panels_per_grid}..{gi * panels_per_grid + len(chunk) - 1}",
            grid_banner_h=grid_banner_h,
        )
        out_path = (
            video_dir
            / f"{pose_name}_{filename_prefix}_g{gi:02d}.mp4"
        )
        write_mp4(grid, out_path, fps=fps)
        log.info(f"  [{pose_name}] {filename_prefix} grid {gi}: wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pick_random_clips(
    test_indices: np.ndarray, K: int, seed: int, probe_frame: int,
) -> tuple[np.ndarray, int]:
    """Pick K random start clips with NO posture constraint.

    Used by the ``random`` pose: instead of finding K clips that share a
    Walk/Rear pose at a probe frame (which caps out at the few clips labeled
    that behavior), we draw K random clips from the test split and reset each
    at ``probe_frame``. The same seed → the same K initializations, which are
    then reused for *every* code, so differences across codes are purely the
    held code, not the init. With-replacement only if the pool < K.
    """
    rng = np.random.RandomState(seed)
    pool = np.asarray(test_indices, dtype=np.int64)
    replace = len(pool) < K
    chosen = np.sort(rng.choice(pool, size=K, replace=replace))
    return chosen.astype(np.int64), int(probe_frame)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_grid_rollouts_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Grid Rollouts experiment ===")

    out_base = Path(_abs(cfg.output.base_dir)) / str(cfg.output.run_name)
    data_dir = out_base / "data"
    videos_raw_dir = out_base / "videos_raw"
    merged_dir = out_base / "merged"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_raw_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # --- Checkpoint + inference fn ---
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path),
    )
    if not bool(ckpt_cfg.network_config.get("use_rnn_decoder", False)):
        raise ValueError(
            "This experiment requires an RNN decoder checkpoint."
        )
    # Override the checkpoint's saved codes_path if requested. This handles
    # the case where the tokenizer files have moved since the ckpt was
    # written.
    codes_override = cfg.checkpoint.get("codes_path_override", None)
    if codes_override is not None:
        ckpt_cfg.kpms_config.codes_path = _abs(codes_override)
        log.info(
            f"Overriding ckpt codes_path → {ckpt_cfg.kpms_config.codes_path}"
        )
    z_e_scale = float(cfg.get("z_e_scale", 0.0))
    from c2a.experiments.shared.decoder_only import (
        assert_decoder_only_settings, describe_decoder_only_mode,
    )
    assert_decoder_only_settings(z_e_scale, deterministic=True)
    log.info(describe_decoder_only_mode(ckpt_cfg.network_config))
    inf_fn = make_inference_fn(
        ppo_networks, use_rnn=True,
        deterministic=True, z_e_scale=z_e_scale,
    )
    params = (norm_state, policy_params)
    log.info(f"Inference fn: deterministic=True, z_e_scale={z_e_scale}")

    # --- Codes + balanced split ---
    splits = load_balanced_splits(_abs(cfg.data.balanced_split_path))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    clip_len = int(ckpt_cfg.env_config.clip_length)
    codes_payload = load_codes(ckpt_cfg, train_indices, test_indices, clip_len)
    num_codes = int(codes_payload.num_codes)
    log.info(
        f"Codes: source={codes_payload.code_source}, num_codes={num_codes}"
    )

    # --- Rank codes by usage (zero-usage codes dropped entirely) ---
    ranked_codes, usage = rank_codes_by_usage(
        codes_payload.train_codes, codes_payload.test_codes, num_codes,
    )
    used_ranked = [int(c) for c in ranked_codes.tolist() if usage[int(c)] > 0]
    n_dropped = num_codes - len(used_ranked)
    if n_dropped:
        log.info(
            f"Dropped {n_dropped} zero-usage code(s); "
            f"{len(used_ranked)} code(s) have usage > 0"
        )
    if not used_ranked:
        raise ValueError("No codes with nonzero usage — nothing to roll out.")
    top_n = min(len(used_ranked), int(cfg.top_n_codes))
    top_codes = used_ranked[:top_n]
    log.info(
        f"Top-{top_n} codes by usage (first 10 shown): {top_codes[:10]} "
        f"(usage range {int(usage[top_codes[0]])}..{int(usage[top_codes[-1]])})"
    )
    # Persist ranking for downstream interpretation.
    np.savez(
        data_dir / "usage_rank.npz",
        ranked_codes=ranked_codes, usage=usage, top_codes=np.asarray(top_codes),
    )

    # --- Pose-conditioned start clips ---
    label_arrays = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    behavior_arr = label_arrays.behavior

    # Per-code labels are computed AFTER rollouts (they require the
    # generated qpos), so this block lives further down. The labels CSV is
    # still loaded above because `_pick_start_clips` uses `behavior_arr` to
    # find clips matching the requested start pose.

    K = int(cfg.K_per_panel)
    probe_frame = int(cfg.start_pose.probe_frame)
    probe_frame_max = cfg.start_pose.get("probe_frame_max", None)
    if probe_frame_max is not None:
        probe_frame_max = int(probe_frame_max)
    seed = int(cfg.seed)

    # Which start poses to roll out. Subset of {walk, rear}; setting
    # cfg.poses=[walk] does a walk-only run (skips rear entirely).
    poses = [str(p) for p in cfg.get("poses", ["walk", "rear"])]
    for p in poses:
        if p not in ("walk", "rear", "random"):
            raise ValueError(
                f"Unknown pose {p!r}; expected 'walk', 'rear', or 'random'."
            )
    log.info(f"Poses: {poses}")

    pose_start_label = {
        "walk": cfg.start_pose.walking_label,
        "rear": cfg.start_pose.rearing_label,
    }
    clip_seed_off = {"walk": 0, "rear": 1, "random": 7}
    rollout_seed = {"walk": seed, "rear": seed + 2000, "random": seed + 4000}

    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

    def _codes_for_pose(pose_clips: np.ndarray) -> np.ndarray:
        test_pos_lookup = {int(c): i for i, c in enumerate(test_indices)}
        rows = []
        for clip_id in pose_clips:
            pos = test_pos_lookup[int(clip_id)]
            rows.append(codes_payload.test_codes[pos])
        return np.stack(rows, axis=0)

    hold_frames = int(cfg.hold_frames)
    reuse = bool(cfg.get("reuse_rollouts", False))
    log.info(
        f"\nRollouts: {top_n} codes × {len(poses)} pose(s) × {K} bodies × "
        f"{hold_frames} frames (reuse_rollouts={reuse})"
    )

    # Per-pose env + rollouts, built only for the requested poses.
    pose_envs = {}
    pose_rollouts = {}
    for p in poses:
        if p == "random":
            clips_p, probe_p = _pick_random_clips(
                test_indices, K=K, seed=seed + clip_seed_off[p],
                probe_frame=probe_frame,
            )
        else:
            clips_p, probe_p = _pick_start_clips(
                pose_start_label[p], test_indices, behavior_arr,
                probe_frame=probe_frame, probe_frame_max=probe_frame_max,
                K=K, seed=seed + clip_seed_off[p],
            )
        log.info(f"Pose clips — {p} @ frame {probe_p}: {clips_p.tolist()}")
        codes_p = _codes_for_pose(clips_p)
        log.info(f"Building {p}-pose env...")
        env_p = _build_pose_env(
            ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
            keep_clips_idx=clips_p, test_codes=codes_p,
            code_stack_size=code_stack_size, probe_frame=probe_p,
        )
        jit_reset_p = jax.jit(env_p.reset)
        jit_step_p = jax.jit(env_p.step)
        log.info(f"  {p} pose rollouts ...")
        pose_rollouts[p] = _rollouts_for_pose(
            env_p, inf_fn, params, ppo_networks,
            clips_p, top_codes, hold_frames, rollout_seed[p],
            jit_reset_p, jit_step_p, p, data_dir, reuse, code_stack_size,
        )
        pose_envs[p] = env_p

    # --- Per-code labels from ROLLED-OUT qpos (TopoMimic auto-labeler) ---
    # Each (code, body) rollout's longest behavior segment + majority
    # direction within that span votes for the code's modal label. Pools
    # bodies from both walk and rear poses. Replaces the prior
    # corpus-frame-majority labeling which collapsed onto a handful of
    # over-frequent classes (Immobile_S / Rear_S).
    labels_dir = out_base / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    code_labels_path = labels_dir / "code_labels.json"
    reuse_labels = bool(cfg.get("reuse_labels", True))
    if reuse_labels and code_labels_path.exists():
        code_labels = load_code_labels_json(code_labels_path)
        log.info(f"Loaded cached code labels from {code_labels_path}")
    else:
        log.info(
            "Computing per-code rollout-derived labels via TopoMimic auto-labeler ..."
        )
        code_labels = label_codes_from_rollouts_dir(
            data_dir, top_codes, num_codes=num_codes,
            poses=tuple(poses),
            src_fps=100.0,
        )
        save_code_labels_json(
            code_labels_path, code_labels,
            source=f"rollouts:{data_dir}",
        )
        log.info(f"  Saved {code_labels_path}")
    top_label_preview = [
        (c, code_labels.get(int(c), {}).get("label", "unlabeled"))
        for c in top_codes[:10]
    ]
    log.info(f"  Top-10 code labels: {top_label_preview}")

    # --- Colors ---
    # Raw + combined grids: color by behavior label (all Rear_S codes share
    # one color, etc.) so a behavior's color is stable across every video.
    # Merged grids: color by code id (per-code distinct) so sub-codes within
    # a cluster panel stay individually distinguishable.
    label_colors = make_label_colors(code_labels, num_codes)
    code_colors = make_code_colors(num_codes)

    # --- Raw grid rendering (per pose) ---
    if bool(cfg.get("render_video", True)):
        log.info("\n=== Rendering raw grids ===")
        for pose_name in poses:
            _render_raw_grid_for_pose(
                pose_envs[pose_name], pose_name, top_codes,
                pose_rollouts[pose_name],
                label_colors, usage, cfg, videos_raw_dir,
                code_labels=code_labels,
            )
        if "walk" in poses and "rear" in poses:
            log.info("\n=== Rendering combined walk+rear grids ===")
            _render_combined_grid(
                pose_envs["walk"], top_codes,
                pose_rollouts["walk"], pose_rollouts["rear"],
                label_colors, usage, cfg, videos_raw_dir,
                code_labels=code_labels,
            )

    # --- MMD merge per pose ---
    log.info("\n=== Computing MMD merge ===")
    reuse_merge = bool(cfg.get("reuse_merge", False))
    merge_cfg = cfg.merge
    pose_clusters: dict[str, list[list[int]]] = {}

    for pose_name, rollouts_pose in pose_rollouts.items():
        dist_path = merged_dir / f"distance_{pose_name}.npz"
        if reuse_merge and dist_path.exists():
            d = np.load(dist_path, allow_pickle=True)
            D = d["distance"]
            code_ids = list(d["code_ids"].tolist())
            sigma = float(d["sigma"])
            log.info(
                f"  [{pose_name}] reused cached MMD: D={D.shape}, sigma={sigma:.4g}"
            )
        else:
            qpos_per_code = {
                int(c): rollouts_pose[int(c)]["qpos"] for c in top_codes
            }
            D, code_ids, sigma = compute_mmd_distance_matrix(
                qpos_per_code,
                strip_root_pos=bool(merge_cfg.get("strip_root_pos", True)),
                kernel_sigma=merge_cfg.get("kernel_sigma", "median"),
            )
            np.savez(
                dist_path, distance=D, code_ids=np.asarray(code_ids),
                sigma=np.float32(sigma),
            )
            log.info(
                f"  [{pose_name}] MMD: D={D.shape}, sigma={sigma:.4g}, "
                f"median(D)={float(np.median(D)):.4f}"
            )

        clusters, info = cluster_codes(
            D, code_ids,
            cutoff=str(merge_cfg.get("cutoff", "elbow")),
            fixed_k=int(merge_cfg.get("fixed_k", 10)),
            frac_median=float(merge_cfg.get("frac_median", 0.5)),
        )
        save_clusters_json(
            merged_dir / f"clusters_{pose_name}.json",
            pose_name, clusters, info, usage,
        )
        # Augment clusters JSON with per-cluster majority label so the
        # downstream dynamics module can read it without recomputing.
        _augment_clusters_with_labels(
            merged_dir / f"clusters_{pose_name}.json", clusters, code_labels,
        )
        log.info(
            f"  [{pose_name}] {len(clusters)} clusters: sizes "
            f"{[len(c) for c in clusters[:10]]}..."
        )
        pose_clusters[pose_name] = clusters

    # --- Label-based clustering (pose-agnostic: codes with the same
    # majority behavior+direction tuple form one cluster regardless of
    # which pose the rollout starts from). We still write per-pose JSONs
    # so the downstream dynamics module can consume them with the same
    # naming convention as the MMD clusters; the partitions are identical. ---
    log.info("\n=== Label-based clustering ===")
    lbl_clusters, lbl_info = label_based_clusters(top_codes, code_labels)
    log.info(
        f"  label-clusters: "
        f"{[(lbl_info['labels'][i], len(c)) for i, c in enumerate(lbl_clusters)]}"
    )
    pose_label_clusters: dict[str, list[list[int]]] = {}
    for pose_name in poses:
        save_clusters_json(
            merged_dir / f"clusters_by_label_{pose_name}.json",
            pose_name, lbl_clusters, lbl_info, usage,
        )
        _augment_clusters_with_labels(
            merged_dir / f"clusters_by_label_{pose_name}.json",
            lbl_clusters, code_labels,
        )
        pose_label_clusters[pose_name] = lbl_clusters

    # --- Merged grid rendering ---
    if bool(cfg.get("render_video", True)):
        log.info("\n=== Rendering MMD-merged grids ===")
        for pose_name in poses:
            _render_merged_grid_for_pose(
                pose_envs[pose_name], pose_name,
                pose_clusters[pose_name], pose_rollouts[pose_name],
                code_colors, usage, cfg, merged_dir,
                code_labels=code_labels,
                filename_prefix="merged_clusters",
                title_kind="MMD merged clusters",
            )
        log.info("\n=== Rendering label-merged grids ===")
        for pose_name in poses:
            _render_merged_grid_for_pose(
                pose_envs[pose_name], pose_name,
                pose_label_clusters[pose_name], pose_rollouts[pose_name],
                code_colors, usage, cfg, merged_dir,
                code_labels=code_labels,
                filename_prefix="merged_by_label",
                title_kind="label-merged clusters",
            )

    log.info(f"\n=== Done. Artifacts under {out_base}/ ===")
    log.info(f"  Rollouts:        {data_dir}/")
    log.info(f"  Raw grids (mp4): {videos_raw_dir}/")
    log.info(f"  Merged:          {merged_dir}/")
    log.info(f"  Labels:          {labels_dir}/")


def _augment_clusters_with_labels(
    path: Path,
    clusters: list[list[int]],
    code_labels: dict[int, dict[str, Any]],
) -> None:
    """Add ``label`` field to each cluster entry in a clusters JSON in place."""
    import json
    with path.open() as f:
        payload = json.load(f)
    for i, cl in enumerate(clusters):
        if i < len(payload.get("clusters", [])):
            payload["clusters"][i]["label"] = cluster_majority_label(
                cl, code_labels,
            )
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
