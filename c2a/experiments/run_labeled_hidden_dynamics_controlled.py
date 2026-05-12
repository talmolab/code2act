"""Experiment: controlled multi-code rollouts from fixed starting poses.

Companion to ``run_labeled_hidden_dynamics`` that runs a more targeted
diagnostic: hold each of the 5 directional-turn codes for a fixed
duration in a fixed ORDER, all from a controlled starting pose, in
**decoder-only** mode (``z_e_scale = 0``).

The matrix of rollouts is::

    starting pose          ×   code sequence
    --------------------------+----------------------------------------
    walking-pose (low Z root) |  walk → turn_R → turn_L → walk → rear
                              |  rear → walk → turn_R → turn_L → walk
                              |  immobile → walk → turn_R → turn_L → walk
    rearing-pose (high Z root)|  walk → turn_R → turn_L → walk → rear
                              |  rear → walk → turn_R → turn_L → walk
                              |  immobile → walk → turn_R → turn_L → walk

That's 2 × 3 = 6 rollouts. Each rollout has K bodies (default 10) all
starting from the same pose-type and receiving the same per-frame code
sequence; each body uses a different test-clip reset so we get a
spread of within-pose variability.

Key design choices (with rationale)
-----------------------------------

* **Decoder-only mode** (``z_e_scale = 0.0``): the network's continuous
  encoder reads ``[task_obs, proprio]`` and produces ``z_e``. Setting
  ``z_e_scale = 0`` multiplies ``z_e`` by zero before it enters the GRU,
  effectively removing the encoder's contribution. The decoder runs
  autoregressively on ``[code_embedding, 0, proprio]`` with proprio
  updated each step by the simulator. This is what the network factory
  was built to support — see ``MoSeqRecurrentDecoderNetwork``
  (``z_e_scale: 0.0 = decoder-only``).

* **Starting pose = "walking" / "rearing"**: implemented as start-frame
  selection from the labels CSV. For "walking-pose start", we pick K
  test clips whose directional-turn label at the configured probe
  frame is ``Walk`` (low root-Z, body laid out); same for ``Rear``
  (high root-Z, body upright). The env is built with
  ``start_frame_range = [probe_frame, probe_frame]`` so the reset
  pose is exactly the labeled walking/rearing frame.

* **Hold duration per code = 5 s** (configurable). The trained policy
  saw 2 s clips, so 5 s is 2.5× horizon — long enough to see settling
  but short enough to stay in a reasonable regime. The full sequence
  is 5 codes × 5 s = 25 s per video.

* **Joint PCA across all 6 rollouts**: a single PCA is fit on the
  concatenation of all hidden trajectories so the rotating-PCA panel
  in every video uses identical axes. Cross-video trajectory
  comparison is then visually meaningful.

* **K random labeled frames per pose**: K clips drawn (seeded,
  without replacement) from the pool of test clips whose probe-frame
  label matches the desired pose. Each body in a rollout uses one of
  those K clips. Same K clips used across all sequences for a given
  starting pose so cross-sequence comparisons hold the start pose
  constant.

Outputs (under ``output.base_dir``)
-----------------------------------

* ``data/hidden_<pose>_<seq>.npz`` — for each of 6 (pose, sequence)
  combinations: ``hidden``, ``qpos``, ``proprio``, ``codes``,
  ``rewards`` arrays of shape ``(K, T, ...)``, plus the K reset clip
  indices and the resolved code sequence.
* ``data/pca_basis.npz`` — fit PCA components + variance explained;
  used by re-render runs.
* ``videos/<pose>_<seq>.mp4`` — ghost + rotating-PCA + phase banner
  composite per (pose, sequence). 6 files total.

Usage::

    cd code2act
    python -m c2a.experiments.run_labeled_hidden_dynamics_controlled
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
from pathlib import Path
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from c2a.paths import get_c2a_root, get_repo_root

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()

from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.codegen.code_sources import (
    SOURCE_GT_DIRECTIONAL_TURN,
    load_codes,
)
from c2a.experiments.run_hidden_dynamics import run_rollout_with_hidden
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.codegen.kpms.label_alignment import (
    BEHAVIORS,
    BEHAVIOR_TO_INT,
    DIRECTIONAL_TURN_LABELS,
    DIRECTIONAL_TURN_TO_INT,
    UNLABELED,
    load_label_frames,
)
from c2a.moseq_env_wrapper import MoSeqImitation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _abs(p) -> str:
    """Resolve ``p`` against the repo root if it's relative."""
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Starting-pose clip selection
# ---------------------------------------------------------------------------


def _pick_start_clips(
    pose_label: str,
    test_indices: np.ndarray,
    behavior_array: np.ndarray,
    probe_frame: int,
    K: int,
    seed: int,
    probe_frame_max: int | None = None,
) -> tuple[np.ndarray, int]:
    """Sample K test-clip indices whose behavior matches at some frame.

    Driven off the raw ``behavior`` track (4-class:
    ``Immobile/Rear/Turn/Walk``) rather than the 5-class
    directional-turn projection. Pose selection only cares about
    behavior (``Walk`` = low root-Z, ``Rear`` = high root-Z,
    ``Immobile`` = no displacement); the ``Turn`` direction is
    irrelevant for what the rodent looks like at reset.

    Frame search policy:
        * If ``probe_frame_max is None`` (or equal to ``probe_frame``):
          strict — match ONLY at ``probe_frame``, raise if fewer than
          K clips qualify.
        * Otherwise: scan ``probe_frame, probe_frame+1, …,
          probe_frame_max`` and return the earliest frame at which
          ≥K clips match. This handles poses that emerge later in the
          clip (e.g. ``Walk`` rarely appears at frame 20 because clips
          start with the rodent stationary or rearing).

    Args:
        pose_label: one of ``BEHAVIORS``.
        test_indices: ids of clips in the test split.
        behavior_array: ``(N_clips, T)`` int32 behavior labels with
            values in ``{-1, 0, 1, 2, 3}`` (UNLABELED + BEHAVIORS).
        probe_frame: earliest frame to consider. Must be in
            ``[0, behavior_array.shape[1])``.
        K: number of clips to sample.
        seed: PRNG seed.
        probe_frame_max: latest frame to consider (inclusive). When
            ``None`` or ``< probe_frame``, falls back to strict mode.

    Returns:
        ``(clip_ids[K], chosen_probe_frame)`` — the K sampled test-clip
        ids and the frame at which they were sampled. Use
        ``chosen_probe_frame`` to set the env's ``start_frame_range``
        so the reset pose is exactly the frame whose label was checked.

    Raises:
        ValueError: ``pose_label`` not in ``BEHAVIORS`` or
            ``probe_frame``/``probe_frame_max`` out of range.
        RuntimeError: no frame in the scan range had ≥K matches; the
            error includes a per-class coverage table at the *initial*
            probe frame to help pick a different pose or widen the
            range.
    """
    if pose_label not in BEHAVIOR_TO_INT:
        raise ValueError(
            f"pose_label={pose_label!r} not in {BEHAVIORS}. "
            f"(Note: starting-pose selection uses the 4-class behavior "
            f"track, not the 5-class directional-turn projection.)"
        )
    target = BEHAVIOR_TO_INT[pose_label]

    T_max = int(behavior_array.shape[1])
    if probe_frame < 0 or probe_frame >= T_max:
        raise ValueError(
            f"probe_frame={probe_frame} out of range [0, {T_max})."
        )
    if probe_frame_max is None:
        scan_end = probe_frame
    else:
        scan_end = int(probe_frame_max)
        if scan_end < probe_frame:
            scan_end = probe_frame
        if scan_end >= T_max:
            scan_end = T_max - 1

    # Restrict test clips to those addressable by behavior_array.
    in_range = [
        int(c) for c in test_indices if int(c) < behavior_array.shape[0]
    ]
    if not in_range:
        raise RuntimeError(
            "No test clip ids are within behavior_array's clip range; "
            "check labeled_clip_range vs balanced_split test_indices."
        )

    in_range_arr = np.asarray(in_range, dtype=np.int64)

    # Vectorised scan: for each candidate frame, count matches across
    # the in-range test clips, take the earliest frame that hits K.
    matches_per_frame = (
        behavior_array[in_range_arr, probe_frame:scan_end + 1] == target
    )  # (n_in_range, n_frames_in_scan)
    counts = matches_per_frame.sum(axis=0)  # (n_frames_in_scan,)
    ok = np.where(counts >= K)[0]
    if ok.size == 0:
        # Coverage at the initial probe_frame for the error message.
        coverage: dict[str, int] = {b: 0 for b in BEHAVIORS}
        coverage["UNLABELED"] = 0
        col = behavior_array[in_range_arr, probe_frame]
        for v in col.tolist():
            v = int(v)
            if v == UNLABELED:
                coverage["UNLABELED"] += 1
            elif 0 <= v < len(BEHAVIORS):
                coverage[BEHAVIORS[v]] += 1
        cov_str = " ".join(f"{k}={v}" for k, v in coverage.items())
        max_in_scan = int(counts.max()) if counts.size > 0 else 0
        argmax_in_scan = probe_frame + int(counts.argmax()) if counts.size > 0 else probe_frame
        raise RuntimeError(
            f"Fewer than K={K} test clips match behavior={pose_label!r} "
            f"anywhere in frame range [{probe_frame}, {scan_end}]. "
            f"Best count in range: {max_in_scan} at frame "
            f"{argmax_in_scan}. Coverage at frame {probe_frame} across "
            f"{len(in_range)} test clips: {cov_str}. Widen "
            f"start_pose.probe_frame_max, lower K, or pick a different "
            f"pose label."
        )

    chosen_frame = probe_frame + int(ok[0])
    pool = in_range_arr[matches_per_frame[:, ok[0]]]
    rng = np.random.RandomState(seed)
    clip_ids = rng.choice(pool, size=K, replace=False)
    return clip_ids, chosen_frame


# ---------------------------------------------------------------------------
# Label-name → code-int resolution
# ---------------------------------------------------------------------------


def _resolve_name_to_code(
    name: str, ckpt_cfg: DictConfig, num_codes: int,
) -> int:
    """Map a directional-turn label name to a KPMS code integer.

    For ``gt_directional_turn``-trained policies the mapping is the
    identity ``DIRECTIONAL_TURN_TO_INT[name]``. For other code sources,
    this raises — those policies don't have a clean 5-class vocabulary,
    so the user must extend this function (or pass per-sequence integer
    codes directly in the yaml) before running.
    """
    code_source = str(ckpt_cfg.kpms_config.get("code_source", ""))
    if code_source == SOURCE_GT_DIRECTIONAL_TURN:
        if name not in DIRECTIONAL_TURN_TO_INT:
            raise ValueError(
                f"Unknown directional-turn name={name!r}. "
                f"Expected one of {DIRECTIONAL_TURN_LABELS}."
            )
        code = DIRECTIONAL_TURN_TO_INT[name]
        if code >= num_codes:
            raise RuntimeError(
                f"Resolved code {code} for {name!r} exceeds num_codes "
                f"= {num_codes}. The checkpoint's code embedding table "
                f"can't address this code."
            )
        return code
    raise NotImplementedError(
        f"Sequence-name resolution for code_source={code_source!r} is "
        f"not implemented. Either retrain the policy with "
        f"code_source=gt_directional_turn or pass per-sequence integer "
        f"codes directly (extend this function)."
    )


def _build_code_sequence(
    sequence_names: list[str], steps_per_code: int,
    ckpt_cfg: DictConfig, num_codes: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Concatenate ``[code] * steps_per_code`` blocks into one (T,) array.

    Returns:
        ``(codes, phase_boundaries, phase_code_ints)``:
          * ``codes`` — shape ``(len(sequence_names) * steps_per_code,)``
            int32 array of per-step code ids.
          * ``phase_boundaries`` — frame indices that delimit each
            phase. ``len == len(sequence_names) + 1``, starts at 0,
            ends at ``T``.
          * ``phase_code_ints`` — resolved code integer per phase
            (in order). Useful for diagnostics.
    """
    ints = [
        _resolve_name_to_code(n, ckpt_cfg, num_codes) for n in sequence_names
    ]
    blocks = [
        np.full(steps_per_code, c, dtype=np.int32) for c in ints
    ]
    codes = np.concatenate(blocks)
    boundaries = [i * steps_per_code for i in range(len(sequence_names) + 1)]
    return codes, boundaries, ints


# ---------------------------------------------------------------------------
# Env construction (one per starting pose)
# ---------------------------------------------------------------------------


def _build_pose_env(
    ckpt_cfg: DictConfig,
    test_clips_data_path: str,
    clip_len: int,
    keep_clips_idx: np.ndarray,
    test_codes: np.ndarray,
    code_stack_size: int,
    probe_frame: int,
    disable_far_termination: bool = True,
):
    """Build a MoSeqImitation env constrained to ``keep_clips_idx`` and
    forced to reset at ``probe_frame``.
    """
    clips = ReferenceClips(
        data_path=test_clips_data_path,
        n_frames_per_clip=clip_len,
        keep_clips_idx=np.asarray(keep_clips_idx, dtype=np.int64),
    )
    # The test_codes argument must match the clips set in length AND
    # ordering. Subset rows to align.
    # NB: test_codes is indexed by POSITION in the original test split,
    # which is already the order ReferenceClips preserves when given
    # keep_clips_idx. So we need to map keep_clips_idx → positions in
    # the test split and subset.
    # We delay this to the caller (it knows the mapping).

    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [probe_frame, probe_frame]
    env_cfg.domain_randomization.use_domain_randomization = False
    if disable_far_termination:
        if hasattr(env_cfg.termination_criteria, "root_too_far"):
            env_cfg.termination_criteria.root_too_far.max_distance = 1e6
        if hasattr(env_cfg.termination_criteria, "root_too_rotated"):
            env_cfg.termination_criteria.root_too_rotated.max_degrees = 1e6
        if hasattr(env_cfg.termination_criteria, "pose_error"):
            env_cfg.termination_criteria.pose_error.max_l2_error = 1e6
    env = MoSeqImitation(
        config=env_cfg, clips=clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="labeled_hidden_dynamics_controlled_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Controlled labeled-behavior hidden-state experiment ===")

    output_dir = Path(_abs(cfg.output.base_dir))
    data_dir = output_dir / "data"
    video_dir = output_dir / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint + decoder-only inference fn
    # ------------------------------------------------------------------
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path),
    )
    if not bool(ckpt_cfg.network_config.get("use_rnn_decoder", False)):
        raise ValueError(
            "This experiment requires an RNN decoder checkpoint."
        )
    z_e_scale = float(cfg.get("z_e_scale", 0.0))
    log.info(
        f"Inference fn: deterministic=True, z_e_scale={z_e_scale}. "
        f"z_e_scale=0 ⇒ decoder-only (encoder bypassed)."
    )
    inf_fn = make_inference_fn(
        ppo_networks, use_rnn=True,
        deterministic=True, z_e_scale=z_e_scale,
    )
    params = (norm_state, policy_params)

    # ------------------------------------------------------------------
    # Codes (same source the policy was trained with) + balanced split
    # ------------------------------------------------------------------
    splits = load_balanced_splits(_abs(cfg.data.balanced_split_path))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    clip_len = int(ckpt_cfg.env_config.clip_length)

    codes_payload = load_codes(ckpt_cfg, train_indices, test_indices, clip_len)
    test_codes_full = codes_payload.test_codes
    num_codes = int(codes_payload.num_codes)
    log.info(
        f"Codes: source={codes_payload.code_source}, num_codes={num_codes}"
    )

    # ------------------------------------------------------------------
    # Labels for starting-pose selection. Use the raw behavior track
    # (4-class: Immobile/Rear/Turn/Walk) rather than the 5-class
    # directional-turn projection — pose only cares about behavior,
    # not turn direction, and the behavior track is densely labeled
    # even at early-clip frames where direction may be UNLABELED.
    # ------------------------------------------------------------------
    label_arrays = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    behavior_arr = label_arrays.behavior

    # ------------------------------------------------------------------
    # Pick K test clips per starting pose. Each pose can land on a
    # different frame within the configured scan range — Walk typically
    # appears tens of frames into the clip while Rear/Immobile are
    # common right from frame 20. We pick per-pose probe frames and
    # propagate them to the env's start_frame_range below.
    # ------------------------------------------------------------------
    K = int(cfg.K_per_rollout)
    probe_frame = int(cfg.start_pose.probe_frame)
    probe_frame_max = cfg.start_pose.get("probe_frame_max", None)
    if probe_frame_max is not None:
        probe_frame_max = int(probe_frame_max)
    seed = int(cfg.seed)

    walking_clips, walking_probe_frame = _pick_start_clips(
        cfg.start_pose.walking_label, test_indices, behavior_arr,
        probe_frame=probe_frame, probe_frame_max=probe_frame_max,
        K=K, seed=seed,
    )
    rearing_clips, rearing_probe_frame = _pick_start_clips(
        cfg.start_pose.rearing_label, test_indices, behavior_arr,
        probe_frame=probe_frame, probe_frame_max=probe_frame_max,
        K=K, seed=seed + 1,
    )
    log.info(
        f"Start clips — walking-pose ({cfg.start_pose.walking_label}) "
        f"@ frame {walking_probe_frame}: {walking_clips.tolist()}"
    )
    log.info(
        f"Start clips — rearing-pose ({cfg.start_pose.rearing_label}) "
        f"@ frame {rearing_probe_frame}: {rearing_clips.tolist()}"
    )

    # ------------------------------------------------------------------
    # Build TWO envs (one per starting pose). Each env restricts to its
    # K clips and forces start_frame_range = [probe_frame, probe_frame].
    # The reset_clip_idx kwarg of run_rollout_with_hidden then picks
    # which of the K clips to start from for a given body.
    # ------------------------------------------------------------------
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))

    # For each pose, build a (K, T) test_codes matrix aligned to the
    # `keep_clips_idx` row order. ReferenceClips preserves the order
    # given in keep_clips_idx, so test_codes_pose[i] must correspond
    # to clip id keep_clips_idx[i].
    def _test_codes_for_pose(pose_clips: np.ndarray) -> np.ndarray:
        # Map each clip id in pose_clips to its position in test_indices.
        test_pos_lookup = {int(c): i for i, c in enumerate(test_indices)}
        rows = []
        for clip_id in pose_clips:
            pos = test_pos_lookup[int(clip_id)]
            rows.append(test_codes_full[pos])
        return np.stack(rows, axis=0)

    test_codes_walking = _test_codes_for_pose(walking_clips)
    test_codes_rearing = _test_codes_for_pose(rearing_clips)

    # Decide reuse before paying the JIT-compile cost. Envs are still
    # built unconditionally because the rendering phase later needs
    # ghost models extracted from them.
    sequence_names_for_paths = list(cfg.sequences.keys())
    expected_npz_paths = [
        data_dir / f"hidden_{pose_name}_{seq_name}.npz"
        for pose_name in ("walking", "rearing")
        for seq_name in sequence_names_for_paths
    ]
    reuse_rollouts_cfg = bool(cfg.get("reuse_rollouts", True))
    can_reuse_rollouts = (
        reuse_rollouts_cfg and all(p.exists() for p in expected_npz_paths)
    )
    if reuse_rollouts_cfg and not can_reuse_rollouts:
        missing = [p.name for p in expected_npz_paths if not p.exists()]
        log.info(
            f"reuse_rollouts=true but {len(missing)} of "
            f"{len(expected_npz_paths)} rollout npzs missing "
            f"({missing[:3]}{'...' if len(missing) > 3 else ''}); "
            f"running rollouts fresh."
        )
    elif can_reuse_rollouts:
        log.info(
            f"reuse_rollouts=true and all {len(expected_npz_paths)} "
            f"rollout npzs present under {data_dir.relative_to(REPO_ROOT) if str(data_dir).startswith(str(REPO_ROOT)) else data_dir}/ "
            f"— skipping JIT-compile + rollout loop."
        )

    log.info("Building walking-pose env...")
    env_walking = _build_pose_env(
        ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
        keep_clips_idx=walking_clips, test_codes=test_codes_walking,
        code_stack_size=code_stack_size, probe_frame=walking_probe_frame,
    )
    jit_reset_w = (
        jax.jit(env_walking.reset) if not can_reuse_rollouts else None
    )
    jit_step_w = (
        jax.jit(env_walking.step) if not can_reuse_rollouts else None
    )

    log.info("Building rearing-pose env...")
    env_rearing = _build_pose_env(
        ckpt_cfg, _abs(cfg.data.reference_data_path), clip_len,
        keep_clips_idx=rearing_clips, test_codes=test_codes_rearing,
        code_stack_size=code_stack_size, probe_frame=rearing_probe_frame,
    )
    jit_reset_r = (
        jax.jit(env_rearing.reset) if not can_reuse_rollouts else None
    )
    jit_step_r = (
        jax.jit(env_rearing.step) if not can_reuse_rollouts else None
    )

    # ------------------------------------------------------------------
    # Resolve sequences
    # ------------------------------------------------------------------
    sequences: dict[str, list[str]] = dict(cfg.sequences)
    if not sequences:
        raise ValueError("cfg.sequences is empty.")

    hold_seconds = float(cfg.hold_seconds_per_code)
    ctrl_hz = float(cfg.get("ctrl_hz", 100.0))
    steps_per_code = int(round(hold_seconds * ctrl_hz))
    log.info(
        f"Hold {hold_seconds}s × {ctrl_hz}Hz = {steps_per_code} ctrl "
        f"steps per code; sequence length = "
        f"{steps_per_code} × N_codes per sequence."
    )

    resolved_sequences: dict[str, dict] = {}
    for seq_name, code_names in sequences.items():
        codes_arr, phase_bounds, phase_ints = _build_code_sequence(
            list(code_names), steps_per_code, ckpt_cfg, num_codes,
        )
        resolved_sequences[seq_name] = {
            "names": list(code_names),
            "codes": codes_arr,
            "phase_bounds": phase_bounds,
            "phase_code_ints": phase_ints,
        }
        log.info(
            f"  Sequence {seq_name!r}: {' → '.join(code_names)} "
            f"(codes={phase_ints}, T={len(codes_arr)})"
        )

    # ------------------------------------------------------------------
    # Run rollouts: 2 starting poses × 3 sequences × K bodies
    # ------------------------------------------------------------------
    pose_specs = [
        ("walking", env_walking, jit_reset_w, jit_step_w, walking_clips),
        ("rearing", env_rearing, jit_reset_r, jit_step_r, rearing_clips),
    ]

    pose_to_probe_frame = {
        "walking": walking_probe_frame,
        "rearing": rearing_probe_frame,
    }
    pose_to_clips = {
        "walking": walking_clips,
        "rearing": rearing_clips,
    }

    rollouts: dict[tuple[str, str], dict[str, Any]] = {}

    if can_reuse_rollouts:
        for pose_name in ("walking", "rearing"):
            for seq_name in resolved_sequences.keys():
                npz_path = data_dir / f"hidden_{pose_name}_{seq_name}.npz"
                d = np.load(npz_path, allow_pickle=True)
                agg = {k: d[k] for k in d.files}
                # Numpy 0-d / 1-elt arrays of strings come back wrapped;
                # unwrap pose_name / seq_name for the rendering pass.
                agg["pose_name"] = str(agg["pose_name"])
                agg["seq_name"] = str(agg["seq_name"])
                rollouts[(pose_name, seq_name)] = agg
                log.info(
                    f"  Loaded {npz_path.name}: hidden={agg['hidden'].shape}, "
                    f"qpos={agg['qpos'].shape}"
                )
    else:
        for pose_name, env, jit_reset, jit_step, pose_clips in pose_specs:
            for seq_name, sinfo in resolved_sequences.items():
                log.info(
                    f"\n=== Pose={pose_name} | Sequence={seq_name} ==="
                )
                codes_arr = sinfo["codes"]
                T_total = int(codes_arr.shape[0])

                hidden_all: list[np.ndarray] = []
                qpos_all: list[np.ndarray] = []
                proprio_all: list[np.ndarray] = []
                codes_actual_all: list[np.ndarray] = []
                rewards_all: list[np.ndarray] = []

                for k, clip_id in enumerate(pose_clips):
                    # reset_clip_idx is the POSITION within
                    # keep_clips_idx, i.e. k itself (each body uses its
                    # k-th clip).
                    # NOTE on "survival": run_rollout_with_hidden (the
                    # run_hidden_dynamics flavour) NEVER breaks on
                    # state.done — the loop always runs the full
                    # max_steps and merely records the step at which
                    # `done` first fires. The env's imitation-clip
                    # truncation (cur_frame > clip_length -
                    # reference_length - 1) flips `done` around ctrl
                    # step ~448 with start_frame≈20, but physics keeps
                    # integrating, MJX state keeps evolving, and
                    # qpos/hidden/proprio remain valid through
                    # T_total. In decoder-only mode (z_e_scale=0) the
                    # stale imitation target is irrelevant because the
                    # encoder's contribution is zeroed at the GRU
                    # input. The `survival` number reported below is
                    # therefore just informational ("when did the
                    # imitation clock expire") and not a loss-of-data
                    # signal.
                    key = jax.random.PRNGKey(
                        seed + 1000 * len(rollouts) + k
                    )
                    result = run_rollout_with_hidden(
                        env, inf_fn, params, ppo_networks, key,
                        max_steps=T_total,
                        code_override=codes_arr,
                        reset_clip_idx=int(k),
                        jit_reset=jit_reset, jit_step=jit_step,
                    )

                    def _rect(a: np.ndarray) -> np.ndarray:
                        """Pad / truncate so each body's array is
                        exactly (T_total, ...) — termination would
                        otherwise leave a ragged tensor across the K
                        bodies."""
                        if a.shape[0] >= T_total:
                            return a[:T_total]
                        pad = np.zeros(
                            (T_total - a.shape[0],) + a.shape[1:],
                            dtype=a.dtype,
                        )
                        return np.concatenate([a, pad], axis=0)

                    hidden_all.append(_rect(result["hidden_states"]))
                    qpos_all.append(_rect(result["qpos"]))
                    proprio_all.append(_rect(result["proprio"]))
                    codes_actual_all.append(_rect(result["code_indices"]))
                    rewards_all.append(_rect(result["rewards"]))
                    log.info(
                        f"  body {k} (clip {int(clip_id)}): "
                        f"first_done_step={result['survival']}/{T_total} "
                        f"(rollout completes T_total regardless), "
                        f"mean_reward={np.mean(result['rewards']):.3f}"
                    )

                agg = {
                    "hidden": np.stack(hidden_all, axis=0),
                    "qpos": np.stack(qpos_all, axis=0),
                    "proprio": np.stack(proprio_all, axis=0),
                    "codes": np.stack(codes_actual_all, axis=0),
                    "rewards": np.stack(rewards_all, axis=0),
                    "start_clips": np.asarray(pose_clips, dtype=np.int64),
                    "code_sequence": codes_arr,
                    "phase_bounds": np.asarray(
                        sinfo["phase_bounds"], dtype=np.int64,
                    ),
                    "phase_names": np.asarray(sinfo["names"], dtype=object),
                    "phase_code_ints": np.asarray(
                        sinfo["phase_code_ints"], dtype=np.int64,
                    ),
                    "pose_name": pose_name,
                    "seq_name": seq_name,
                    "probe_frame": np.int32(pose_to_probe_frame[pose_name]),
                    "z_e_scale": np.float32(z_e_scale),
                }
                rollouts[(pose_name, seq_name)] = agg

                # Per-rollout npz (granular, easy to re-render).
                out_npz = (
                    data_dir / f"hidden_{pose_name}_{seq_name}.npz"
                )
                np.savez_compressed(out_npz, **agg)
                log.info(f"  Saved: {out_npz}")

    # ------------------------------------------------------------------
    # Joint PCA across all 6 rollouts
    # ------------------------------------------------------------------
    log.info("\n=== Fitting joint PCA across all 6 rollouts ===")
    from c2a.experiments.shared.rotating_pca_video import (
        fit_joint_pca, _compute_pca_limits, render_rotating_pca_video,
    )
    pca, var_explained = fit_joint_pca(
        [r["hidden"] for r in rollouts.values()], n_components=3,
    )
    # Compute joint PCA-space limits so every video uses identical axes.
    emb_all = pca.transform(
        np.concatenate(
            [r["hidden"].reshape(-1, r["hidden"].shape[-1])
             for r in rollouts.values()], axis=0,
        )
    )
    xlim, ylim, zlim = _compute_pca_limits(emb_all)
    np.savez(
        data_dir / "pca_basis.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        xlim=np.asarray(xlim), ylim=np.asarray(ylim), zlim=np.asarray(zlim),
    )
    log.info(
        f"  PCA explained variance: "
        f"PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, "
        f"PC3={var_explained[2]:.3f}"
    )

    # ------------------------------------------------------------------
    # Render 6 composite videos
    # ------------------------------------------------------------------
    if not bool(cfg.get("render_video", True)):
        log.info("render_video=false; skipping video render.")
        return

    log.info("\n=== Rendering composite videos ===")
    render_kwargs = dict(
        fps=int(cfg.get("render_fps", 50)),
        panel_w=int(cfg.get("panel_w", 640)),
        panel_h=int(cfg.get("panel_h", 360)),
        banner_h=int(cfg.get("banner_h", 44)),
        azim_start=float(cfg.get("azim_start", 140.0)),
        azim_end=float(cfg.get("azim_end", 260.0)),
    )

    pose_to_env = {"walking": env_walking, "rearing": env_rearing}
    for (pose_name, seq_name), agg in rollouts.items():
        out_mp4 = video_dir / f"{pose_name}_{seq_name}.mp4"
        env = pose_to_env[pose_name]
        render_rotating_pca_video(
            hidden=agg["hidden"],
            qpos=agg["qpos"],
            phase_boundaries=list(agg["phase_bounds"].tolist()),
            phase_names=[str(n) for n in agg["phase_names"]],
            output_path=out_mp4,
            pca=pca, var_explained=var_explained,
            xlim=xlim, ylim=ylim, zlim=zlim,
            ghost_env=env,
            title_left="Hidden trajectory (joint PCA, decoder-only)",
            title_right=f"{pose_name} pose — {seq_name}",
            **render_kwargs,
        )

    log.info("\n=== Done ===")
    log.info(f"  6 npz files at {data_dir}/")
    log.info(f"  6 mp4 files at {video_dir}/")
    log.info(f"  Joint PCA basis at {data_dir}/pca_basis.npz")


if __name__ == "__main__":
    main()
