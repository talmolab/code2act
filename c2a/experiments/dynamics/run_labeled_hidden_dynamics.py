"""Experiment: RNN hidden state dynamics under MANUALLY LABELED behavior.

Counterpart to ``run_hidden_dynamics.py`` but anchored on explicit human
labels (``data/rodent/rodent_labels.csv`` from
https://github.com/KevinBian107/TopoMimic-annotation) instead of
kinematic heuristics over KPMS codes.

The schema is two-track: ``behavior`` ∈ {Immobile, Rear, Turn, Walk}
crossed with ``direction`` ∈ {Left, Right, Straight}. This experiment
projects the two tracks onto the **5-class directional-turn scheme**
that mirrors ``c2a.codegen.code_sources.SOURCE_GT_DIRECTIONAL_TURN``: direction
is collapsed for non-Turn behaviors so the rollouts cover

    Immobile, Rear, Walk, Turn_Left, Turn_Right.

(``Turn_Straight`` is absent in the current CSV; if a future CSV
introduces it, the directional-turn mapping leaves it UNLABELED, the
agreement-code finder ignores it, and no rollout is produced for it.)

Design (vs ``run_hidden_dynamics.py``)
--------------------------------------
The old script picked a single representative KPMS code per behavior by
**kinematic heuristic** (z_rise / xy_disp / xyz_disp) and held it for
max_steps frames.  That heuristic was fragile and indirect — it tried
to *infer* which code corresponds to walking instead of *asking the
labels*.

This experiment uses the same sustained-code-with-K-bodies design but
selects the code via **label agreement**:

1.  For each populated directional-turn label L, count the KPMS codes
    occurring on the manually-labeled frames of L across the entire
    labeled corpus.  The mode is the *agreement code* for L.  We print
    the agreement fraction (= mode count / total labeled frames) for
    each label so the user can see how cleanly KPMS clusters that
    behavior.
2.  Pick **K starting body positions** ONCE from the test split (same
    positions used across every label, so the only difference between
    rollouts of two labels is the held code).
3.  For each label with a valid agreement code, sustain that code for
    ``target_steps`` control steps × K bodies, capturing hidden /
    qpos / proprio / codes / rewards.
4.  **Termination criteria are disabled** (`root_too_far`,
    `root_too_rotated`, `pose_error`) so the policy can drift over the
    full 1000-step window without auto-reset producing trajectory
    discontinuities. ``nan_termination`` is kept as a safety net.
5.  Render one ghost-overlay MP4 per label.

Outputs
-------
* ``outputs/<base_dir>/data/hidden_dynamics_labeled.npz``::

     agreement_codes:    object array of (label, code, agreement, n_frames)
     starting_positions: (K,) int — test-clip positions used for resets
     hidden_<label>:     (K, T, hidden_dim)
     qpos_<label>:       (K, T, nq)
     proprio_<label>:    (K, T, P)
     codes_<label>:      (K, T)
     rewards_<label>:    (K, T)

  Labels are the 5-class directional-turn strings: ``Immobile``,
  ``Rear``, ``Walk``, ``Turn_Left``, ``Turn_Right``. Plot scripts
  (``plot_hidden_dsa.py`` / ``plot_hidden_rslds.py``) detect this as
  legacy (no level prefix, no ``__`` separator) and process each
  ``hidden_<label>`` array as one behavior class.

* ``outputs/<base_dir>/data/labeled_<label>.mp4`` — per-label
  ghost-overlay verification clip (K bodies running the same code).

Usage::

    cd code2act
    python -m c2a.experiments.dynamics.run_labeled_hidden_dynamics
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import shutil
from pathlib import Path
from typing import Any

from c2a.paths import get_c2a_root, get_repo_root
import h5py
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()
from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from c2a.moseq_env_wrapper import MoSeqImitation

from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.ghost_rendering import (
    build_ghost_model,
    render_ghost_video,
)
from c2a.experiments.shared.plotting import get_trajectory_colors

from c2a.codegen.shared.label_alignment import (
    DIRECTIONAL_TURN_LABELS,
    UNLABELED,
    directional_turn_array,
    load_label_frames,
)
from c2a.experiments.dynamics.run_hidden_dynamics import run_rollout_with_hidden

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution (Hydra-cwd aware)
# ---------------------------------------------------------------------------


def _abs(p) -> str:
    """Resolve ``p`` against the repo root if relative.

    Hydra changes cwd to ``outputs/<date>/<time>/`` before main(), and
    Orbax requires absolute checkpoint paths.  Always go through this.
    """
    p = Path(p)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


# ---------------------------------------------------------------------------
# Agreement-code identification
# ---------------------------------------------------------------------------


def _find_agreement_codes(
    label_array: np.ndarray,
    label_clip_indices: np.ndarray,
    label_names: tuple[str, ...],
    all_codes: np.ndarray,
    code_clip_indices: np.ndarray,
    intersect_clips: set[int],
    num_codes: int,
    min_label_frames: int,
    selection_rule: str = "mode_excluding",
    exclude_codes: tuple[int, ...] = (0,),
    min_within_label_frac: float = 0.05,
) -> dict[str, dict]:
    """Build the (label → KPMS code) lookup table.

    Generic over the label scheme — pass any per-frame integer label
    array indexed by ``range(len(label_names))`` plus UNLABELED (-1).
    Use :data:`c2a.codegen.shared.label_alignment.COMBINED_LABELS` for the
    12-class combined behavior×direction breakdown, or
    :data:`DIRECTIONAL_TURN_LABELS` for the 5-class
    {Immobile, Rear, Walk, Turn_Left, Turn_Right} breakdown.

    Three selection rules, configurable via ``selection_rule``:

    * ``"mode"``: argmax P(c | L). Simplest. Vulnerable to KPMS junk
      clusters (the current ``best_codes.npz`` has code 0 dominating
      ~26% of all frames, so mode picks code 0 for every label).
    * ``"mode_excluding"``: argmax P(c | L) restricted to codes not in
      ``exclude_codes``. Default. With ``exclude_codes=(0,)`` this skips
      the junk cluster and produces useful per-label codes for ~3 of 7
      labels, with the rest still colliding on the next-most-common
      shared code (code 2 in the current model).
    * ``"enrichment"``: argmax P(c | L) / P(c) subject to
      ``P(c | L) >= min_within_label_frac``. Picks the code most
      *over-represented* in this label vs. the corpus base rate, while
      requiring the code to actually appear with reasonable frequency
      within the label (otherwise pure enrichment over-fits on rare
      codes that happen to occur once in only one label).

    Args:
        label_array: ``(N_clips, T)`` int32 with values in
            ``{-1, 0, …, len(label_names) - 1}`` (UNLABELED = -1).
        label_clip_indices: clip-id index for ``label_array`` rows.
        label_names: per-class string names. Index ``i`` of
            ``label_names`` must correspond to the integer ``i`` used in
            ``label_array``.
        all_codes: ``(N_clips_codes, T)`` KPMS code per mocap frame.
        code_clip_indices: clip-id index for ``all_codes`` rows.
        intersect_clips: set of clip ids to use (in both label & code views).
        num_codes: total KPMS codes (histogram bin count).
        min_label_frames: drop labels below this total-frame threshold.
        selection_rule: see above.
        exclude_codes: codes to ignore for ``mode_excluding`` (junk clusters).
        min_within_label_frac: floor for ``enrichment`` rule.

    Returns:
        ``{label_str: {code, agreement, n_frames, histogram, rule_used,
        top3, base_rate_top3}}``. ``top3`` is a ``[(code, P(c|L))]`` list
        of the three most common codes in this label (always reported
        for diagnostics regardless of selection rule).
    """
    if selection_rule not in ("mode", "mode_excluding", "enrichment"):
        raise ValueError(
            f"unknown selection_rule={selection_rule!r}; expected one of "
            f"('mode', 'mode_excluding', 'enrichment')"
        )

    n_label_classes = len(label_names)
    clip_to_label_row = {int(c): i for i, c in enumerate(label_clip_indices)}
    clip_to_code_row = {int(c): i for i, c in enumerate(code_clip_indices)}

    histograms = np.zeros((n_label_classes, num_codes), dtype=np.int64)
    for clip_id in intersect_clips:
        if clip_id not in clip_to_label_row or clip_id not in clip_to_code_row:
            continue
        lbl_row = clip_to_label_row[clip_id]
        code_row = clip_to_code_row[clip_id]
        per_frame_label = label_array[lbl_row]
        per_frame_code = all_codes[code_row]
        for label_int in range(n_label_classes):
            mask = per_frame_label == label_int
            if not mask.any():
                continue
            codes_here = per_frame_code[mask].astype(np.int64)
            np.add.at(histograms[label_int], codes_here, 1)

    # Base rate: P(c) over all labeled frames. Used by 'enrichment' rule.
    base_counts = histograms.sum(axis=0).astype(np.float64)
    base_rate = base_counts / max(base_counts.sum(), 1.0)

    excl = set(int(c) for c in (exclude_codes or ()))

    out: dict[str, dict] = {}
    for label_int, label_str in enumerate(label_names):
        h = histograms[label_int].astype(np.float64)
        n_frames = int(h.sum())
        if n_frames < min_label_frames:
            continue

        within = h / n_frames

        if selection_rule == "mode":
            code = int(within.argmax())
        elif selection_rule == "mode_excluding":
            mask = np.ones(num_codes, dtype=bool)
            for c in excl:
                if 0 <= c < num_codes:
                    mask[c] = False
            scored = np.where(mask, within, -np.inf)
            if not np.isfinite(scored).any():
                # All codes excluded — fall back to plain mode.
                code = int(within.argmax())
            else:
                code = int(scored.argmax())
        else:  # enrichment
            enrich = within / np.maximum(base_rate, 1e-9)
            floor_mask = within >= min_within_label_frac
            scored = np.where(floor_mask, enrich, -np.inf)
            if not np.isfinite(scored).any():
                # No code clears the floor — fall back to mode_excluding.
                mask = np.ones(num_codes, dtype=bool)
                for c in excl:
                    if 0 <= c < num_codes:
                        mask[c] = False
                code = int(np.where(mask, within, -np.inf).argmax())
            else:
                code = int(scored.argmax())

        agreement = float(within[code])

        # Top-3 diagnostic.
        top_idx = np.argsort(-h)[:3]
        top3 = [(int(c), float(within[int(c)])) for c in top_idx]

        out[label_str] = {
            "code": code,
            "agreement": agreement,
            "n_frames": n_frames,
            "histogram": histograms[label_int].copy(),
            "rule_used": selection_rule,
            "top3": top3,
        }
    return out


def _print_agreement_table(
    agreement: dict[str, dict],
    skipped_labels: list[tuple[str, int]],
    selection_rule: str,
    exclude_codes: tuple[int, ...],
) -> None:
    """Pretty-print the agreement-code lookup with full diagnostics."""
    log.info(
        f"=== Agreement codes (rule={selection_rule}, "
        f"exclude={list(exclude_codes)}) ==="
    )
    log.info(
        f"  {'label':<22s} | {'code':>4s}  {'agree':>6s}  {'n':>8s}  "
        f"{'top-3 (code: P(c|L))':<40s}"
    )
    log.info(f"  {'-' * 22}-+-{'-' * 4}--{'-' * 6}--{'-' * 8}--{'-' * 40}")
    for label, info in agreement.items():
        top3_str = ", ".join(f"{c}:{p:.2f}" for c, p in info["top3"])
        log.info(
            f"  {label:<22s} | {info['code']:>4d}  "
            f"{info['agreement']:>6.3f}  {info['n_frames']:>8d}  {top3_str}"
        )
    if skipped_labels:
        log.info("Skipped (too few labeled frames):")
        for label, n in skipped_labels:
            log.info(f"  {label:<22s} | {n:>3d} frames")

    # Collisions: multiple labels with the same agreement code → those rollouts
    # will produce identical hidden trajectories (only K starting poses differ).
    # Useful as a control but worth flagging.
    by_code: dict[int, list[str]] = {}
    for label, info in agreement.items():
        by_code.setdefault(info["code"], []).append(label)
    shared = {c: labs for c, labs in by_code.items() if len(labs) > 1}
    if shared:
        log.warning(
            "Selection-rule collisions — labels below share an agreement code. "
            "Hidden trajectories for these labels will be IDENTICAL up to "
            "starting body pose. This is informative as a 'does the code carry "
            "this distinction?' control but limits per-label dynamics analysis. "
            "Try selection_rule=enrichment or re-fit KPMS to break ties."
        )
        for c, labs in shared.items():
            log.warning(f"  code {c}: {labs}")


# ---------------------------------------------------------------------------
# Termination-disabling
# ---------------------------------------------------------------------------


def _disable_far_termination(env_cfg) -> None:
    """High-threshold the termination criteria.

    Sustained-code rollouts of 1000 ctrl steps (~10 sec sim) regularly
    drift past the standard ``root_too_far=0.2`` and ``pose_error=6.0``
    thresholds — especially for codes that the policy was trained to
    track close to a reference.  Auto-reset on done would discontinue
    the hidden trajectory mid-rollout.

    We retain ``nan_termination`` as a safety net against actual numerical
    blow-ups.
    """
    if hasattr(env_cfg.termination_criteria, "root_too_far"):
        env_cfg.termination_criteria.root_too_far.max_distance = 1e6
    if hasattr(env_cfg.termination_criteria, "root_too_rotated"):
        env_cfg.termination_criteria.root_too_rotated.max_degrees = 1e6
    if hasattr(env_cfg.termination_criteria, "pose_error"):
        env_cfg.termination_criteria.pose_error.max_l2_error = 1e6


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_label_video(
    *,
    label: str,
    code: int,
    qpos: np.ndarray,        # (K, T, nq)
    codes: np.ndarray,       # (K, T)
    env,
    output_path: Path,
    num_video_bodies: int,
) -> None:
    """Per-label ghost-overlay verification clip."""
    n_show = min(num_video_bodies, qpos.shape[0])
    qpos = qpos[:n_show]
    codes = codes[:n_show]
    traj_colors = get_trajectory_colors(n_show)
    ghost_model, base_nq = build_ghost_model(
        env,
        num_ghosts=max(0, n_show - 1),
        ghost_colors=traj_colors[1:n_show],
        camera_distance=0.8,
        camera_elevation=-30.0,
        camera_azimuth=135.0,
    )
    try:
        render_ghost_video(
            ghost_model=ghost_model,
            base_nq=base_nq,
            trajectories_qpos=[q for q in qpos],
            code_sequences=[c for c in codes],
            trajectory_colors=traj_colors,
            output_path=output_path,
            title=f"{label} — code {code}, n={n_show}",
        )
        log.info(f"  Saved video: {output_path}")
    except Exception as e:
        log.warning(f"  Failed to render {label}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="labeled_hidden_dynamics_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Labeled-behavior RNN hidden state dynamics ===")

    output_dir = Path(_abs(cfg.output.base_dir))
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(cfg.checkpoint.path),
    )
    if not bool(ckpt_cfg.network_config.get("use_rnn_decoder", False)):
        raise ValueError("This experiment requires an RNN decoder checkpoint.")
    params = (norm_state, policy_params)

    # ------------------------------------------------------------------
    # Codes + balanced split
    # ------------------------------------------------------------------
    # IMPORTANT: build codes the same way the policy was trained,
    # dispatched on ``ckpt_cfg.kpms_config.code_source``. For
    # ``gt_directional_turn`` checkpoints (num_codes=5, labels-from-CSV)
    # there is no on-disk codes .npz — codes are computed from the
    # labels CSV at call time. For ``kpms_raw`` / ``kpms_merged`` this
    # reads the .npz path baked into the checkpoint config. This must
    # match training, otherwise the agreement-code histogram lives in
    # the wrong vocabulary and the sustained-code rollouts replay codes
    # the policy never saw.
    splits = load_balanced_splits(_abs(cfg.data.balanced_split_path))
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    clip_len = int(ckpt_cfg.env_config.clip_length)

    from c2a.codegen.code_sources import load_codes
    codes_payload = load_codes(ckpt_cfg, train_indices, test_indices, clip_len)
    train_codes = codes_payload.train_codes
    test_codes = codes_payload.test_codes
    num_codes = int(codes_payload.num_codes)
    log.info(
        f"Codes: source={codes_payload.code_source}, num_codes={num_codes}, "
        f"train={train_codes.shape}, test={test_codes.shape}"
    )

    # Train + test view used by the agreement-code histogram. Rows
    # correspond positionally to ``code_clip_indices``.
    all_codes = np.concatenate([train_codes, test_codes], axis=0)
    code_clip_indices = np.concatenate([train_indices, test_indices])

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    label_arrays = load_label_frames(
        _abs(cfg.data.labels_path),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    )
    log.info(
        f"Labels: {label_arrays.behavior.shape[0]} clips, "
        f"median {int(np.median(label_arrays.n_labeled_per_clip))} "
        f"combined-labeled frames/clip."
    )

    # ------------------------------------------------------------------
    # Agreement codes — across the whole labeled+coded intersection
    # (not restricted to test, since the code-label association is a
    # property of KPMS+labels, not of the policy).
    # ------------------------------------------------------------------
    label_clip_set = set(int(c) for c in label_arrays.clip_indices)
    code_clip_set = set(int(c) for c in code_clip_indices)
    intersect_for_agreement = label_clip_set & code_clip_set
    log.info(
        f"Agreement-code corpus: {len(intersect_for_agreement)} clips "
        f"(labeled ∩ coded)."
    )

    min_label_frames = int(cfg.get("min_label_frames", 100))
    selection_rule = str(cfg.get("selection_rule", "mode_excluding"))
    exclude_codes = tuple(int(c) for c in cfg.get("exclude_codes", [0]))
    min_within_label_frac = float(cfg.get("min_within_label_frac", 0.05))

    # 5-class directional-turn scheme: collapse direction for non-Turn
    # behaviors so the experiment reports Immobile / Rear / Walk /
    # Turn_Left / Turn_Right (mirrors c2a.codegen.code_sources.gt_directional_turn).
    label_names = DIRECTIONAL_TURN_LABELS
    label_array_5 = directional_turn_array(label_arrays)

    agreement = _find_agreement_codes(
        label_array=label_array_5,
        label_clip_indices=label_arrays.clip_indices,
        label_names=label_names,
        all_codes=all_codes,
        code_clip_indices=code_clip_indices,
        intersect_clips=intersect_for_agreement,
        num_codes=num_codes,
        min_label_frames=min_label_frames,
        selection_rule=selection_rule,
        exclude_codes=exclude_codes,
        min_within_label_frac=min_within_label_frac,
    )

    # Tally what we skipped for the printed table.
    skipped: list[tuple[str, int]] = []
    for label_int, label_str in enumerate(label_names):
        if label_str in agreement:
            continue
        mask_frames = (label_array_5 == label_int).sum()
        skipped.append((label_str, int(mask_frames)))
    _print_agreement_table(agreement, skipped, selection_rule, exclude_codes)

    if not agreement:
        raise RuntimeError(
            "No directional-turn label has enough labeled frames to compute "
            f"an agreement code (min_label_frames={min_label_frames}). "
            "Lower min_label_frames or check the labels CSV."
        )

    # ------------------------------------------------------------------
    # Build env on test clips with termination disabled
    # ------------------------------------------------------------------
    test_clips = ReferenceClips(
        data_path=_abs(cfg.data.reference_data_path),
        n_frames_per_clip=clip_len,
        keep_clips_idx=np.array(test_indices),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    if bool(cfg.get("disable_far_termination", True)):
        _disable_far_termination(env_cfg)
        log.info(
            "Termination criteria disabled "
            "(root_too_far / root_too_rotated / pose_error → 1e6); "
            "nan_termination retained."
        )
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    inf_fn = make_inference_fn(ppo_networks, use_rnn=True, deterministic=True)

    # ------------------------------------------------------------------
    # Pick K starting body positions ONCE (used for every label)
    # ------------------------------------------------------------------
    target_steps = int(cfg.target_steps)
    K_per_label = int(cfg.K_per_label)
    seed = int(cfg.seed)
    num_video_bodies = int(cfg.num_video_bodies)

    n_test_clips = test_clips.qpos.shape[0]
    if K_per_label > n_test_clips:
        log.warning(
            f"K_per_label={K_per_label} exceeds n_test_clips={n_test_clips}; "
            f"reducing K to n_test_clips."
        )
        K_per_label = n_test_clips

    rng_pos = np.random.RandomState(seed)
    chosen_positions = rng_pos.choice(
        n_test_clips, size=K_per_label, replace=False,
    )
    log.info(
        f"\nUsing K={K_per_label} test-clip positions for every label: "
        f"{chosen_positions.tolist()}"
    )
    log.info(
        f"Sustaining each label's agreement code for "
        f"{target_steps} ctrl steps "
        f"(= {target_steps / 100.0:.1f} sec at ctrl_dt=0.01)."
    )

    # ------------------------------------------------------------------
    # Per-label sustained-code rollouts
    # ------------------------------------------------------------------
    save_dict: dict[str, Any] = {}
    populated: list[tuple[str, int, int, float]] = []  # (label, code, K, agreement)

    for label_int, label_str in enumerate(label_names):
        if label_str not in agreement:
            continue
        info = agreement[label_str]
        code = info["code"]
        log.info(
            f"\n[{label_str}] sustaining code {code} "
            f"(agreement={info['agreement']:.3f}) for K={K_per_label} bodies"
        )

        code_seq = np.full(target_steps, code, dtype=np.int32)

        beh_hidden, beh_qpos, beh_proprio, beh_codes, beh_rewards = [], [], [], [], []
        for k, pos in enumerate(chosen_positions):
            key = jax.random.PRNGKey(seed + 1000 * label_int + k)
            result = run_rollout_with_hidden(
                env, inf_fn, params, ppo_networks, key,
                max_steps=target_steps,
                code_override=code_seq,
                reset_clip_idx=int(pos),
                jit_reset=jit_reset, jit_step=jit_step,
            )
            beh_hidden.append(result["hidden_states"])
            beh_qpos.append(result["qpos"])
            beh_proprio.append(result["proprio"])
            beh_codes.append(result["code_indices"])
            beh_rewards.append(result["rewards"])
            log.info(
                f"  body {k} (clip pos {int(pos)}): survival={result['survival']}/"
                f"{target_steps}, mean_reward={np.mean(result['rewards']):.2f}"
            )

        # Pad-or-truncate so the array stack is rectangular even if
        # nan_termination kicks in mid-rollout.
        def _rect(arrs, T):
            out = []
            for a in arrs:
                if a.shape[0] >= T:
                    out.append(a[:T])
                else:
                    pad_shape = (T - a.shape[0],) + a.shape[1:]
                    out.append(np.concatenate([a, np.zeros(pad_shape, dtype=a.dtype)]))
            return np.stack(out)

        T = target_steps
        save_dict[f"hidden_{label_str}"]  = _rect(beh_hidden, T)
        save_dict[f"qpos_{label_str}"]    = _rect(beh_qpos, T)
        save_dict[f"proprio_{label_str}"] = _rect(beh_proprio, T)
        save_dict[f"codes_{label_str}"]   = _rect(beh_codes, T)
        save_dict[f"rewards_{label_str}"] = _rect(beh_rewards, T)
        populated.append((label_str, code, K_per_label, info["agreement"]))

        # Render verification video for this label.
        if cfg.get("render_video", True):
            _render_label_video(
                label=label_str, code=code,
                qpos=save_dict[f"qpos_{label_str}"],
                codes=save_dict[f"codes_{label_str}"],
                env=env,
                output_path=data_dir / f"labeled_{label_str}.mp4",
                num_video_bodies=num_video_bodies,
            )

    # ------------------------------------------------------------------
    # Save npz
    # ------------------------------------------------------------------
    save_dict["agreement_codes"] = np.array(
        [(label, code, K, agree) for (label, code, K, agree) in populated],
        dtype=object,
    )
    save_dict["starting_positions"] = np.asarray(chosen_positions, dtype=np.int32)
    save_dict["target_steps"] = np.int32(target_steps)
    save_dict["K_per_label"] = np.int32(K_per_label)
    save_dict["num_codes"] = np.int32(num_codes)
    save_dict["min_label_frames"] = np.int32(min_label_frames)
    save_dict["seed"] = np.int32(seed)

    out_path = data_dir / "hidden_dynamics_labeled.npz"
    np.savez_compressed(out_path, **save_dict)
    log.info(f"\nSaved: {out_path}")

    # Mirror to figures/data/ for the plot scripts.
    fig_data_dir = MOSEQ_DIR / "figures" / "data"
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    dst = fig_data_dir / "hidden_dynamics_labeled.npz"
    shutil.copy2(out_path, dst)
    log.info(f"Mirrored: {dst}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n=== Summary ===")
    log.info(f"  populated labels: {len(populated)}")
    for label, code, K, agree in populated:
        log.info(
            f"    {label:<22s} ↔ code {code:>3d}  (agreement={agree:.3f}, K={K})"
        )


if __name__ == "__main__":
    main()
