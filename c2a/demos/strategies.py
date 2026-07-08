"""Per-steering-strategy configuration for the presentation demo videos.

Single source of truth shared by every ``c2a/demos`` script so a strategy is
added in exactly one place. ``mlb`` is wired but its data paths are ``None``
until the mlb checkpoint exists (the demo scripts skip / TBD-placeholder it).

All paths are relative to the repo root (resolve with
``c2a.paths.get_repo_root()``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyDemoConfig:
    """Everything a demo needs for one steering strategy.

    Attributes:
        name: short tag (``kpms`` / ``kvae`` / ``mlb``).
        display: human-readable label for banners/titles.
        policy_ckpt: a MoSeqPPONetwork checkpoint dir (used only to build the
            env / ``mj_model``; any in-distribution ckpt for the strategy works).
        perturbation_dir: ``outputs/perturbation/perturbation_recovery/<run>``
            (kick→recovery rollouts; ``rollout_code<cid>_amp<amp>.npz``).
        parade_dir: a behavior-parade rollout dir with
            ``data/parade_rollouts.npz`` + ``code_selection.json``.
        rollouts_dir: ``outputs/code_grid/code_grid_rollouts/<run>`` (held-code
            rollouts + ``labels/code_labels.json`` for the code atlas).
        dynamics_dir: ``outputs/code_grid/code_grid_dynamics/<run>`` (hidden-state
            trajectories for the within-syllable dynamics demo).
    """

    name: str
    display: str
    policy_ckpt: str | None
    perturbation_dir: str | None
    parade_dir: str | None
    rollouts_dir: str | None
    dynamics_dir: str | None


STRATEGIES: dict[str, StrategyDemoConfig] = {
    "kpms": StrategyDemoConfig(
        name="kpms",
        display="Keypoint-MoSeq (kpms)",
        policy_ckpt="model_checkpoints/260407_031233_484020",
        perturbation_dir="outputs/perturbation/perturbation_recovery/kpms",
        parade_dir="outputs/behavior/cand_kpms_parade2/walk_groom_rear",
        rollouts_dir="outputs/code_grid/code_grid_rollouts/kpms",
        dynamics_dir="outputs/code_grid/code_grid_dynamics/kpms",
    ),
    "kvae": StrategyDemoConfig(
        name="kvae",
        display="Kinematics-VAE (kvae)",
        policy_ckpt="model_checkpoints/260603_180852_140475",
        perturbation_dir="outputs/perturbation/perturbation_recovery/best_vae",
        parade_dir="outputs/behavior/moseq_behavior_parade/walk_groom_rear",
        rollouts_dir="outputs/code_grid/code_grid_rollouts/best_vae",
        dynamics_dir="outputs/code_grid/code_grid_dynamics/best_vae",
    ),
    # mlb: the manually-labeled controller. Checkpoint exists / trains at
    # 260629_212713_669229; the data dirs below are the paths the Phase-B
    # pipeline writes (run_name=mlb). They do not exist until the pipeline runs,
    # so `available()` (existence-checked) keeps mlb out of the demos until then.
    "mlb": StrategyDemoConfig(
        name="mlb",
        display="Manually labeled (mlb)",
        policy_ckpt="model_checkpoints/260629_212713_669229",
        perturbation_dir="outputs/perturbation/perturbation_recovery/mlb",
        parade_dir="outputs/behavior/mlb_parade/walk_groom_rear",
        rollouts_dir="outputs/code_grid/code_grid_rollouts/mlb",
        dynamics_dir="outputs/code_grid/code_grid_dynamics/mlb",
    ),
}

# Canonical demo order (rows / panels); mlb last so it lands as a TBD slot.
STRATEGY_ORDER: list[str] = ["kpms", "kvae", "mlb"]


def available(field: str) -> list[str]:
    """Strategies whose ``field`` path is set AND exists on disk.

    Existence-checked so mlb (whose paths are pre-wired above) only enters a
    demo once the Phase-B pipeline has actually produced that demo's data.
    """
    from c2a.paths import get_repo_root

    root = get_repo_root()
    out = []
    for s in STRATEGY_ORDER:
        p = getattr(STRATEGIES[s], field)
        if p is not None and (root / p).exists():
            out.append(s)
    return out
