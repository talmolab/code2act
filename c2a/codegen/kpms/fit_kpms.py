"""Single KPMS fit on keypoint data.

**CRITICAL**: This module sets ``jax_enable_x64 = True`` at import time,
which is process-global and irreversible.  It MUST run in a separate process
from the RL training (which requires float32 JAX).
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import keypoint_moseq as kpms

    HAS_KPMS = True
    KPMS_IMPORT_ERROR: Exception | None = None
except ImportError as _e:
    HAS_KPMS = False
    # Capture the real cause so a sweep can report it later instead of
    # the generic "not installed" message.  Common causes seen in the
    # wild: (1) lark / lark_parser version conflict, (2) jaxlib-cuda /
    # CUDA driver mismatch, (3) mujoco SO not loadable.
    KPMS_IMPORT_ERROR = _e

from c2a.codegen.kpms.config import KPMSHyperparams
from c2a.codegen.kpms.keypoint_loader import prepare_keypoints_for_kpms


@dataclass
class KPMSFitResult:
    """Result of a single KPMS fit.

    Attributes:
        model: Model checkpoint dict.
        results: Per-recording extraction results.
        data: Formatted data dict.
        metadata: Format metadata tuple.
        model_name: Name of this model.
        project_dir: Project directory path.
        labels_list: Per-recording syllable label arrays.
    """

    model: dict
    results: dict
    data: dict
    metadata: tuple
    model_name: str
    project_dir: str
    labels_list: list[np.ndarray]


# -----------------------------------------------------------------------------
# Manuscript setup (NeMo 2026). The defaults below are the Keypoint-MoSeq setup
# behind every KPMS result in the paper: tokenizer
# outputs/tokenizer/kpms_sweep/s100_k1e+02_l10_arhmm/seed4 (its saved
# config.yml records fix_heading=True, fps=30, anterior/posterior AnkleL/WristR)
# driving controller model_checkpoints/260611_073323_750085. Do not change them
# without re-running every KPMS experiment.
#
# fix_heading=True is upstream's head-fixed setting: Keypoint-MoSeq skips
# egocentric rotation and pins the heading state at zero, so the PCA latent
# keeps the rat's world heading and the anterior/posterior bodyparts are inert.
#
# Using fix_heading=True gives results similar to fix_heading=False
# (Snout -> TailBase body axis, fps=50, kappa=1e4 to match the 400 ms target
# duration). Compared on the fitted tokenizers:
#   - effective codes (of 100):  6.7 (fix_heading=True)  vs  8.5 (False)
#   - NMI vs behavior labels:    0.294                   vs  0.268
#   - NMI vs world yaw:          0.021                   vs  0.040
# The syllables carry little heading information in either setup, and both
# show the same usage concentration on a handful of codes that the paper
# reports. This comparison is at the tokenizer level; the paper's controller
# experiments were run only on the fix_heading=True tokenizer. Details:
# scratch/research/2026-08-09-kpms-fix-heading/. fix_heading=False is opt-in
# via the keyword arguments of fit_kpms_keypoints
# (c2a/configs/kpms_sweep_k100_fix_heading_false.yaml).
# -----------------------------------------------------------------------------
MANUSCRIPT_FIX_HEADING = True
MANUSCRIPT_FPS = 30


def _resolve_alignment_bodyparts(
    kp_names: list[str],
    anterior: str | None,
    posterior: str | None,
) -> tuple[list[str], list[str]]:
    """Resolve the egocentric-alignment axis.

    With both names ``None`` this reproduces the manuscript setup, which picks
    ``kp_names[0]`` / ``kp_names[-1]`` positionally (``AnkleL`` / ``WristR``
    for the alphabetical H5 order; inert under ``fix_heading=True``). Named
    bodyparts are validated rather than guessed.

    Args:
        kp_names: Available keypoint names.
        anterior: Anterior bodypart name, or ``None`` for the manuscript pick.
        posterior: Posterior bodypart name, or ``None`` for the manuscript pick.

    Returns:
        ``(anterior_bodyparts, posterior_bodyparts)`` as single-element lists.

    Raises:
        ValueError: If only one name is given, or a name is not in
            ``kp_names``.
    """
    if anterior is None and posterior is None:
        return [kp_names[0]], [kp_names[-1]]
    if anterior is None or posterior is None:
        raise ValueError(
            "Pass both anterior_bodypart and posterior_bodypart, or neither "
            "(manuscript setup)."
        )
    missing = [n for n in (anterior, posterior) if n not in kp_names]
    if missing:
        raise ValueError(
            f"Egocentric-alignment bodyparts {missing} are not in kp_names "
            f"({kp_names})."
        )
    return [anterior], [posterior]


def fit_kpms_keypoints(
    keypoint_data: np.ndarray,
    n_states: int,
    project_dir: str,
    hyperparams: KPMSHyperparams,
    seed: int,
    kp_names: list[str] | None = None,
    clear_project: bool = True,
    anterior_bodypart: str | None = None,
    posterior_bodypart: str | None = None,
    fps: int = MANUSCRIPT_FPS,
    fix_heading: bool = MANUSCRIPT_FIX_HEADING,
) -> KPMSFitResult:
    """Fit a KPMS model on keypoint trajectories.

    Args:
        keypoint_data: Shape ``[N, T, K, 3]``.
        n_states: Number of syllable states.
        project_dir: Output directory for this fit.
        hyperparams: KPMS hyperparameters.
        seed: Random seed.
        kp_names: Keypoint names (defaults to ``kp_0``, ``kp_1``, ...).
        clear_project: Remove existing project directory before fitting.
        anterior_bodypart: Anterior end of the alignment axis. ``None`` (with
            ``posterior_bodypart=None``) keeps the manuscript's positional pick.
        posterior_bodypart: Posterior end of the alignment axis.
        fps: Frame rate written to the KPMS project. Defaults to the
            manuscript value (30); the reference clips are 50 Hz.
        fix_heading: Skip egocentric alignment and pin the heading state at
            zero. Defaults to the manuscript value (``True``); see the
            manuscript-setup note above.

    Returns:
        KPMSFitResult with model, results, labels.

    Note:
        ``fix_heading=True`` makes Keypoint-MoSeq skip egocentric rotation
        (``jax_moseq`` ``alignment.preprocess_for_pca``) and stop resampling
        the heading state (``gibbs.resample_model``), so the PCA latent the
        AR-HMM segments retains the animal's world heading, and
        ``anterior_bodyparts`` / ``posterior_bodyparts`` become inert. On this
        corpus that left ~79% of latent variance as heading. Switching it off
        needs ~100x higher ``kappa`` to reach the 400 ms target duration.
    """
    if not HAS_KPMS:
        raise ImportError(
            f"keypoint_moseq not importable. Underlying cause: "
            f"{type(KPMS_IMPORT_ERROR).__name__}: {KPMS_IMPORT_ERROR}"
        )

    coordinates, confidences = prepare_keypoints_for_kpms(keypoint_data)

    if kp_names is None:
        n_keypoints = keypoint_data.shape[2]
        kp_names = [f"kp_{i}" for i in range(n_keypoints)]

    project_path = Path(project_dir)
    if clear_project and project_path.exists():
        shutil.rmtree(project_path)
    project_path.mkdir(parents=True, exist_ok=True)

    anterior_bodyparts, posterior_bodyparts = _resolve_alignment_bodyparts(
        kp_names, anterior_bodypart, posterior_bodypart
    )

    # Setup project config
    kpms.setup_project(
        project_dir,
        fix_heading=fix_heading,
        bodyparts=kp_names,
        skeleton=[],
        fps=fps,
        overwrite=True,
    )

    kpms.update_config(
        project_dir,
        fix_heading=fix_heading,
        bodyparts=kp_names,
        use_bodyparts=kp_names,
        anterior_bodyparts=anterior_bodyparts,
        posterior_bodyparts=posterior_bodyparts,
        verbose=False,
        latent_dim=hyperparams.latent_dim,
        num_states=n_states,
        kappa=hyperparams.kappa,
    )

    config = kpms.load_config(project_dir)
    data, metadata = kpms.format_data(coordinates, confidences, **config)

    model_name = f"keypoint_{hyperparams.model_type}_states{n_states}_seed{seed}"

    model = kpms.init_model(
        data=data,
        metadata=metadata,
        states=None,
        **config,
        seed=jax.random.PRNGKey(seed + n_states * 1000),
    )
    model = kpms.update_hypparams(model, kappa=hyperparams.kappa)

    # AR-only fitting
    model = kpms.fit_model(
        model=model,
        data=data,
        metadata=metadata,
        project_dir=project_dir,
        model_name=model_name,
        ar_only=True,
        num_iters=hyperparams.ar_iters,
        num_states=n_states,
        generate_progress_plots=False,
    )[0]

    # Full SLDS fitting (if requested)
    if hyperparams.model_type == "slds":
        model = kpms.update_hypparams(model, kappa=hyperparams.kappa)
        kpms.fit_model(
            model=model,
            data=data,
            metadata=metadata,
            project_dir=project_dir,
            model_name=model_name,
            ar_only=False,
            start_iter=hyperparams.ar_iters,
            num_iters=hyperparams.full_iters,
            generate_progress_plots=False,
        )

    # Post-process
    kpms.reindex_syllables_in_checkpoint(project_dir, model_name)
    model_ckpt, _, _, _ = kpms.load_checkpoint(project_dir, model_name)
    model_ckpt["project_dir"] = project_dir
    model_ckpt["model_name"] = model_name

    results = kpms.extract_results(model_ckpt, metadata, project_dir, model_name)

    rec_names = sorted(results.keys())
    labels_list = [results[rn]["syllable"] for rn in rec_names]

    return KPMSFitResult(
        model=model_ckpt,
        results=results,
        data=data,
        metadata=metadata,
        model_name=model_name,
        project_dir=project_dir,
        labels_list=labels_list,
    )
