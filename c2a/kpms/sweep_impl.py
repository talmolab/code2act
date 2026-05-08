"""KPMS hyperparameter grid search (paper-faithful selection).

Fits KPMS models across a grid of (num_states, kappa, latent_dim, model_type)
with multiple seeds.  Selection follows **Weinreb et al. 2024 (Nature Methods)**:

    "We adjusted κ separately for each dataset to ensure a median state
     duration of 400 ms."  ...and across runs/seeds: "we developed a
     likelihood-based metric [Expected Marginal Likelihood] that allows
     post hoc ranking of model runs ... lowest for outlier models, highest
     for those that are consensus-like."

Concretely, ``_select_best_kpms`` picks the (κ × num_states) cell whose
median state duration falls nearest the configured target (400 ms by
default), then within that cell ranks seeds by EML.  The previous
selection criterion in this file ("lowest reconstruction MSE → highest
EML → highest usage ratio") was **not** what the KPMS paper actually
recommends — MSE is reported as a diagnostic but no longer used for
selection.

This module is invoked indirectly via ``c2a.sweep.run_sweep`` (the
dispatcher).  The dispatcher reads the YAML config's ``model: kpms``
field and routes here.

**CRITICAL**: this module sets ``jax_enable_x64 = True`` at import time,
which is process-global and irreversible. It MUST run in a separate
process from RL training (which uses float32 JAX).

Usage::

    python -m c2a.sweep.run_sweep --config c2a/configs/kpms_sweep.yaml
"""

import argparse
import itertools
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


from c2a.paths import get_c2a_root, get_repo_root
import matplotlib.pyplot as plt
import numpy as np

# Enable x64 before any JAX import
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

# Add repo root to path
MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()
import yaml

from c2a.kpms.config import KPMSHyperparams
from c2a.kpms.fit_kpms import fit_kpms_keypoints
from c2a.kpms.label_alignment import (
    BEHAVIORS,
    COMBINED_LABELS,
    DIRECTIONS,
    load_label_frames,
)
from c2a.kpms.coverage_metrics import compute_coverage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_config(config_path: str | None = None) -> dict:
    """Load sweep config from YAML and resolve ``${oc.env:...}`` expressions.

    Uses OmegaConf so that ``${oc.env:C2A_ROOT,.}`` is substituted from the
    environment at load time (the bash scripts export ``C2A_ROOT``). Returns
    a plain ``dict`` so downstream code can continue using ``.get()`` etc.
    """
    from omegaconf import OmegaConf

    if config_path is None:
        config_path = str(MOSEQ_DIR / "configs" / "kpms_sweep.yaml")
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


# =============================================================================
# Paper-faithful selection (Weinreb 2024, Methods)
# =============================================================================


def _select_best_kpms(
    valid_results: list[dict],
    target_duration_ms: float = 400.0,
    mocap_hz: float = 15.0,
    tolerance_frac: float = 0.3,
) -> tuple[dict, list[dict]]:
    """Select the best KPMS fit per the Weinreb 2024 recipe.

    Steps:
      1. Convert ``target_duration_ms`` to a frame count via ``mocap_hz``.
      2. Filter fits whose ``mean_duration`` (in frames) lies within
         ``[target × (1-tol), target × (1+tol)]``.
      3. If no fit clears the window, fall back to the 5 fits closest
         to the target (so we never silently return nothing).
      4. Rank survivors by EML (highest = consensus-like) with usage
         ratio as a tie-breaker.  ``reconstruction_mse`` is **not** used
         as a selection key — the KPMS paper does not recommend it.

    Args:
        valid_results: per-fit metric dicts (must contain ``mean_duration``;
            ``eml_score`` and ``syllable_usage_ratio`` optional).
        target_duration_ms: median state duration target.  Weinreb 2024
            says 400 ms.
        mocap_hz: data sample rate.  At our H5 rate (250 frames per
            16.67-sec clip) this is 15 Hz.
        tolerance_frac: ±fraction around target for the duration window.

    Returns:
        ``(best_fit_dict, candidates_in_window)``.
    """
    target_frames = target_duration_ms * mocap_hz / 1000.0
    lo = target_frames * (1.0 - tolerance_frac)
    hi = target_frames * (1.0 + tolerance_frac)

    def _dur(r: dict) -> float:
        """Paper criterion: prefer median_duration; fall back to mean for
        results from older sweeps that didn't compute the median."""
        if "median_duration" in r:
            return float(r["median_duration"])
        return float(r.get("mean_duration", 0))

    in_window = [r for r in valid_results if lo <= _dur(r) <= hi]

    if in_window:
        candidates = list(in_window)
    else:
        candidates = sorted(
            valid_results,
            key=lambda r: abs(_dur(r) - target_frames),
        )[:5]
        log.warning(
            f"No fit landed in duration window [{lo:.1f}, {hi:.1f}] frames "
            f"(target {target_frames:.1f} = {target_duration_ms} ms @ "
            f"{mocap_hz} Hz). Falling back to 5 closest-to-target fits."
        )

    candidates.sort(
        key=lambda r: (
            -float(r.get("eml_score", float("-inf"))),
            -float(r.get("syllable_usage_ratio", 0.0)),
        ),
    )
    return candidates[0], candidates


def _setup_stac_model(
    h5_path: str,
    xml_path: str,
) -> tuple["mujoco.MjModel", "mujoco.MjData", list[int], list[str]]:
    """Set up MuJoCo model with keypoint sites from stac-mjx config.

    Replicates what ``stac_mjx.Stac._build_body_spec()`` does:

    1. Load the base XML and add keypoint sites to the correct bodies.
    2. Override site positions with optimized offsets from the H5 file.

    This avoids a dependency on ``stac_mjx`` while producing identical
    FK output.

    Args:
        h5_path: Path to reference clips H5 (must contain ``config``,
            ``offsets``, ``kp_names``).
        xml_path: Path to the stac-mjx rodent XML model.

    Returns:
        ``(mj_model, mj_data, site_ids, kp_names)`` where ``site_ids``
        and ``kp_names`` are both in the H5's alphabetical order.
    """
    import h5py
    import mujoco
    import yaml

    with h5py.File(h5_path, "r") as f:
        cfg_yaml = f["config"][()].decode()
        offsets = f["offsets"][:]  # [K, 3] in kp_names order
        kp_names = [n.decode() for n in f["kp_names"][:]]

    cfg = yaml.safe_load(cfg_yaml)
    kmp = cfg["model"]["KEYPOINT_MODEL_PAIRS"]

    # Build model with keypoint sites using MjSpec
    spec = mujoco.MjSpec.from_file(xml_path)
    name_to_offset = {name: offsets[i] for i, name in enumerate(kp_names)}

    for kp_name, body_name in kmp.items():
        parent = spec.body(body_name)
        pos = name_to_offset[kp_name].tolist()
        parent.add_site(
            name=kp_name,
            size=[0.005, 0.005, 0.005],
            pos=pos,
            group=3,
        )

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)

    # Get site IDs in kp_names (alphabetical) order
    site_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in kp_names
    ]

    return mj_model, mj_data, site_ids, kp_names


def _qpos_to_keypoints_fk(
    qpos: np.ndarray,
    mj_model: "mujoco.MjModel",
    mj_data: "mujoco.MjData",
    site_ids: list[int],
) -> np.ndarray:
    """Compute keypoints from qpos via MuJoCo forward kinematics.

    Uses batched JAX-vmapped FK for speed, matching the reference
    implementation in ``TopoVNL/topo_mimic/benchmark/hyperparam_sweep/
    keypoint_loader.py``.

    Args:
        qpos: Joint positions ``[N_total_frames, nq]``.
        mj_model: Compiled MuJoCo model with keypoint sites.
        mj_data: MuJoCo data template.
        site_ids: Site indices for keypoints.

    Returns:
        Keypoint positions ``[N_total_frames, K, 3]``.
    """
    import jax.numpy as jnp
    import mujoco.mjx as mjx

    site_ids_array = jnp.array(site_ids)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    @jax.jit
    def forward_kinematics(qpos_single):
        data = mjx_data.replace(qpos=qpos_single)
        data = mjx.forward(mjx_model, data)
        return data.site_xpos[site_ids_array]

    # Process in batches to avoid OOM
    batch_fk = jax.vmap(forward_kinematics)
    n_total = qpos.shape[0]
    batch_size = 1000
    all_kps = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_qpos = jnp.array(qpos[start:end])
        batch_kps = batch_fk(batch_qpos)
        all_kps.append(np.array(batch_kps))

    return np.concatenate(all_kps, axis=0)


def _load_keypoints(
    data_path: str,
    stac_xml_path: str,
    balanced_split_path: str | None = None,
    n_frames_per_clip: int = 250,
) -> tuple[np.ndarray, list[str]]:
    """Load qpos from H5 and convert to keypoints via FK.

    Matches the reference implementation in ``TopoVNL/topo_mimic/benchmark/
    hyperparam_sweep/kpms/kpms_sweep.py::load_qpos_and_convert_keypoints``:

    1. Load ``qpos`` from the reference clips H5.
    2. Set up a MuJoCo model with keypoint sites (replicating stac-mjx).
    3. Run forward kinematics to get keypoint positions ``[N, T, K, 3]``.
    4. Optionally filter to balanced clips.

    Returns:
        ``(keypoints, kp_names)`` where keypoints has shape ``[N, T, K, 3]``
        with columns in the H5's kp_names order.
    """
    import h5py

    with h5py.File(data_path, "r") as f:
        qpos = f["qpos"][:]  # [N_total_frames, nq]

    log.info(f"Loaded qpos: {qpos.shape}")

    # Set up model with keypoint sites
    mj_model, mj_data, site_ids, kp_names = _setup_stac_model(data_path, stac_xml_path)
    log.info(f"Set up model with {len(site_ids)} keypoint sites")

    # FK: qpos → keypoints
    log.info("Running forward kinematics...")
    kp_3d = _qpos_to_keypoints_fk(qpos, mj_model, mj_data, site_ids)
    log.info(f"FK output: {kp_3d.shape}")

    # Reshape from [N_total_frames, K, 3] to [N_clips, T, K, 3]
    n_total = kp_3d.shape[0]
    n_clips = n_total // n_frames_per_clip
    keypoints = kp_3d[: n_clips * n_frames_per_clip].reshape(
        n_clips, n_frames_per_clip, *kp_3d.shape[1:]
    )

    if balanced_split_path is not None:
        bsp = Path(balanced_split_path)
        if not bsp.is_absolute():
            bsp = get_repo_root() / bsp
        if not bsp.exists():
            raise FileNotFoundError(
                f"data.balanced_split_path={balanced_split_path} does not "
                f"resolve (tried {bsp}). Set to null to sweep on all clips."
            )
        with open(bsp) as f:
            splits = json.load(f)
        train_idx = splits["balanced"]["train_indices"]
        test_idx = splits["balanced"]["test_indices"]
        all_idx = sorted(set(train_idx) | set(test_idx))
        keypoints = keypoints[all_idx]
        log.info(f"Using {len(all_idx)} balanced clips (train+test)")

    return keypoints, kp_names


def _reconstruct_keypoints(
    fit_result,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract original and reconstructed keypoints from KPMS fit.

    Uses ``data["Y"]`` from ``format_data`` as original and
    ``estimate_coordinates`` + ``unbatch`` for reconstruction, matching
    the reference implementation in ``TopoVNL/topo_mimic/benchmark/
    hyperparam_sweep/kpms/kpms_sweep.py``.

    Both outputs are flattened to ``[N, T, K*D]`` for per-dimension
    comparison.

    Args:
        fit_result: KPMSFitResult from fitting.

    Returns:
        ``(original, reconstructed)`` both as ``[N, T, K*D]``.
    """
    import jax.numpy as jnp
    from jax_moseq.models.keypoint_slds.alignment import estimate_coordinates
    from jax_moseq.utils import unbatch

    # Original: data["Y"] from format_data [n_segs, seg_length, K, D]
    Y_data = fit_result.data["Y"]
    if isinstance(Y_data, dict):
        keys = sorted(Y_data.keys())
        original = np.stack([np.array(Y_data[k]) for k in keys], axis=0)
    else:
        original = np.array(Y_data)

    if original.ndim == 4:
        n, t, k, d = original.shape
        original = original.reshape(n, t, k * d)

    # Reconstruction: estimate_coordinates + unbatch
    model = fit_result.model
    Y_est = estimate_coordinates(
        jnp.array(model["states"]["x"]),
        jnp.array(model["states"]["v"]),
        jnp.array(model["states"]["h"]),
        jnp.array(model["params"]["Cd"]),
    )
    coords_dict = unbatch(np.array(Y_est), *fit_result.metadata)
    rec_keys = sorted(coords_dict.keys())
    recon_list = []
    for rk in rec_keys:
        arr = np.array(coords_dict[rk])
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.ndim == 4:
            arr = arr.reshape(arr.shape[0], -1)
        recon_list.append(arr)
    reconstructed = np.stack(recon_list, axis=0)
    if reconstructed.ndim == 4:
        n, t, k, d = reconstructed.shape
        reconstructed = reconstructed.reshape(n, t, k * d)

    return original, reconstructed


def _plot_keypoint_sample(
    Y: np.ndarray,
    kp_names: list[str],
    clip_idx: int = 0,
    n_keypoints: int = 6,
) -> plt.Figure:
    """Plot raw keypoint trajectories (X/Y/Z) for one clip.

    Args:
        Y: Keypoints ``[T, K, 3]``.
        kp_names: Keypoint names.
        clip_idx: Clip index (for title).
        n_keypoints: How many keypoints to show.
    """
    kp_indices = np.linspace(0, len(kp_names) - 1, n_keypoints, dtype=int)
    dims = ["X", "Y", "Z"]

    fig, axes = plt.subplots(n_keypoints, 3, figsize=(14, 2.5 * n_keypoints))
    if n_keypoints == 1:
        axes = axes[np.newaxis, :]

    for row, ki in enumerate(kp_indices):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            ax.plot(Y[:, ki, col], linewidth=1)
            if row == 0:
                ax.set_title(dim)
            if col == 0:
                ax.set_ylabel(kp_names[ki], fontsize=8)
            if row == n_keypoints - 1:
                ax.set_xlabel("Frame")
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Clip {clip_idx}: Input Keypoints", fontsize=11)
    fig.tight_layout()
    return fig


def _plot_reconstruction(
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
    clip_idx: int = 0,
    n_dims: int = 5,
    title_prefix: str = "Reconstruction",
    max_frames: int = 500,
) -> plt.Figure:
    """Plot original vs reconstructed timeseries for flattened dimensions.

    Matches the reference ``plot_reconstruction_timeseries`` from
    ``TopoVNL/topo_mimic/benchmark/hyperparam_sweep/visualization.py``.

    Args:
        original_data: Original data ``[N, T, K*D]`` (flattened dims).
        reconstructed_data: Reconstructed data ``[N, T, K*D]``.
        clip_idx: Which clip to plot.
        n_dims: Number of flattened dimensions to show.
        title_prefix: Title prefix string.
        max_frames: Maximum frames to show.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    n_frames = min(original_data.shape[1], reconstructed_data.shape[1], max_frames)
    t = np.arange(n_frames)

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]

        orig = original_data[clip_idx, :n_frames, dim_idx]
        recon = reconstructed_data[clip_idx, :n_frames, dim_idx]

        ax.plot(t, orig, label="Original", color="blue", alpha=0.7, linewidth=1)
        ax.plot(t, recon, label="Reconstructed", color="red", alpha=0.7, linewidth=1)

        mse = np.mean((orig - recon) ** 2)
        ax.set_ylabel(f"Dim {dim_idx}")
        ax.set_title(f"Dimension {dim_idx} (MSE: {mse:.4f})", fontsize=10)
        if dim_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"{title_prefix} - Clip {clip_idx}", fontsize=12)
    plt.tight_layout()
    return fig


RODENT_SKELETON = [
    ("Snout", "SpineF"),
    ("SpineF", "SpineM"),
    ("SpineM", "SpineL"),
    ("SpineL", "TailBase"),
    ("SpineF", "ShoulderL"),
    ("ShoulderL", "ElbowL"),
    ("ElbowL", "WristL"),
    ("WristL", "HandL"),
    ("SpineF", "ShoulderR"),
    ("ShoulderR", "ElbowR"),
    ("ElbowR", "WristR"),
    ("WristR", "HandR"),
    ("SpineL", "HipL"),
    ("HipL", "KneeL"),
    ("KneeL", "AnkleL"),
    ("AnkleL", "FootL"),
    ("SpineL", "HipR"),
    ("HipR", "KneeR"),
    ("KneeR", "AnkleR"),
    ("AnkleR", "FootR"),
    ("Snout", "EarL"),
    ("Snout", "EarR"),
]


def _render_keypoint_video(
    Y: np.ndarray,
    output_path: str,
    kp_names: list[str],
    Y_bar: np.ndarray | None = None,
    codes: np.ndarray | None = None,
    fps: int = 30,
    title: str = "Keypoint 3D",
    elev: float = 20,
    azim: float = -60,
) -> str:
    """Render 3D keypoint skeleton animation as mp4.

    Draws keypoints as dots connected by skeleton edges (``RODENT_SKELETON``).
    Optionally overlays reconstructed keypoints in red with syllable labels.

    Adapted from the reference implementation in
    ``TopoVNL/topo_mimic/benchmark/hyperparam_sweep/visualization.py``.

    Args:
        Y: Original keypoints ``[T, K, 3]``.
        output_path: Path to save mp4 video.
        kp_names: Keypoint names (used to resolve skeleton edges).
        Y_bar: Reconstructed keypoints ``[T, K, 3]`` (optional overlay).
        codes: Syllable labels ``[T]`` (shown in frame text).
        fps: Frames per second for output video.
        title: Video title.
        elev: 3D view elevation angle.
        azim: 3D view azimuth angle.

    Returns:
        The output path.
    """
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Point matplotlib to imageio_ffmpeg's bundled ffmpeg binary
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    T, K, _ = Y.shape

    # Build skeleton edge index pairs from keypoint names
    name_to_idx = {n: i for i, n in enumerate(kp_names)}
    edges = []
    for a, b in RODENT_SKELETON:
        if a in name_to_idx and b in name_to_idx:
            edges.append((name_to_idx[a], name_to_idx[b]))

    # Compute axis limits (equal range, centered)
    all_pts = Y if Y_bar is None else np.concatenate([Y, Y_bar], axis=0)
    mins = all_pts.reshape(-1, 3).min(0)
    maxs = all_pts.reshape(-1, 3).max(0)
    rng = float(np.max(maxs - mins))
    ctr = (mins + maxs) / 2
    pad = 0.05 * rng if rng > 0 else 1e-3

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")

    # Initial frame: original keypoints + skeleton
    (pts,) = ax.plot(Y[0, :, 0], Y[0, :, 1], Y[0, :, 2], "o", ms=4, color="blue")
    lines = [
        ax.plot(
            [Y[0, i, 0], Y[0, j, 0]],
            [Y[0, i, 1], Y[0, j, 1]],
            [Y[0, i, 2], Y[0, j, 2]],
            "-",
            color="gray",
            linewidth=2,
        )[0]
        for i, j in edges
    ]

    # Optional: reconstructed overlay
    pts_recon = None
    lines_recon = []
    if Y_bar is not None:
        (pts_recon,) = ax.plot(
            Y_bar[0, :, 0],
            Y_bar[0, :, 1],
            Y_bar[0, :, 2],
            "o",
            ms=3,
            color="red",
        )
        lines_recon = [
            ax.plot(
                [Y_bar[0, i, 0], Y_bar[0, j, 0]],
                [Y_bar[0, i, 1], Y_bar[0, j, 1]],
                [Y_bar[0, i, 2], Y_bar[0, j, 2]],
                "-",
                color="salmon",
                linewidth=1,
                alpha=0.6,
            )[0]
            for i, j in edges
        ]

    ax.set_xlim(ctr[0] - rng / 2 - pad, ctr[0] + rng / 2 + pad)
    ax.set_ylim(ctr[1] - rng / 2 - pad, ctr[1] + rng / 2 + pad)
    ax.set_zlim(ctr[2] - rng / 2 - pad, ctr[2] + rng / 2 + pad)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)

    frame_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, fontsize=10)

    def update(t):
        pts.set_data(Y[t, :, 0], Y[t, :, 1])
        pts.set_3d_properties(Y[t, :, 2])
        for ln, (i, j) in zip(lines, edges):
            ln.set_data([Y[t, i, 0], Y[t, j, 0]], [Y[t, i, 1], Y[t, j, 1]])
            ln.set_3d_properties([Y[t, i, 2], Y[t, j, 2]])

        if pts_recon is not None:
            pts_recon.set_data(Y_bar[t, :, 0], Y_bar[t, :, 1])
            pts_recon.set_3d_properties(Y_bar[t, :, 2])
            for ln, (i, j) in zip(lines_recon, edges):
                ln.set_data(
                    [Y_bar[t, i, 0], Y_bar[t, j, 0]],
                    [Y_bar[t, i, 1], Y_bar[t, j, 1]],
                )
                ln.set_3d_properties([Y_bar[t, i, 2], Y_bar[t, j, 2]])

        label = f"Frame {t}/{T}"
        if codes is not None and t < len(codes):
            label += f"  code={codes[t]}"
        frame_text.set_text(label)
        return [pts, *lines, frame_text]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    log.info(f"Saved 3D keypoint video: {output_path}")
    return output_path


def _compute_metrics(
    fit_result,
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_states: int,
    label_arrays=None,
    all_clip_indices: np.ndarray | None = None,
) -> dict:
    """Compute reconstruction, syllable-quality, and (optional) behavior-coverage metrics.

    Args:
        fit_result: KPMSFitResult from fitting.
        original: Flattened original data ``[N, T, K*D]``.
        reconstructed: Flattened reconstructed data ``[N, T, K*D]``.
        n_states: Number of syllable states.
        label_arrays: Optional ``LabelArrays`` from
            :func:`c2a.kpms.label_alignment.load_label_frames` containing
            per-clip frame labels at three granularities (behavior,
            direction, combined).  Pass together with ``all_clip_indices``
            to enable coverage metrics (NMI + Hungarian purity).
        all_clip_indices: Clip indices passed to KPMS in positional order
            (i.e. matches ``fit_result.labels_list``).

    Returns:
        Dict with reconstruction_mse, syllable stats, transition entropy,
        and (if all coverage-args provided) coverage at three granularities
        keyed as ``coverage_<level>_<metric>`` for level ∈ {behavior,
        direction, combined}.
    """
    labels = fit_result.labels_list

    min_len = min(original.shape[1], reconstructed.shape[1])
    mse = float(np.mean((original[:, :min_len] - reconstructed[:, :min_len]) ** 2))

    # Duration stats
    all_durations = []
    for lbl in labels:
        changes = np.where(np.diff(lbl) != 0)[0] + 1
        segments = np.split(lbl, changes)
        all_durations.extend(len(s) for s in segments)
    durations = np.array(all_durations) if all_durations else np.array([1])

    # Active syllables
    all_labels = np.concatenate(labels) if labels else np.array([0])
    active = len(np.unique(all_labels))

    # Transition entropy
    n = n_states
    trans_matrix = np.zeros((n, n))
    for lbl in labels:
        for i in range(len(lbl) - 1):
            trans_matrix[lbl[i], lbl[i + 1]] += 1
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    trans_probs = trans_matrix / row_sums
    entropy = -np.nansum(trans_probs * np.log(trans_probs + 1e-10), axis=1)
    mean_entropy = float(np.mean(entropy[row_sums.squeeze() > 0]))

    result: dict = {
        "reconstruction_mse": mse,
        "active_syllables": active,
        "syllable_usage_ratio": active / n_states,
        # Per the Weinreb 2024 recipe the *median* state duration is the
        # selection criterion.  Mean is reported alongside as a diagnostic
        # — for HMM-like geometric dwell distributions, mean ≈ 1.44 ×
        # median, so they're not interchangeable.
        "mean_duration": float(np.mean(durations)),
        "median_duration": float(np.median(durations)),
        "std_duration": float(np.std(durations)),
        "transition_entropy": mean_entropy,
    }

    # Optional behavior-coverage metrics (NMI + Hungarian purity) at three
    # granularities.  All three are reported; the caller selects which to
    # use for downstream model-selection.
    if label_arrays is not None and all_clip_indices is not None:
        levels = (
            ("behavior",  label_arrays.behavior,  len(BEHAVIORS)),
            ("direction", label_arrays.direction, len(DIRECTIONS)),
            ("combined",  label_arrays.combined,  len(COMBINED_LABELS)),
        )
        cov: dict = {}
        for level_name, level_frames, level_n_classes in levels:
            level_cov = compute_coverage(
                syllables_list=fit_result.labels_list,
                frame_labels=level_frames,
                labeled_clip_indices=label_arrays.clip_indices,
                all_clip_indices=all_clip_indices,
                n_syll=n_states,
                n_beh=int(level_n_classes),
            )
            # Re-key with the level prefix so all three coexist.
            for k, v in level_cov.items():
                cov[f"coverage_{level_name}_{k}"] = v
        # Convenience alias: keep legacy keys (no prefix) pointing to behavior
        # level so downstream parsers that read 'nmi_global' / 'purity_global'
        # still work.
        for legacy_k in (
            "nmi_global", "purity_global", "n_labeled_clips_used",
            "n_labeled_frames_used", "coverage_mapping",
        ):
            src = f"coverage_behavior_{legacy_k}"
            if src in cov:
                cov[legacy_k] = cov[src]
        # ``cov`` is empty when no labeled clip intersects the fit set.
        result.update(cov)

    return result


def _save_sweep_summary_plots(
    setting_seeds: dict[str, list[dict]],
    output_dir: Path,
    target_duration_ms: float = 400.0,
    mocap_hz: float = 15.0,
    tolerance_frac: float = 0.3,
) -> None:
    """Save summary plots (usage ratio, transition entropy, scatter,
    duration-vs-(K, κ)) as PNGs.

    The duration plot marks the paper's target window (``[lo, hi]`` frames
    around ``target_duration_ms``) so cells that pass the duration filter
    are visible at a glance.
    """
    plots_dir = output_dir / "summary_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per-setting means
    settings = []
    usage_means = []
    entropy_means = []
    median_durs = []
    mean_durs = []
    kappas = []
    n_states_list = []
    for setting_key, seeds_results in setting_seeds.items():
        settings.append(setting_key)
        usage_means.append(np.mean([r["syllable_usage_ratio"] for r in seeds_results]))
        entropy_means.append(np.mean([r["transition_entropy"] for r in seeds_results]))
        median_durs.append(np.mean([
            r.get("median_duration", r.get("mean_duration", 0)) for r in seeds_results
        ]))
        mean_durs.append(np.mean([r.get("mean_duration", 0) for r in seeds_results]))
        # Pull (kappa, K) for the heatmap.
        if seeds_results:
            kappas.append(float(seeds_results[0].get("kappa", float("nan"))))
            n_states_list.append(int(seeds_results[0].get("n_states", -1)))
        else:
            kappas.append(float("nan")); n_states_list.append(-1)

    # Bar chart: usage ratio per setting
    fig, ax = plt.subplots(figsize=(max(8, len(settings) * 0.5), 5))
    ax.bar(range(len(settings)), usage_means)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, rotation=90, fontsize=6)
    ax.set_ylabel("Usage Ratio")
    ax.set_title("Syllable Usage Ratio per Setting")
    fig.tight_layout()
    fig.savefig(plots_dir / "usage_ratio_bar.png", dpi=150)
    plt.close(fig)

    # Bar chart: transition entropy per setting
    fig, ax = plt.subplots(figsize=(max(8, len(settings) * 0.5), 5))
    ax.bar(range(len(settings)), entropy_means)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, rotation=90, fontsize=6)
    ax.set_ylabel("Transition Entropy")
    ax.set_title("Transition Entropy per Setting")
    fig.tight_layout()
    fig.savefig(plots_dir / "transition_entropy_bar.png", dpi=150)
    plt.close(fig)

    # Scatter: usage ratio vs transition entropy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(usage_means, entropy_means, s=40)
    for i, s in enumerate(settings):
        ax.annotate(s, (usage_means[i], entropy_means[i]), fontsize=5)
    ax.set_xlabel("Usage Ratio")
    ax.set_ylabel("Transition Entropy")
    ax.set_title("Usage Ratio vs Transition Entropy")
    fig.tight_layout()
    fig.savefig(plots_dir / "usage_vs_entropy.png", dpi=150)
    plt.close(fig)

    # Duration-target heatmap: median (and mean) state duration vs (κ, K)
    # with the paper's [lo, hi] window highlighted.
    target_frames = target_duration_ms * mocap_hz / 1000.0
    lo_frames = target_frames * (1.0 - tolerance_frac)
    hi_frames = target_frames * (1.0 + tolerance_frac)
    if any(not np.isnan(k) for k in kappas) and any(n > 0 for n in n_states_list):
        kappa_uniq = sorted(set(k for k in kappas if not np.isnan(k)))
        n_uniq = sorted(set(n for n in n_states_list if n > 0))
        med_grid = np.full((len(n_uniq), len(kappa_uniq)), np.nan)
        for k_v, n_v, md in zip(kappas, n_states_list, median_durs):
            if np.isnan(k_v) or n_v <= 0:
                continue
            r = n_uniq.index(n_v); c = kappa_uniq.index(k_v)
            med_grid[r, c] = md
        fig, ax = plt.subplots(figsize=(max(6, len(kappa_uniq) * 0.85),
                                         max(3, len(n_uniq) * 0.85)))
        # Log-color scale because durations span > 2 orders of magnitude.
        from matplotlib.colors import LogNorm
        valid = ~np.isnan(med_grid)
        if valid.any():
            vmin = max(np.nanmin(med_grid[valid]), 1e-2)
            vmax = max(np.nanmax(med_grid[valid]), vmin * 1.01)
            im = ax.imshow(med_grid, aspect="auto", cmap="viridis",
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        for i in range(med_grid.shape[0]):
            for j in range(med_grid.shape[1]):
                v = med_grid[i, j]
                if np.isnan(v):
                    continue
                in_window = lo_frames <= v <= hi_frames
                color = "white" if v > target_frames else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color=color,
                        fontweight="bold" if in_window else "normal")
                if in_window:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.45, i - 0.45), 0.9, 0.9,
                        fill=False, edgecolor="#e74c3c", lw=2,
                    ))
        ax.set_xticks(range(len(kappa_uniq)))
        ax.set_xticklabels([f"{k:.0e}" for k in kappa_uniq], rotation=45, fontsize=8)
        ax.set_yticks(range(len(n_uniq)))
        ax.set_yticklabels([str(n) for n in n_uniq], fontsize=8)
        ax.set_xlabel(r"$\kappa$ (transition stickiness)")
        ax.set_ylabel("num_states")
        ax.set_title(
            f"Median state duration (frames) vs ($\\kappa$, K). "
            f"Red boxes: in target window [{lo_frames:.1f}, {hi_frames:.1f}]f "
            f"≈ {target_duration_ms:.0f} ms ± {int(tolerance_frac*100)}% @ {mocap_hz} Hz"
        )
        if valid.any():
            fig.colorbar(im, ax=ax, label="median duration (frames, log)")
        fig.tight_layout()
        fig.savefig(plots_dir / "duration_vs_kappa_K.png", dpi=150)
        plt.close(fig)

    log.info(f"Saved sweep summary plots to {plots_dir}")


def run_sweep(cfg: dict) -> dict:
    """Run the full KPMS grid search.

    Args:
        cfg: Sweep configuration dict.

    Returns:
        Dict with ``"best_model"`` info and ``"all_results"`` list.
    """
    sweep_cfg = cfg["sweep"]
    output_dir = Path(cfg["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load qpos and convert to keypoints via FK (matching the reference)
    keypoints, kp_names = _load_keypoints(
        cfg["data"]["reference_data_path"],
        cfg["data"]["stac_xml_path"],
        cfg["data"].get("balanced_split_path"),
    )
    log.info(f"Loaded keypoints: {keypoints.shape}, {len(kp_names)} keypoints")

    # Derive the positional clip-index array for coverage metrics. KPMS fits
    # will produce ``fit_result.labels_list`` in the same order as
    # ``keypoints``, which is the balanced-split sorted order (see
    # ``_load_keypoints``) or the raw H5 order if no split is configured.
    balanced_split_path = cfg["data"].get("balanced_split_path")
    if balanced_split_path:
        _bsp = Path(balanced_split_path)
        if not _bsp.is_absolute():
            _bsp = get_repo_root() / _bsp
        with open(_bsp) as f:
            _splits = json.load(f)
        all_clip_indices = np.array(
            sorted(
                set(_splits["balanced"]["train_indices"])
                | set(_splits["balanced"]["test_indices"])
            ),
            dtype=np.int32,
        )
    else:
        all_clip_indices = np.arange(keypoints.shape[0], dtype=np.int32)

    # Load human behavior labels (optional). When present, every fit will
    # report NMI + Hungarian purity at THREE granularities (4-class behavior,
    # 3-class direction, 12-class behavior×direction) alongside the existing
    # MSE-based metrics. Model selection stays on MSE — coverage is reported,
    # not selected on.
    label_arrays = None  # type: c2a.kpms.label_alignment.LabelArrays | None
    labels_path = cfg["data"].get("labels_path")
    if labels_path:
        _lp = Path(labels_path)
        if not _lp.is_absolute():
            _lp = get_repo_root() / _lp
        if not _lp.exists():
            raise FileNotFoundError(
                f"data.labels_path={labels_path} does not resolve to an "
                f"existing file (tried {_lp}). Set labels_path=null to skip "
                "coverage metrics."
            )
        # Default range covers all 842 clips in the new schema; override
        # via labeled_clip_range in the YAML if you want a sub-range.
        _clip_range = tuple(cfg["data"].get("labeled_clip_range", [0, 841]))
        _subsample = int(cfg["data"].get("csv_to_h5_subsample", 2))
        label_arrays = load_label_frames(
            _lp, clip_range=_clip_range, n_frames=keypoints.shape[1],
            csv_to_h5_subsample=_subsample,
        )
        n_intersect = int(
            np.isin(label_arrays.clip_indices, all_clip_indices).sum()
        )
        log.info(
            f"Coverage metrics enabled (3 granularities): "
            f"{n_intersect}/{len(label_arrays.clip_indices)} labeled clips "
            f"in the fit set; "
            f"median combined-labeled frames/clip = "
            f"{int(np.median(label_arrays.n_labeled_per_clip))}."
        )

    # Save an input keypoint sample figure to disk (one-time, diagnostic).
    try:
        fig = _plot_keypoint_sample(keypoints[0], kp_names, clip_idx=0)
        fig.savefig(output_dir / "input_keypoint_sample.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        log.warning(f"Failed to save input keypoint sample: {e}")

    # Grid
    grid = list(
        itertools.product(
            sweep_cfg["num_states"],
            sweep_cfg["kappa"],
            sweep_cfg["latent_dim"],
            sweep_cfg["model_type"],
        )
    )
    seeds = list(range(sweep_cfg["seeds_per_setting"]))
    log.info(
        f"Grid: {len(grid)} settings × {len(seeds)} seeds = {len(grid) * len(seeds)} fits"
    )

    all_results = []

    for gi, (n_states, kappa, latent_dim, model_type) in enumerate(grid):
        setting_key = f"s{n_states}_k{kappa:.0e}_l{latent_dim}_{model_type}"
        setting_dir = output_dir / setting_key
        setting_results = []

        for seed in seeds:
            hp = KPMSHyperparams(
                kappa=kappa,
                latent_dim=latent_dim,
                num_states=n_states,
                ar_iters=sweep_cfg["ar_iters"],
                full_iters=sweep_cfg["full_iters"],
                model_type=model_type,
            )

            project_dir = str(setting_dir / f"seed{seed}")
            log.info(f"[{gi + 1}/{len(grid)}] {setting_key} seed={seed}")

            try:
                fit_result = fit_kpms_keypoints(
                    keypoint_data=keypoints,
                    n_states=n_states,
                    project_dir=project_dir,
                    hyperparams=hp,
                    seed=seed,
                    kp_names=kp_names,
                )
                original, reconstructed = _reconstruct_keypoints(fit_result)
                metrics = _compute_metrics(
                    fit_result, original, reconstructed, n_states,
                    label_arrays=label_arrays,
                    all_clip_indices=all_clip_indices,
                )

                result = {
                    "setting": setting_key,
                    "n_states": n_states,
                    "kappa": kappa,
                    "latent_dim": latent_dim,
                    "model_type": model_type,
                    "seed": seed,
                    "project_dir": project_dir,
                    "model_name": fit_result.model_name,
                    **metrics,
                }
                setting_results.append(result)
                all_results.append(result)

                # Save reconstruction timeseries PNG to the seed's project_dir
                # (same location KPMS writes its own artifacts). Cheap, useful
                # diagnostic if a fit looks off.
                try:
                    fig = _plot_reconstruction(
                        original,
                        reconstructed,
                        clip_idx=0,
                        title_prefix=f"{setting_key} seed={seed}",
                    )
                    fig.savefig(
                        Path(project_dir) / "reconstruction.png", dpi=150,
                    )
                    plt.close(fig)
                except Exception as ve:
                    log.warning(f"  Failed to save reconstruction plot: {ve}")

                coverage_str = ""
                if "coverage_behavior_nmi_global" in metrics:
                    coverage_str = (
                        f", NMI[beh]={metrics['coverage_behavior_nmi_global']:.3f}/"
                        f"[dir]={metrics.get('coverage_direction_nmi_global', float('nan')):.3f}/"
                        f"[12]={metrics.get('coverage_combined_nmi_global', float('nan')):.3f}, "
                        f"purity[beh]={metrics['coverage_behavior_purity_global']:.3f}"
                    )
                log.info(
                    f"  MSE={metrics['reconstruction_mse']:.4f}, "
                    f"active={metrics['active_syllables']}/{n_states}, "
                    f"dur={metrics['mean_duration']:.1f}±{metrics['std_duration']:.1f}, "
                    f"H={metrics['transition_entropy']:.2f}"
                    f"{coverage_str}"
                )

            except Exception as e:
                log.warning(f"  FAILED: {e}")
                all_results.append(
                    {
                        "setting": setting_key,
                        "seed": seed,
                        "error": str(e),
                    }
                )

    # Compute EML per setting (requires ≥2 seeds), then select best
    valid = [r for r in all_results if "error" not in r]

    setting_seeds = defaultdict(list)
    for r in valid:
        setting_seeds[r["setting"]].append(r)

    for setting_key, seeds_results in setting_seeds.items():
        if len(seeds_results) >= 2:
            checkpoint_paths = [
                os.path.join(r["project_dir"], r["model_name"], "checkpoint.h5")
                for r in seeds_results
            ]
            try:
                from keypoint_moseq.fitting import expected_marginal_likelihoods

                scores, std_errors = expected_marginal_likelihoods(
                    checkpoint_paths=checkpoint_paths
                )
                for i, r in enumerate(seeds_results):
                    r["eml_score"] = float(scores[i])
                    r["eml_stderr"] = float(std_errors[i])
                log.info(
                    f"  EML for {setting_key}: "
                    f"scores={[f'{s:.2f}' for s in scores]}"
                )
            except Exception as e:
                log.warning(f"  EML computation failed for {setting_key}: {e}")

    # Paper-faithful selection: median state duration close to target,
    # then EML rank within that window. See _select_best_kpms docstring.
    sel_cfg = cfg.get("selection", {}) or {}
    target_duration_ms = float(sel_cfg.get("target_duration_ms", 400.0))
    mocap_hz = float(sel_cfg.get("mocap_hz", 15.0))
    tolerance_frac = float(sel_cfg.get("tolerance_frac", 0.3))

    if valid:
        best, candidates_in_window = _select_best_kpms(
            valid,
            target_duration_ms=target_duration_ms,
            mocap_hz=mocap_hz,
            tolerance_frac=tolerance_frac,
        )
        target_frames = target_duration_ms * mocap_hz / 1000.0
        log.info(
            f"\nBest model: {best['setting']} seed={best['seed']}, "
            f"dur={best['mean_duration']:.2f} frames "
            f"(target {target_frames:.2f} = {target_duration_ms} ms @ {mocap_hz} Hz), "
            f"EML={best.get('eml_score', 'N/A')}, "
            f"usage={best['syllable_usage_ratio']:.2f}, "
            f"MSE={best.get('reconstruction_mse', float('nan')):.4f}"
        )
        log.info(
            f"  ({len(candidates_in_window)}/{len(valid)} models inside the "
            f"duration window ±{tolerance_frac:.0%} of target)"
        )
    else:
        best = None
        log.warning("No successful fits!")

    # Summary plots to disk.
    if valid:
        try:
            _save_sweep_summary_plots(
                setting_seeds, output_dir,
                target_duration_ms=target_duration_ms,
                mocap_hz=mocap_hz,
                tolerance_frac=tolerance_frac,
            )
        except Exception as e:
            log.warning(f"Failed to create summary plots: {e}")

    # Save results
    summary = {"best_model": best, "all_results": all_results}
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved results to {results_path}")

    # Auto-render per-syllable example grids for the top-K fits.  The
    # renderer is also a standalone CLI (``python -m
    # c2a.kpms.render_syllable_grids --sweep-dir ...``) — use that to
    # re-render with different ``--bout-strategy`` or ``--top-k`` without
    # rerunning the sweep.
    auto_render_cfg = (cfg.get("auto_render") or {}) if isinstance(cfg, dict) else {}
    if auto_render_cfg.get("syllable_grids", True):
        try:
            from c2a.kpms.render_syllable_grids import render as _render_grids
            _render_grids(
                sweep_dir=output_dir,
                rank=auto_render_cfg.get("rank", "mse"),
                top_k=int(auto_render_cfg.get("top_k", 5)),
                n_examples=int(auto_render_cfg.get("n_examples", 5)),
                max_rows_per_page=int(auto_render_cfg.get("max_rows_per_page", 5)),
                bout_strategy=auto_render_cfg.get("bout_strategy", "longest"),
                min_bout_duration=int(auto_render_cfg.get("min_bout_duration", 3)),
                output_format=auto_render_cfg.get("format", "both"),
                fps=int(auto_render_cfg.get("fps", 30)),
            )
        except Exception as e:
            log.warning(f"Syllable-grid auto-render failed: {e}")

    return summary


# Public alias: the dispatcher (c2a.sweep.run_sweep) imports `run_sweep`.
# Already defined above as `def run_sweep(cfg: dict) -> dict:`.


def main():
    """Direct CLI entry point.

    Prefer ``python -m c2a.sweep.run_sweep --config ...`` (the dispatcher),
    which routes to this module when ``model: kpms`` in the YAML. This
    direct entry point exists for backward compatibility with bash
    scripts that called ``python -m c2a.sweep.run_sweep`` before the
    dispatcher refactor.
    """
    parser = argparse.ArgumentParser(description="KPMS hyperparameter sweep")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
