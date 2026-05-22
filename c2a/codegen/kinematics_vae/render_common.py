"""Shared helpers for the kinematics-VAE family's top-K band rendering.

All three variants (VAE / VQVAE / PS-VQVAE) emit the same
``best_codes.npz`` schema (``all_codes``, ``train_indices``,
``num_states``, ...), so the orchestrator below works for any variant
that supplies its own ``model_rows`` factory.

The actual figure / video drawing is delegated to
:mod:`c2a.render.topk_bands_render`, which is the same module
:mod:`c2a.codegen.kpms.render_topk_bands` uses. That keeps the output
PNG/MP4 visually identical to kpms so figures are comparable across
tokenizer families.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from c2a.codegen.kpms.sweep_impl import RODENT_SKELETON, _load_keypoints
from c2a.codegen.shared.label_alignment import load_label_frames
from c2a.paths import get_repo_root
from c2a.render.topk_bands_render import (
    auto_select_clips,
    render_clip_figure,
    render_clip_video,
    render_combined_figure,
    render_combined_video,
)

log = logging.getLogger(__name__)


ModelRow = tuple[str, np.ndarray, int]
"""(short-label, per-frame code int seq, n_classes-for-colormap)."""

ModelRowsFactory = Callable[[np.ndarray, dict], list[ModelRow]]
"""Variant-specific: given per-clip codes (T,) and the codes-npz extras
dict, build the list of band rows for that clip (one band per row)."""


def _load_balanced_split(balanced_split_path: Path) -> np.ndarray:
    with balanced_split_path.open() as f:
        splits = json.load(f)
    all_idx = sorted(
        set(splits["balanced"]["train_indices"])
        | set(splits["balanced"]["test_indices"])
    )
    return np.array(all_idx, dtype=np.int32)


def _load_codes_npz(codes_npz_path: Path) -> dict:
    """Load the ``best_codes.npz`` written by any kinematics-VAE variant's
    ``tokenize`` step. Returns a plain dict with raw numpy arrays + scalars.
    Mirrors :data:`c2a.codegen.shared.npz_schema.CodesPayload` fields.
    """
    with np.load(codes_npz_path, allow_pickle=True) as data:
        out = {k: np.asarray(data[k]) for k in data.files}
    return out


def load_fk_keypoints(
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    balanced_split_path: Path | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """One-stop FK loader for variant render scripts.

    Returns ``(keypoints, kp_names, edges)`` for the balanced clips,
    where ``edges`` is the ``RODENT_SKELETON`` adjacency resolved
    against the actual keypoint name ordering. Useful when a caller
    wants to run multiple renderers (e.g. band figures + top-codes
    movement video) without paying the FK cost twice.
    """
    repo = get_repo_root()
    if reference_data_path is None:
        reference_data_path = repo / "data" / "rodent" / "rodent_reference_clips.h5"
    if stac_xml_path is None:
        stac_xml_path = repo / "c2a" / "assets" / "rodent.xml"
    if balanced_split_path is None:
        balanced_split_path = repo / "data" / "rodent" / "rodent_balanced_splits.json"

    from c2a.render.topk_bands_render import _resolve_skeleton_edges
    keypoints, kp_names = _load_keypoints(
        str(reference_data_path), str(stac_xml_path), str(balanced_split_path),
    )
    edges = _resolve_skeleton_edges(kp_names, RODENT_SKELETON)
    return keypoints, kp_names, edges


def render_topk_bands(
    *,
    codes_npz_path: Path,
    model_rows_factory: ModelRowsFactory,
    variant_label: str,
    output_dir: Path,
    clips: list[int] | None = None,
    labels_path: Path | None = None,
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    labeled_clip_range: tuple[int, int] = (0, 841),
    csv_to_h5_subsample: int = 2,
    n_frames: int = 250,
    n_snapshots: int = 8,
    output_format: str = "both",
    fps: int = 30,
    keypoints: np.ndarray | None = None,
    kp_names: list[str] | None = None,
) -> None:
    """End-to-end orchestrator for kinematics-VAE-family band rendering.

    Args:
        codes_npz_path: Path to the variant's ``best_codes.npz`` (sidecar
            of the variant's training output).
        model_rows_factory: Function mapping ``(per_clip_codes, npz_extras)``
            → list of ``ModelRow`` tuples. Each variant builds its own
            band rows (single-band for VAE/VQVAE; multi-band for PS-VQVAE).
        variant_label: short string for the output filename prefix
            (e.g. ``"kmeans"``, ``"vqvae"``, ``"ps_vqvae"``).
        output_dir: Where to drop the figures (typically next to the
            ``best_codes.npz``).
        clips: Override auto-selected clips with 3 explicit ids.
        labels_path / balanced_split_path / reference_data_path /
            stac_xml_path: Override the standard data locations.
        labeled_clip_range / csv_to_h5_subsample / n_frames: GT-label
            decode knobs (mirror ``load_label_frames``).
        n_snapshots / fps: Filmstrip / MP4 cosmetics.
        output_format: ``"video"`` | ``"png"`` | ``"both"``.
    """
    repo = get_repo_root()
    if labels_path is None:
        labels_path = repo / "data" / "rodent" / "rodent_labels.csv"
    if balanced_split_path is None:
        balanced_split_path = repo / "data" / "rodent" / "rodent_balanced_splits.json"
    if reference_data_path is None:
        reference_data_path = repo / "data" / "rodent" / "rodent_reference_clips.h5"
    if stac_xml_path is None:
        stac_xml_path = repo / "c2a" / "assets" / "rodent.xml"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load codes (sorted-union order over balanced clips).
    extras = _load_codes_npz(codes_npz_path)
    all_codes = extras["all_codes"]                # (N_bal, T) int32
    train_idx = np.asarray(extras["train_indices"], dtype=np.int32)
    test_idx = np.asarray(extras["test_indices"], dtype=np.int32)
    bal_idx = np.array(sorted(set(train_idx.tolist()) | set(test_idx.tolist())),
                       dtype=np.int32)
    if all_codes.shape[0] != bal_idx.size:
        raise RuntimeError(
            f"all_codes rows ({all_codes.shape[0]}) != balanced clips "
            f"({bal_idx.size}). Re-run tokenize."
        )
    log.info(
        f"Loaded codes: variant={extras['model_type']} K={int(extras['num_states'])} "
        f"shape={all_codes.shape}"
    )

    # 2. GT labels for clip-selection + GT band row.
    label_arrays = load_label_frames(
        labels_path, clip_range=labeled_clip_range,
        n_frames=n_frames, csv_to_h5_subsample=csv_to_h5_subsample,
    )
    frame_labels = label_arrays.behavior            # (N_lbl, T) — 4-class
    labeled_clip_indices = np.asarray(label_arrays.clip_indices, dtype=np.int32)

    # 3. Restrict to clips that are (a) labeled and (b) in the balanced
    # split (so we have codes for them).
    bal_set = set(bal_idx.tolist())
    intersect_mask = np.array(
        [int(c) in bal_set for c in labeled_clip_indices], dtype=bool,
    )
    log.info(
        f"Labeled clips: {len(labeled_clip_indices)}; in balanced split: "
        f"{int(intersect_mask.sum())}"
    )

    # 4. Keypoints via forward kinematics (one-time cost, ~30s) — skip
    # if the caller already loaded them (e.g. when also running the
    # top-codes-movement renderer right after).
    if keypoints is None or kp_names is None:
        log.info("Loading keypoints via FK (one-time; ~30s)...")
        keypoints, kp_names = _load_keypoints(
            str(reference_data_path), str(stac_xml_path),
            str(balanced_split_path),
        )
        log.info(f"Keypoints loaded: {keypoints.shape}")

    # 5. Select the 3 demo clips.
    if clips is not None:
        if len(clips) != 3:
            raise ValueError(f"--clips must be 3 ints, got {len(clips)}")
        for ci in clips:
            if int(ci) not in bal_set:
                raise ValueError(
                    f"Clip {ci} not in balanced split (size {len(bal_set)})."
                )
        selected_clips = list(clips)
    else:
        selected_clips = auto_select_clips(
            frame_labels, labeled_clip_indices, intersect_mask,
        )
    log.info(f"Rendering clips: {selected_clips}")

    lbl_pos = {int(c): i for i, c in enumerate(labeled_clip_indices)}
    bal_pos = {int(c): i for i, c in enumerate(bal_idx)}

    # 6. Render per-clip + combined.
    gt_rows: list[np.ndarray] = []
    model_rows_per_clip: list[list[ModelRow]] = []
    keypoint_trajs: list[np.ndarray] = []

    for ci in selected_clips:
        gt_row = frame_labels[lbl_pos[int(ci)]]
        kp_traj = keypoints[bal_pos[int(ci)]]
        clip_codes = all_codes[bal_pos[int(ci)]]
        model_rows = model_rows_factory(clip_codes, extras)

        if output_format in ("png", "both"):
            render_clip_figure(
                clip_idx=int(ci),
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                skeleton=RODENT_SKELETON,
                output_path=(
                    output_dir / f"topk_bands_{variant_label}_clip{int(ci)}.png"
                ),
                n_frames=n_frames,
                n_snapshots=n_snapshots,
                title_prefix=f"{variant_label} — kinematics-VAE family",
            )
        if output_format in ("video", "both"):
            render_clip_video(
                clip_idx=int(ci),
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                skeleton=RODENT_SKELETON,
                output_path=(
                    output_dir / f"topk_bands_{variant_label}_clip{int(ci)}.mp4"
                ),
                n_frames=n_frames,
                fps=fps,
            )
        gt_rows.append(gt_row)
        model_rows_per_clip.append(model_rows)
        keypoint_trajs.append(kp_traj)

    if output_format in ("png", "both"):
        render_combined_figure(
            clip_indices=[int(c) for c in selected_clips],
            gt_rows=gt_rows,
            model_rows_per_clip=model_rows_per_clip,
            keypoint_trajs=keypoint_trajs,
            kp_names=kp_names,
            skeleton=RODENT_SKELETON,
            output_path=output_dir / f"topk_bands_{variant_label}_combined.png",
            n_frames=n_frames,
            n_snapshots=n_snapshots,
        )
    if output_format in ("video", "both"):
        render_combined_video(
            clip_indices=[int(c) for c in selected_clips],
            gt_rows=gt_rows,
            model_rows_per_clip=model_rows_per_clip,
            keypoint_trajs=keypoint_trajs,
            kp_names=kp_names,
            skeleton=RODENT_SKELETON,
            output_path=output_dir / f"topk_bands_{variant_label}_combined.mp4",
            n_frames=n_frames,
            fps=fps,
        )
    log.info(f"Done. Wrote outputs to {output_dir}")
