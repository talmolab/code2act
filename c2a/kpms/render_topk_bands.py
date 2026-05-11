"""Post-hoc top-K syllable-band figure for a KPMS sweep.

Reads ``sweep_results.json``, loads each top-K model's ``results.h5``, and
renders a stacked band figure for each of three selected labeled clips:

* Row 0 — ground-truth human behavior bands, labeled with behavior names.
* Rows 1..K — each top-K model's syllable bands, colored by syllable ID.

Produces one tall figure per clip plus a combined 3-column side-by-side.
The shared figure/video plumbing (skeleton filmstrip, animated 3D
skeleton, behavior-band coloring) lives in
:mod:`c2a.render.topk_bands_render`.

Usage::

    python -m c2a.kpms.render_topk_bands --sweep-dir outputs/kpms_sweep
    python -m c2a.kpms.render_topk_bands --sweep-dir outputs/kpms_sweep \\
        --rank nmi --top-k 5
    python -m c2a.kpms.render_topk_bands --sweep-dir outputs/kpms_sweep \\
        --clips 700 720 735

Defaults: ``--top-k 5``, ``--rank mse``. Clip selection heuristic picks one
clip each dominated by Walking / Grooming / any Rear among the labeled
clips that intersect the fit set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from c2a.kpms.label_alignment import load_behavior_frames
from c2a.kpms.sweep_impl import RODENT_SKELETON, _load_keypoints
from c2a.paths import get_repo_root
from c2a.render.topk_bands_render import (
    auto_select_clips,
    render_clip_figure,
    render_clip_video,
    render_combined_figure,
    render_combined_video,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_balanced_split(balanced_split_path: Path) -> np.ndarray:
    """Return the sorted union of train+test balanced clip indices."""
    with balanced_split_path.open() as f:
        splits = json.load(f)
    all_idx = sorted(
        set(splits["balanced"]["train_indices"])
        | set(splits["balanced"]["test_indices"])
    )
    return np.array(all_idx, dtype=np.int32)


def _load_syllables_for_fit(project_dir: Path, model_name: str) -> list[np.ndarray]:
    """Load per-clip syllable arrays from a KPMS fit's ``results.h5``.

    Returns the syllable array for each recording, sorted by recording name
    (zero-padded so alphabetical == numerical order — matches the sort
    semantics in ``c2a/kpms/fit_kpms.py:162``).
    """
    results_h5 = project_dir / model_name / "results.h5"
    if not results_h5.exists():
        raise FileNotFoundError(f"Results HDF5 not found: {results_h5}")
    with h5py.File(results_h5, "r") as f:
        rec_names = sorted(f.keys())
        return [np.asarray(f[rn]["syllable"][:], dtype=np.int32) for rn in rec_names]


def _select_topk(
    all_results: list[dict],
    rank: str,
    top_k: int,
) -> list[dict]:
    """Pick top-K valid results by the requested ranking.

    Rank modes:
      * ``"mse"`` — sort ascending by ``reconstruction_mse`` (legacy
        default; doesn't measure behavioral interpretability).
      * ``"nmi"`` — sort descending by ``nmi_global`` (alignment with
        manual behavior labels, if computed).
      * ``"eml"`` — **Weinreb 2024** selection: filter to fits whose
        ``median_duration`` falls in the 400 ms ±30% window (4.2–7.8
        frames at 15 Hz), then rank by ``eml_score`` descending with
        ``syllable_usage_ratio`` as tiebreaker. Mirrors what
        :func:`c2a.kpms.sweep_impl._select_best_kpms` does for the
        single ``best_model`` pick. Falls back to the K fits whose
        ``median_duration`` is closest to target if the window is empty.
    """
    valid = [r for r in all_results if "error" not in r]
    if rank == "mse":
        valid.sort(key=lambda r: r["reconstruction_mse"])
        return valid[:top_k]
    if rank == "nmi":
        valid.sort(key=lambda r: -r.get("nmi_global", float("-inf")))
        return valid[:top_k]
    if rank == "eml":
        # Weinreb 2024 defaults — 400 ms median target, 15 Hz mocap, ±30%.
        target_frames = 400.0 * 15.0 / 1000.0  # = 6 frames
        tolerance_frac = 0.3
        lo, hi = target_frames * (1.0 - tolerance_frac), target_frames * (1.0 + tolerance_frac)

        def _dur(r: dict) -> float:
            if "median_duration" in r:
                return float(r["median_duration"])
            return float(r.get("mean_duration", 0.0))

        in_window = [r for r in valid if lo <= _dur(r) <= hi]
        if in_window:
            candidates = list(in_window)
        else:
            # No fit in window — fall back to closest-to-target.
            candidates = sorted(
                valid, key=lambda r: abs(_dur(r) - target_frames),
            )[: max(5, top_k)]
        candidates.sort(
            key=lambda r: (
                -float(r.get("eml_score", float("-inf"))),
                -float(r.get("syllable_usage_ratio", 0.0)),
            ),
        )
        return candidates[:top_k]
    raise ValueError(
        f"Unknown rank={rank!r}; choose 'mse', 'nmi', or 'eml'."
    )


def render(
    sweep_dir: Path,
    rank: str = "mse",
    top_k: int = 5,
    clips: list[int] | None = None,
    labels_path: Path | None = None,
    labeled_clip_range: tuple[int, int] = (642, 741),
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    n_frames: int = 250,
    n_snapshots: int = 8,
    output_format: str = "video",
    fps: int = 30,
) -> None:
    """Entry point that coordinates the whole rendering pipeline."""
    sweep_dir = Path(sweep_dir).resolve()
    results_path = sweep_dir / "sweep_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"sweep_results.json not found under {sweep_dir}")
    with results_path.open() as f:
        results = json.load(f)

    top = _select_topk(results["all_results"], rank=rank, top_k=top_k)
    if not top:
        raise RuntimeError("No valid fits in sweep_results.json — cannot render.")
    log.info(f"Top-{top_k} by {rank}:")
    for k, r in enumerate(top, 1):
        log.info(
            f"  M{k}: {r['setting']} seed={r['seed']} "
            f"MSE={r['reconstruction_mse']:.4f} "
            f"NMI={r.get('nmi_global', float('nan')):.3f}"
        )

    repo_root = get_repo_root()
    if labels_path is None:
        labels_path = repo_root / "data" / "rodent" / "rodent_labels.csv"
    if balanced_split_path is None:
        balanced_split_path = repo_root / "data" / "rodent" / "rodent_balanced_splits.json"
    if reference_data_path is None:
        reference_data_path = repo_root / "data" / "rodent" / "rodent_reference_clips.h5"
    if stac_xml_path is None:
        stac_xml_path = repo_root / "c2a" / "assets" / "rodent.xml"

    frame_labels, labeled_clip_indices, _ = load_behavior_frames(
        labels_path, clip_range=labeled_clip_range, n_frames=n_frames,
    )
    all_clip_indices = _load_balanced_split(balanced_split_path)
    intersect_mask = np.isin(labeled_clip_indices, all_clip_indices)
    log.info(
        f"Labeled clips: {len(labeled_clip_indices)}; intersect with fit set: "
        f"{int(intersect_mask.sum())}"
    )

    log.info("Loading keypoints via FK (one-time; ~30s)...")
    keypoints, kp_names = _load_keypoints(
        str(reference_data_path), str(stac_xml_path), str(balanced_split_path),
    )
    log.info(f"Keypoints loaded: {keypoints.shape}")

    if clips is not None:
        if len(clips) != 3:
            raise ValueError(f"--clips must have exactly 3 values, got {len(clips)}.")
        for ci in clips:
            if ci not in labeled_clip_indices or ci not in all_clip_indices:
                raise ValueError(
                    f"Clip {ci} is not in both the labeled range and the fit "
                    f"set (labeled={list(labeled_clip_indices[:5])}..., "
                    f"fit={list(all_clip_indices[:5])}...)."
                )
        selected_clips = list(clips)
    else:
        selected_clips = auto_select_clips(
            frame_labels, labeled_clip_indices, intersect_mask,
        )
    log.info(f"Rendering clips: {selected_clips}")

    lbl_pos = {int(c): i for i, c in enumerate(labeled_clip_indices)}
    fit_pos = {int(c): i for i, c in enumerate(all_clip_indices)}

    syllables_per_model: list[list[np.ndarray]] = []
    model_titles: list[str] = []
    model_n_syll: list[int] = []
    for k, r in enumerate(top, 1):
        project_dir = Path(r["project_dir"])
        if not project_dir.is_absolute():
            project_dir = repo_root / project_dir
        sylls = _load_syllables_for_fit(project_dir, r["model_name"])
        if not sylls:
            raise RuntimeError(
                f"M{k}: no recordings found in results.h5 at "
                f"{project_dir / r['model_name']}."
            )
        if len(sylls) != len(all_clip_indices):
            log.warning(
                f"M{k}: results.h5 has {len(sylls)} recordings but balanced "
                f"split has {len(all_clip_indices)} clips. Using the shorter."
            )
        syllables_per_model.append(sylls)
        short = r["setting"].replace("_l10_arhmm", "")
        model_titles.append(
            f"M{k} {short}/s{r['seed']}  NMI={r.get('nmi_global', float('nan')):.3f}"
        )
        n_states_fallback = max(1, max(int(np.max(s)) for s in sylls) + 1)
        model_n_syll.append(int(r.get("n_states", n_states_fallback)))

    output_dir = sweep_dir / "figures"

    gt_rows: list[np.ndarray] = []
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]] = []
    keypoint_trajs: list[np.ndarray] = []

    for ci in selected_clips:
        gt_row = frame_labels[lbl_pos[ci]]
        pos = fit_pos[ci]
        kp_traj = keypoints[pos]
        model_rows = [
            (model_titles[k], syllables_per_model[k][pos], model_n_syll[k])
            for k in range(len(top))
        ]
        if output_format in ("png", "both"):
            render_clip_figure(
                clip_idx=ci,
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                skeleton=RODENT_SKELETON,
                output_path=output_dir / f"topk_bands_{rank}_clip{ci}.png",
                n_frames=n_frames,
                n_snapshots=n_snapshots,
                title_prefix=f"top-{len(top)} KPMS models",
            )
        if output_format in ("video", "both"):
            render_clip_video(
                clip_idx=ci,
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                skeleton=RODENT_SKELETON,
                output_path=output_dir / f"topk_bands_{rank}_clip{ci}.mp4",
                n_frames=n_frames,
                fps=fps,
            )
        gt_rows.append(gt_row)
        model_rows_per_clip.append(model_rows)
        keypoint_trajs.append(kp_traj)

    if output_format in ("png", "both"):
        render_combined_figure(
            clip_indices=selected_clips,
            gt_rows=gt_rows,
            model_rows_per_clip=model_rows_per_clip,
            keypoint_trajs=keypoint_trajs,
            kp_names=kp_names,
            skeleton=RODENT_SKELETON,
            output_path=output_dir / f"topk_bands_{rank}_combined.png",
            n_frames=n_frames,
            n_snapshots=n_snapshots,
        )
    if output_format in ("video", "both"):
        render_combined_video(
            clip_indices=selected_clips,
            gt_rows=gt_rows,
            model_rows_per_clip=model_rows_per_clip,
            keypoint_trajs=keypoint_trajs,
            kp_names=kp_names,
            skeleton=RODENT_SKELETON,
            output_path=output_dir / f"topk_bands_{rank}_combined.mp4",
            n_frames=n_frames,
            fps=fps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument(
        "--rank", choices=("mse", "nmi"), default="mse",
        help="How to rank the top-K models (default: mse).",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--clips", type=int, nargs=3, default=None,
        help="Override auto-selected clips (3 ints, all in labeled ∩ fit).",
    )
    parser.add_argument(
        "--labels-path", type=Path, default=None,
        help="Path to rodent_labels.csv (default: <repo>/data/rodent/rodent_labels.csv).",
    )
    parser.add_argument(
        "--balanced-split", type=Path, default=None,
        help="Path to balanced splits JSON.",
    )
    parser.add_argument("--clip-range", type=int, nargs=2, default=[642, 741])
    parser.add_argument("--n-frames", type=int, default=250)
    parser.add_argument(
        "--reference-data", type=Path, default=None,
        help="Reference clips H5 (default: <repo>/data/rodent/rodent_reference_clips.h5).",
    )
    parser.add_argument(
        "--stac-xml", type=Path, default=None,
        help="stac-mjx rodent XML (default: <repo>/c2a/assets/rodent.xml).",
    )
    parser.add_argument(
        "--n-snapshots", type=int, default=8,
        help="Number of skeleton snapshots across the filmstrip row (PNG only).",
    )
    parser.add_argument(
        "--format", dest="output_format", choices=("video", "png", "both"),
        default="video",
        help=(
            "Output format. 'video' (default) = animated mp4 with skeleton + "
            "moving cursor. 'png' = static filmstrip + bands. 'both' = emit "
            "both for every clip."
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS.")
    args = parser.parse_args()

    render(
        sweep_dir=args.sweep_dir,
        rank=args.rank,
        top_k=args.top_k,
        clips=args.clips,
        labels_path=args.labels_path,
        labeled_clip_range=tuple(args.clip_range),
        balanced_split_path=args.balanced_split,
        reference_data_path=args.reference_data,
        stac_xml_path=args.stac_xml,
        n_frames=args.n_frames,
        n_snapshots=args.n_snapshots,
        output_format=args.output_format,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
