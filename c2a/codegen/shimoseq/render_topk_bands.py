"""Top-K band figures + mp4s for a shMoSeq sweep.

Mirror of :mod:`c2a.codegen.kpms.render_topk_bands`.  Loads top-K state arrays
from ``topk_states.npz`` (produced by
:func:`c2a.codegen.shimoseq.sweep_impl.run_sweep`), and uses the shared helpers
in :mod:`c2a.render.topk_bands_render` to emit:

  * ``topk_bands_loglik_clip{ci}.png`` — filmstrip + GT + K shMoSeq rows
  * ``topk_bands_loglik_clip{ci}.mp4`` — animated skeleton + bands + cursor
  * ``topk_bands_loglik_combined.{png,mp4}`` — 3-column variant

The "rank" axis is **marginal log-likelihood** (highest first), since
that's the per-fit ranking the shMoSeq paper recipe pairs with the
duration-window filter.

Usage::

    python -m c2a.codegen.shimoseq.render_topk_bands --sweep-dir outputs/shimoseq_sweep
    python -m c2a.codegen.shimoseq.render_topk_bands --sweep-dir outputs/shimoseq_sweep \\
        --top-k 5 --format both
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from c2a.codegen.shared.label_alignment import load_label_frames
from c2a.codegen.kpms.sweep_impl import RODENT_SKELETON, _load_keypoints
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


def _load_topk_states(
    topk_states_path: Path,
    top_k: int,
) -> tuple[list[dict[int, np.ndarray]], list[dict]]:
    """Load per-clip state arrays + metadata for the top-K shMoSeq fits.

    Returns:
        ``(states_per_model, meta_per_model)``.

        * ``states_per_model[m]`` is a dict ``{clip_id: state_seq}`` for
          model rank ``m+1`` (m=0 is rank 1).
        * ``meta_per_model[m]`` is the metadata dict for that fit.

    The npz schema is::

        m{r}__clip_{ci}: int (T,)         per-clip state sequence, rank r
        meta_m{r}      : str (1,)         JSON-encoded metadata
    """
    if not topk_states_path.exists():
        raise FileNotFoundError(
            f"topk_states.npz not at {topk_states_path}.  Re-run the shMoSeq "
            f"sweep — the new sweep_impl.py dumps top-K state arrays "
            f"alongside best_states.npz."
        )

    states_per_model: list[dict[int, np.ndarray]] = []
    meta_per_model: list[dict] = []
    with np.load(topk_states_path) as data:
        # Group keys by rank prefix (m{r}__clip_{ci}) and metadata.
        ranks_present = sorted({
            int(k.split("__", 1)[0][1:])
            for k in data.files
            if "__" in k and k.startswith("m")
        })
        for r in ranks_present[:top_k]:
            prefix = f"m{r}__"
            clips: dict[int, np.ndarray] = {}
            for k in data.files:
                if k.startswith(prefix):
                    suffix = k[len(prefix):]
                    if suffix.startswith("clip_"):
                        ci = int(suffix.split("_", 1)[1])
                        clips[ci] = np.asarray(data[k], dtype=np.int32)
            meta_key = f"meta_m{r}"
            if meta_key in data.files:
                meta = json.loads(str(np.asarray(data[meta_key]).item()))
            else:
                meta = {"rank": r}
            states_per_model.append(clips)
            meta_per_model.append(meta)
    return states_per_model, meta_per_model


def _short_label(meta: dict, rank: int) -> str:
    """Compact one-line label for the band row's left margin."""
    return (
        f"M{rank} n={meta.get('n_states', '?')} "
        f"k={meta.get('trans_kappa', float('nan')):.0e} "
        f"s{meta.get('seed', '?')}  "
        f"ll={meta.get('marginal_loglik', float('nan')):.0f} "
        f"med={meta.get('median_duration', float('nan')):.0f}f"
    )


def render(
    sweep_dir: Path,
    top_k: int = 5,
    clips: list[int] | None = None,
    labels_path: Path | None = None,
    labeled_clip_range: tuple[int, int] = (0, 841),
    csv_to_h5_subsample: int = 2,
    balanced_split_path: Path | None = None,
    reference_data_path: Path | None = None,
    stac_xml_path: Path | None = None,
    n_frames: int = 250,
    n_snapshots: int = 8,
    output_format: str = "video",
    fps: int = 30,
) -> None:
    """End-to-end orchestrator for shMoSeq top-K rendering."""
    sweep_dir = Path(sweep_dir).resolve()
    summary_path = sweep_dir / "sweep_results.json"
    topk_states_path = sweep_dir / "topk_states.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"sweep_results.json not found at {summary_path}")

    states_per_model, meta_per_model = _load_topk_states(topk_states_path, top_k)
    if not states_per_model:
        raise RuntimeError(
            f"No model state arrays found in {topk_states_path}."
        )
    actual_k = len(states_per_model)
    log.info(f"Top-{actual_k} shMoSeq fits (by marginal log-lik):")
    for r, m in enumerate(meta_per_model, 1):
        log.info(f"  M{r}: {_short_label(m, r)}")

    repo_root = get_repo_root()
    if labels_path is None:
        labels_path = repo_root / "data" / "rodent" / "rodent_labels.csv"
    if balanced_split_path is None:
        balanced_split_path = repo_root / "data" / "rodent" / "rodent_balanced_splits.json"
    if reference_data_path is None:
        reference_data_path = repo_root / "data" / "rodent" / "rodent_reference_clips.h5"
    if stac_xml_path is None:
        stac_xml_path = repo_root / "c2a" / "assets" / "rodent.xml"

    # Manual behavior labels
    label_arrays = load_label_frames(
        labels_path,
        clip_range=labeled_clip_range,
        n_frames=n_frames,
        csv_to_h5_subsample=csv_to_h5_subsample,
    )
    frame_labels = label_arrays.behavior  # [N_lbl, T]
    labeled_clip_indices = np.asarray(label_arrays.clip_indices, dtype=np.int32)

    # Balanced split (matches positional order of `keypoints`).
    all_clip_indices = _load_balanced_split(balanced_split_path)

    # We can only render clips that are (a) labeled, (b) in the balanced
    # split (so we have keypoints), and (c) covered by the shMoSeq fit
    # (so we have state arrays).  Take the triple-intersect.
    sh_clip_ids = set(states_per_model[0].keys())
    for s in states_per_model[1:]:
        sh_clip_ids &= set(s.keys())
    intersect_mask = np.array([
        bool((c in all_clip_indices) and (int(c) in sh_clip_ids))
        for c in labeled_clip_indices
    ])
    log.info(
        f"Labeled clips: {len(labeled_clip_indices)}; in fit set: "
        f"{int(np.isin(labeled_clip_indices, all_clip_indices).sum())}; "
        f"with shMoSeq states: {int(intersect_mask.sum())}"
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
            if int(ci) not in sh_clip_ids:
                raise ValueError(
                    f"Clip {ci} has no shMoSeq states.  Available: "
                    f"{sorted(sh_clip_ids)[:10]}..."
                )
            if ci not in all_clip_indices:
                raise ValueError(f"Clip {ci} not in balanced split.")
        selected_clips = list(clips)
    else:
        selected_clips = auto_select_clips(
            frame_labels, labeled_clip_indices, intersect_mask,
        )
    log.info(f"Rendering clips: {selected_clips}")

    lbl_pos = {int(c): i for i, c in enumerate(labeled_clip_indices)}
    fit_pos = {int(c): i for i, c in enumerate(all_clip_indices)}

    output_dir = sweep_dir / "figures"

    gt_rows: list[np.ndarray] = []
    model_rows_per_clip: list[list[tuple[str, np.ndarray, int]]] = []
    keypoint_trajs: list[np.ndarray] = []

    for ci in selected_clips:
        gt_row = frame_labels[lbl_pos[int(ci)]]
        kp_traj = keypoints[fit_pos[int(ci)]]
        model_rows: list[tuple[str, np.ndarray, int]] = []
        for r, (states, meta) in enumerate(zip(states_per_model, meta_per_model), 1):
            seq = states[int(ci)]
            n_cls = int(meta.get("n_states") or (int(seq.max()) + 1 if seq.size else 1))
            model_rows.append((_short_label(meta, r), seq, n_cls))
        if output_format in ("png", "both"):
            render_clip_figure(
                clip_idx=int(ci),
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                skeleton=RODENT_SKELETON,
                output_path=output_dir / f"topk_bands_loglik_clip{int(ci)}.png",
                n_frames=n_frames,
                n_snapshots=n_snapshots,
                title_prefix=f"top-{actual_k} shMoSeq fits",
            )
        if output_format in ("video", "both"):
            render_clip_video(
                clip_idx=int(ci),
                gt_row=gt_row,
                model_rows=model_rows,
                keypoint_traj=kp_traj,
                kp_names=kp_names,
                skeleton=RODENT_SKELETON,
                output_path=output_dir / f"topk_bands_loglik_clip{int(ci)}.mp4",
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
            output_path=output_dir / "topk_bands_loglik_combined.png",
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
            output_path=output_dir / "topk_bands_loglik_combined.mp4",
            n_frames=n_frames,
            fps=fps,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--clips", type=int, nargs=3, default=None,
        help="Override auto-selected clips (3 ints).",
    )
    parser.add_argument(
        "--labels-path", type=Path, default=None,
        help="Path to rodent_labels.csv.",
    )
    parser.add_argument(
        "--balanced-split", type=Path, default=None,
        help="Path to balanced splits JSON.",
    )
    parser.add_argument("--clip-range", type=int, nargs=2, default=[0, 841])
    parser.add_argument("--csv-to-h5-subsample", type=int, default=2)
    parser.add_argument("--n-frames", type=int, default=250)
    parser.add_argument(
        "--reference-data", type=Path, default=None,
        help="Reference clips H5.",
    )
    parser.add_argument(
        "--stac-xml", type=Path, default=None,
        help="stac-mjx rodent XML.",
    )
    parser.add_argument("--n-snapshots", type=int, default=8)
    parser.add_argument(
        "--format", dest="output_format", choices=("video", "png", "both"),
        default="video",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    render(
        sweep_dir=args.sweep_dir,
        top_k=args.top_k,
        clips=args.clips,
        labels_path=args.labels_path,
        labeled_clip_range=tuple(args.clip_range),
        csv_to_h5_subsample=args.csv_to_h5_subsample,
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
