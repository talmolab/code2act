"""MLB segmenter: manual behavior labels + kinematic direction -> 9 flat codes.

Builds the ``mlb`` (Manually Labeled Behaviors) steering-strategy code stream in
the same ``CodesPayload`` npz format the training pipeline expects. Each frame's
code is its manually annotated behavior (Immobile / Rear / Turn / Walk) crossed
with a Left / Right / Straight direction derived from the *reference body's* net
heading change over the enclosing behavior bout:

* For each contiguous behavior bout, the net yaw (unwrapped root-quaternion
  heading change from first to last frame) gives the direction. ``Turn`` is
  forced Left/Right by the sign; ``Walk`` / ``Rear`` are Straight when
  ``|net yaw| < deadband`` and Left/Right otherwise; ``Immobile`` is undirected.
* The Left/Right sign convention is calibrated against the manual CSV ``Turn``
  Left/Right annotations (agreement ~98.5%), so handedness is grounded in the
  human labels rather than an arbitrary axis sign.

There are no n-grams and no sentinel: the manual behavior track covers every
reference frame, so all 9 codes (see :mod:`c2a.codegen.shared.mlb_labels`) are
populated and every frame receives one. See the workshop reframe investigation
for the per-code frame-count audit (Walk is intrinsically the rarest behavior in
this corpus, so Walk_Right is the weakest code).

Usage:
    python3.12 bash_scripts/_pyrun.py c2a.codegen.mlb_segmenter \\
        --output outputs/tokenizer/mlb_v1/best_codes.npz \\
        [--deadband 0.10] [--csv ...] [--splits ...] [--ref_h5 ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from c2a.codegen.shared.label_alignment import (
    BEHAVIORS,
    BEHAVIOR_TO_INT,
    DIRECTION_TO_INT,
    UNLABELED,
    load_label_frames,
)
from c2a.codegen.shared.mlb_labels import MLB_NUM_CODES, mlb_code
from c2a.paths import get_repo_root

_N_FRAMES = 250  # H5 frames per reference clip


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def _yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Heading (yaw about world z) from a (..., 4) ``(w, x, y, z)`` quaternion."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _load_reference_yaw(ref_h5: str | Path) -> np.ndarray:
    """Return ``(N_clips, n_frames)`` unwrapped per-frame yaw from the ref H5.

    Clip id == snippet position (the convention the KPMS codes and label
    cross-tabs already use). Root quaternion is ``qpos[:, 3:7]``.
    """
    with h5py.File(ref_h5, "r") as f:
        qpos = f["qpos"][:].reshape(-1, _N_FRAMES, f["qpos"].shape[-1])
    return np.unwrap(_yaw_from_quat(qpos[:, :, 3:7]), axis=1)


def _bouts(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous ``(start, end, label)`` runs over one clip's behavior row."""
    out: list[tuple[int, int, int]] = []
    s = 0
    for i in range(1, len(row)):
        if row[i] != row[i - 1]:
            out.append((s, i, int(row[s])))
            s = i
    out.append((s, len(row), int(row[s])))
    return out


def _calibrate_sign(
    behavior: np.ndarray, direction: np.ndarray, yaw: np.ndarray,
) -> float:
    """Choose the sign so that ``sign * net_yaw > 0`` means CSV-``Left`` turns.

    Returns +1.0 or -1.0. Falls back to +1.0 if no labeled turn bouts exist.
    """
    net_left: list[float] = []
    net_right: list[float] = []
    turn_i = BEHAVIOR_TO_INT["Turn"]
    for pos in range(behavior.shape[0]):
        yc = yaw[pos]
        for s, e, bl in _bouts(behavior[pos]):
            if bl != turn_i:
                continue
            dv = direction[pos, s:e]
            dv = dv[dv != UNLABELED]
            if dv.size == 0:
                continue
            cdir = int(np.bincount(dv).argmax())
            net = float(yc[e - 1] - yc[s])
            if cdir == DIRECTION_TO_INT["Left"]:
                net_left.append(net)
            elif cdir == DIRECTION_TO_INT["Right"]:
                net_right.append(net)
    ml = float(np.mean(net_left)) if net_left else 0.0
    mr = float(np.mean(net_right)) if net_right else 0.0
    return 1.0 if ml >= mr else -1.0


# ---------------------------------------------------------------------------
# Code assignment
# ---------------------------------------------------------------------------
def _direction_idx(net_yaw: float, deadband: float, force_lr: bool) -> int:
    """Net yaw + deadband -> a :data:`DIRECTION_TO_INT` index (Left/Right/Straight)."""
    if not force_lr and abs(net_yaw) < deadband:
        return DIRECTION_TO_INT["Straight"]
    return DIRECTION_TO_INT["Left"] if net_yaw > 0 else DIRECTION_TO_INT["Right"]


def compute_mlb_codes(
    csv_path: str | Path,
    ref_h5: str | Path,
    deadband: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, float, dict[int, int]]:
    """Assign every reference frame one of the 9 MLB codes.

    Returns ``(codes, clip_indices, sign, frame_counts)`` where ``codes`` is
    ``(N_clips, n_frames)`` int32 and ``frame_counts`` maps code id -> frames.
    """
    arr = load_label_frames(csv_path)
    behavior, direction, clip_idx = arr.behavior, arr.direction, arr.clip_indices
    yaw = _load_reference_yaw(ref_h5)
    if yaw.shape[0] != behavior.shape[0]:
        raise ValueError(
            f"reference H5 has {yaw.shape[0]} clips but labels have "
            f"{behavior.shape[0]}; clip alignment broken"
        )

    sign = _calibrate_sign(behavior, direction, yaw)
    turn_i = BEHAVIOR_TO_INT["Turn"]
    immobile_i = BEHAVIOR_TO_INT["Immobile"]

    codes = np.full(behavior.shape, -1, dtype=np.int32)
    for pos in range(behavior.shape[0]):
        yc = yaw[pos]
        for s, e, bl in _bouts(behavior[pos]):
            if bl == UNLABELED:
                continue  # filled below (audited to be empty for current CSV)
            bname = BEHAVIORS[bl]
            if bl == immobile_i:
                code = mlb_code("Immobile", DIRECTION_TO_INT["Straight"])
            else:
                net = sign * float(yc[e - 1] - yc[s])
                d_idx = _direction_idx(net, deadband, force_lr=(bl == turn_i))
                code = mlb_code(bname, d_idx)
            codes[pos, s:e] = code

    # No unlabeled frames in the current CSV; guard against silent gaps.
    n_gap = int((codes < 0).sum())
    if n_gap:
        raise ValueError(
            f"{n_gap} reference frames have no behavior label; MLB requires full "
            f"coverage (D1: no sentinel). Inspect the labels CSV."
        )

    frame_counts = {c: int((codes == c).sum()) for c in range(MLB_NUM_CODES)}
    return codes, clip_idx, sign, frame_counts


# ---------------------------------------------------------------------------
# Build + save (CodesPayload npz)
# ---------------------------------------------------------------------------
def build_codes(
    csv_path: str | Path,
    splits_json: str | Path,
    ref_h5: str | Path,
    deadband: float = 0.10,
) -> dict[str, Any]:
    """Run the segmenter and return arrays + provenance for ``save_codes``."""
    codes, clip_idx, sign, frame_counts = compute_mlb_codes(
        csv_path, ref_h5, deadband
    )
    pos_of = {int(c): i for i, c in enumerate(clip_idx)}

    with open(splits_json) as f:
        splits = json.load(f)
    train_indices = np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)
    test_indices = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)

    def subset(indices: np.ndarray) -> np.ndarray:
        return np.stack([codes[pos_of[int(c)]] for c in indices]).astype(np.int32)

    provenance = {
        "segmenter": "mlb",
        "deadband_rad": float(deadband),
        "left_sign": float(sign),
        "frame_counts": {str(k): v for k, v in frame_counts.items()},
        "csv_path": str(csv_path),
        "ref_h5": str(ref_h5),
    }
    return {
        "train_codes": subset(train_indices),
        "test_codes": subset(test_indices),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "num_codes": MLB_NUM_CODES,
        "frame_counts": frame_counts,
        "provenance": provenance,
    }


def save_codes(payload: dict[str, Any], output: str | Path) -> None:
    """Save in the CodesPayload-compatible npz format (``_load_kpms_npz``)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_codes=payload["train_codes"].astype(np.int32),
        test_codes=payload["test_codes"].astype(np.int32),
        train_indices=payload["train_indices"].astype(np.int64),
        test_indices=payload["test_indices"].astype(np.int64),
        num_states=np.int32(payload["num_codes"]),
        is_merged=np.bool_(False),
        model_type=np.array("mlb"),
        provenance_json=np.array(json.dumps(payload["provenance"])),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--deadband", type=float, default=0.10,
                    help="straight deadband in radians of net heading change")
    ap.add_argument("--csv", type=str,
                    default=str(get_repo_root() / "data/rodent/rodent_labels.csv"))
    ap.add_argument("--splits", type=str,
                    default=str(get_repo_root()
                               / "data/rodent/rodent_balanced_splits.json"))
    ap.add_argument("--ref_h5", type=str,
                    default=str(get_repo_root()
                               / "data/rodent/rodent_reference_clips.h5"))
    args = ap.parse_args(argv)

    payload = build_codes(args.csv, args.splits, args.ref_h5, args.deadband)
    save_codes(payload, args.output)

    from c2a.codegen.shared.mlb_labels import MLB_LABELS
    total = sum(payload["frame_counts"].values())
    print(f"\nSaved {args.output}")
    print(f"  deadband={args.deadband} rad  left_sign={payload['provenance']['left_sign']:+.0f}")
    print(f"  train_codes={payload['train_codes'].shape}  "
          f"test_codes={payload['test_codes'].shape}  num_codes={payload['num_codes']}")
    print(f"  per-code frame counts (full corpus):")
    for c in range(MLB_NUM_CODES):
        n = payload["frame_counts"][c]
        print(f"    {c} {MLB_LABELS[c]:<14} {n:>8}  ({100*n/total:5.2f}%)")


if __name__ == "__main__":
    main()
