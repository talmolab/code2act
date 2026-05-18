"""Numpy-friendly wrappers around the vendored auto-labeler.

Run the TopoMimic-annotation behavior + direction heuristics on a single
rolled-out body's qpos array, after resampling from the env's control
rate (typically 100 Hz) to the labeler's calibrated 30 fps.

Returns a structured dataclass: per-body modal behavior (after the
Groom→Immobile mapping for c2a parity), direction within the modal
behavior's span, and the raw segment lists for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from c2a.codegen.shared.label_alignment import (
    BEHAVIORS,
    BEHAVIOR_TO_INT,
    DIRECTIONS,
    DIRECTION_TO_INT,
    UNLABELED,
)

from .auto_labeler import (
    compute_features,
    load_config,
    _classify_frame,
    _group_segments,
)
from .auto_direction import (
    _signed_yaw_rate,
    _group_and_smooth,
    _classify_frame as _classify_dir_frame,
    reconcile_direction_with_turns,
)
from .auto_labeler import _quat_to_yaw, _unwrap


DEFAULT_LABELER_FPS: float = 30.0


def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample ``(T, D)`` qpos from ``src_fps`` to ``dst_fps``.

    Quaternion dims (q3..q6) are linearly interpolated like everything else
    here. The labeler only takes yaw out of the quat via atan2, and small
    interp drift in the quat manifold gets re-normalized inside atan2; in
    practice this matches the calibration-source behavior of the upstream
    repo which reads pre-resampled CSVs.
    """
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be (T, D); got {qpos.shape}")
    if src_fps == dst_fps:
        return qpos.astype(np.float64, copy=False)
    T = qpos.shape[0]
    dur = (T - 1) / float(src_fps)
    n_new = max(2, int(round(dur * float(dst_fps))) + 1)
    t_src = np.linspace(0.0, dur, num=T)
    t_dst = np.linspace(0.0, dur, num=n_new)
    out = np.empty((n_new, qpos.shape[1]), dtype=np.float64)
    for d in range(qpos.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, qpos[:, d])
    return out


@dataclass
class BodyLabel:
    """Per-body label after running the TopoMimic auto-labeler on one rollout.

    ``behavior_int`` / ``direction_int`` follow the c2a integer schemes in
    :mod:`c2a.codegen.shared.label_alignment`. ``UNLABELED = -1`` when no
    valid label could be determined (e.g. all frames Groom + caller asked
    to drop Groom, or empty rollout).
    """
    behavior: str                  # one of BEHAVIORS or "Groom"-mapped, or "unlabeled"
    direction: str                 # Left / Right / Straight or "unlabeled"
    behavior_int: int
    direction_int: int
    span_frames: int               # length of the chosen behavior span (in labeler-fps frames)
    span_start: int
    span_end: int
    behavior_segments: List[Tuple[str, int, int]] = field(default_factory=list)
    direction_segments: List[Tuple[str, int, int]] = field(default_factory=list)
    total_frames: int = 0


def _normalize_behavior(name: str, groom_to_immobile: bool) -> str:
    """Apply the Groom → Immobile mapping (c2a's 4-class corpus convention)."""
    if groom_to_immobile and name == "Groom":
        return "Immobile"
    return name


def _per_frame_behavior(
    rows: np.ndarray, cfg: Dict[str, float],
) -> List[str]:
    """Per-frame 5-class label (Groom mapping is deferred to segmentation)."""
    rows_list = rows.tolist()
    feats = compute_features(rows_list, cfg)
    return [_classify_frame(f, cfg) for f in feats]


def _per_frame_direction(
    rows: np.ndarray, cfg: Dict[str, float],
) -> Tuple[List[str], List[float]]:
    """Per-frame direction + signed yaw rate."""
    rows_list = rows.tolist()
    raw_yaw = [_quat_to_yaw(r[3], r[4], r[5], r[6]) for r in rows_list]
    yaw_unwrapped = _unwrap(raw_yaw)
    window = int(cfg.get("window", 15))
    thresh = float(cfg.get("dir_yaw_thresh", 0.03))
    signed_rate = _signed_yaw_rate(yaw_unwrapped, window)
    labels = [_classify_dir_frame(r, thresh) for r in signed_rate]
    return labels, signed_rate


def _majority_direction_in_span(
    direction_per_frame: List[str], start: int, end: int,
) -> str:
    """Modal direction string within ``[start, end)``."""
    if end <= start:
        return "Straight"
    counts: Dict[str, int] = {"Left": 0, "Right": 0, "Straight": 0}
    for t in range(start, end):
        if 0 <= t < len(direction_per_frame):
            d = direction_per_frame[t]
            if d in counts:
                counts[d] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def label_rollout_body(
    qpos: np.ndarray,
    *,
    src_fps: float = 100.0,
    dst_fps: float = DEFAULT_LABELER_FPS,
    cfg: Optional[Dict[str, float]] = None,
    groom_to_immobile: bool = True,
) -> BodyLabel:
    """Run the TopoMimic auto-labeler on one body's rollout qpos.

    Pipeline:
      1. Resample ``qpos`` from ``src_fps`` to ``dst_fps`` (linear interp).
      2. Run behavior labeler → list of (behavior, start, end) segments.
         Map Groom → Immobile *after* segmentation if requested (so the
         intra-segment heuristics — bridging, short-absorb, immobile-min-
         duration — still see Groom as itself during smoothing).
      3. Pick the LONGEST behavior segment → body's behavior.
      4. Run direction labeler. Reconcile Turn spans (Turn must be L/R).
      5. Within the longest behavior span, majority-vote the per-frame
         direction → body's direction.

    Returns:
        :class:`BodyLabel` with integer + string labels and the raw segment
        lists for downstream inspection.
    """
    if cfg is None:
        cfg = load_config()
    cfg = dict(cfg)
    cfg["fps"] = float(dst_fps)

    qpos_rs = resample_qpos(qpos, src_fps=src_fps, dst_fps=dst_fps)
    if qpos_rs.shape[0] < 2:
        return BodyLabel(
            behavior="unlabeled", direction="unlabeled",
            behavior_int=UNLABELED, direction_int=UNLABELED,
            span_frames=0, span_start=0, span_end=0,
        )

    # --- Behavior pipeline ---
    beh_per_frame = _per_frame_behavior(qpos_rs, cfg)
    beh_segments_raw = _group_segments(beh_per_frame, cfg)  # 5-class
    # Map Groom → Immobile after smoothing, then coalesce any adjacent runs
    # that became identical.
    beh_segments: List[Tuple[str, int, int]] = []
    for lbl, s, e in beh_segments_raw:
        norm = _normalize_behavior(lbl, groom_to_immobile=groom_to_immobile)
        if beh_segments and beh_segments[-1][0] == norm and beh_segments[-1][2] == s:
            pl, ps, _ = beh_segments[-1]
            beh_segments[-1] = (pl, ps, e)
        else:
            beh_segments.append((norm, s, e))

    if not beh_segments:
        return BodyLabel(
            behavior="unlabeled", direction="unlabeled",
            behavior_int=UNLABELED, direction_int=UNLABELED,
            span_frames=0, span_start=0, span_end=0,
            behavior_segments=[], direction_segments=[],
            total_frames=int(qpos_rs.shape[0]),
        )

    # Longest behavior segment wins. Tie-break by earlier start.
    longest = max(beh_segments, key=lambda seg: (seg[2] - seg[1], -seg[1]))
    b_name, b_start, b_end = longest

    # --- Direction pipeline ---
    dir_per_frame, _signed_rate = _per_frame_direction(qpos_rs, cfg)
    dir_segments_raw = _group_and_smooth(dir_per_frame, _signed_rate, cfg)
    # Reconcile: any Turn span in the (mapped-or-original?) behavior
    # segments must be L or R, never Straight. We use the ORIGINAL 5-class
    # Turn spans (pre Groom→Immobile) for this — reconcile cares about
    # actual rotational events, not about how we mapped Groom.
    turn_spans = [(s, e) for lbl, s, e in beh_segments_raw if lbl == "Turn"]
    if turn_spans:
        # Build a row list copy for reconcile (it expects list[list[float]]).
        dir_segs_dict = [
            {"name": lbl, "start_time": s / float(dst_fps),
             "end_time": e / float(dst_fps),
             "start_frame": s, "end_frame": e}
            for lbl, s, e in dir_segments_raw
        ]
        dir_segs_dict = reconcile_direction_with_turns(
            dir_segs_dict, turn_spans, qpos_rs.tolist(), cfg,
        )
        # Re-materialize the per-frame direction array from the reconciled
        # segment dicts so the majority-vote inside the longest behavior
        # span sees the Turn reconciliation.
        n_frames = qpos_rs.shape[0]
        dir_per_frame = ["Straight"] * n_frames
        for d in dir_segs_dict:
            fs = int(d.get("start_frame", 0))
            fe = int(d.get("end_frame", n_frames))
            name = str(d.get("name", "Straight"))
            for t in range(max(0, fs), min(n_frames, fe)):
                dir_per_frame[t] = name
        dir_segments = [
            (str(d["name"]), int(d["start_frame"]), int(d["end_frame"]))
            for d in dir_segs_dict
        ]
    else:
        dir_segments = list(dir_segments_raw)

    d_name = _majority_direction_in_span(dir_per_frame, b_start, b_end)

    # --- Integer encoding ---
    b_int = BEHAVIOR_TO_INT.get(b_name, UNLABELED)
    d_int = DIRECTION_TO_INT.get(d_name, UNLABELED)

    return BodyLabel(
        behavior=b_name,
        direction=d_name,
        behavior_int=int(b_int),
        direction_int=int(d_int),
        span_frames=int(b_end - b_start),
        span_start=int(b_start),
        span_end=int(b_end),
        behavior_segments=beh_segments,
        direction_segments=dir_segments,
        total_frames=int(qpos_rs.shape[0]),
    )
