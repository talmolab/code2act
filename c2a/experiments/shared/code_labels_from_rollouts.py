"""Per-code / per-cluster labels derived from the rolled-out qpos.

This complements ``code_labels.py`` (which derives labels from corpus
frame-majority over rodent_labels.csv). The pipeline here labels the
*generated motion* produced when each code is held in decoder-only mode:

  body label  = longest behavior segment from TopoMimic auto-labeler,
                with direction = majority of the direction track within
                that span (Groom → Immobile per c2a's 4-class corpus).
  code label  = majority vote across the K rollout bodies of one code.
  cluster label = majority vote across the codes in one merged cluster.

The voting unit at each level is "one entity, one vote" — a body votes
once with its longest-behavior label; a code votes once with its modal
body-label; a cluster aggregates code votes. Ties at the body→code level
are broken by total span-frames (the body with the longer span wins).
Ties at the code→cluster level are broken by total span-frames across
the cluster's bodies.

Same JSON schema as :func:`c2a.experiments.shared.code_labels.compute_code_labels`
output, so the downstream cluster JSONs and behavior_mmd reader don't
have to change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from c2a.codegen.shared.label_alignment import (
    BEHAVIORS,
    BEHAVIOR_TO_INT,
    DIRECTIONS,
    DIRECTION_TO_INT,
    UNLABELED,
)
from c2a.experiments.shared.code_labels import code_label_to_string
from c2a.experiments.shared.topomimic_labeler import (
    DEFAULT_LABELER_FPS,
    label_rollout_body,
)

log = logging.getLogger(__name__)


__all__ = [
    "label_code_from_rollout",
    "label_codes_from_rollouts_dir",
    "code_label_to_string",
    "cluster_majority_label_rollout",
    "label_based_clusters_rollout",
    "save_code_labels_json",
    "load_code_labels_json",
]


# ---------------------------------------------------------------------------
# Per-code labeling
# ---------------------------------------------------------------------------


def label_code_from_rollout(
    qpos_K_T_D: np.ndarray,
    *,
    src_fps: float = 100.0,
    dst_fps: float = DEFAULT_LABELER_FPS,
    cfg: Optional[Dict[str, float]] = None,
    groom_to_immobile: bool = True,
) -> Dict[str, Any]:
    """Label one code given its K-body rollout qpos.

    Args:
        qpos_K_T_D: ``(K, T, nq)`` rolled-out qpos for one code (typically
            K=5 bodies, T=500 frames at 100 Hz).
        src_fps: env control rate (rollouts).
        dst_fps: labeler-calibrated fps; resamples qpos to this before
            running the heuristics.
        cfg: TopoMimic config overrides.
        groom_to_immobile: map Groom → Immobile to match c2a's 4-class
            corpus convention.

    Returns:
        Dict with the same schema as :func:`code_labels.compute_code_labels`
        per-code value, but the distributions count *votes from bodies*
        rather than corpus frames::

            {
                "behavior":     int,  # modal body behavior
                "direction":    int,  # modal body direction in its span
                "label":        str,  # rendered, e.g. "Turn_L"
                "frames_total": int,  # sum of body span lengths
                                      # (in labeler-fps frames)
                "frames_labeled": int,  # same, since UNLABELED bodies
                                        # contribute 0
                "behavior_dist": list[int],  # length BEHAVIORS+1; last = UNLABELED
                "direction_dist": list[int], # length DIRECTIONS+1; last = UNLABELED
                "per_body": list[dict],      # one entry per body
            }
    """
    if qpos_K_T_D.ndim != 3:
        raise ValueError(f"qpos_K_T_D must be (K, T, D); got {qpos_K_T_D.shape}")

    K = qpos_K_T_D.shape[0]
    n_b = len(BEHAVIORS)
    n_d = len(DIRECTIONS)
    behavior_counts = np.zeros(n_b + 1, dtype=np.int64)
    direction_counts = np.zeros(n_d + 1, dtype=np.int64)
    span_by_b = np.zeros(n_b + 1, dtype=np.int64)
    span_by_d = np.zeros(n_d + 1, dtype=np.int64)
    per_body: List[Dict[str, Any]] = []
    frames_total = 0
    frames_labeled = 0

    for k in range(K):
        rl = label_rollout_body(
            qpos_K_T_D[k], src_fps=src_fps, dst_fps=dst_fps,
            cfg=cfg, groom_to_immobile=groom_to_immobile,
        )
        frames_total += int(rl.span_frames)
        b_int = int(rl.behavior_int)
        d_int = int(rl.direction_int)
        b_slot = b_int if 0 <= b_int < n_b else n_b
        d_slot = d_int if 0 <= d_int < n_d else n_d
        behavior_counts[b_slot] += 1
        direction_counts[d_slot] += 1
        span_by_b[b_slot] += int(rl.span_frames)
        span_by_d[d_slot] += int(rl.span_frames)
        if 0 <= b_int < n_b and 0 <= d_int < n_d:
            frames_labeled += int(rl.span_frames)
        per_body.append({
            "body_idx": k,
            "behavior": rl.behavior,
            "direction": rl.direction,
            "behavior_int": b_int,
            "direction_int": d_int,
            "span_frames": int(rl.span_frames),
            "span_start": int(rl.span_start),
            "span_end": int(rl.span_end),
            "total_frames": int(rl.total_frames),
        })

    # Modal behavior: argmax over labeled slots, tie-break by total span
    # frames (so a tie between two behaviors goes to whichever holds the
    # longer cumulative span). UNLABELED only wins if no body has any
    # labeled vote.
    labeled_b = behavior_counts[:n_b]
    if labeled_b.sum() > 0:
        max_count = int(labeled_b.max())
        ties = np.where(labeled_b == max_count)[0]
        # Tie-break: total span frames across the tied slots.
        if ties.size > 1:
            tied_spans = span_by_b[:n_b][ties]
            b_mode = int(ties[int(np.argmax(tied_spans))])
        else:
            b_mode = int(ties[0])
    else:
        b_mode = UNLABELED

    labeled_d = direction_counts[:n_d]
    if labeled_d.sum() > 0:
        max_count = int(labeled_d.max())
        ties = np.where(labeled_d == max_count)[0]
        if ties.size > 1:
            tied_spans = span_by_d[:n_d][ties]
            d_mode = int(ties[int(np.argmax(tied_spans))])
        else:
            d_mode = int(ties[0])
    else:
        d_mode = UNLABELED

    label = code_label_to_string(b_mode, d_mode)
    return {
        "behavior": int(b_mode),
        "direction": int(d_mode),
        "label": label,
        "frames_total": int(frames_total),
        "frames_labeled": int(frames_labeled),
        "behavior_dist": behavior_counts.tolist(),
        "direction_dist": direction_counts.tolist(),
        "per_body": per_body,
    }


def label_codes_from_rollouts_dir(
    rollouts_data_dir: Path,
    code_ids: List[int],
    num_codes: int,
    poses: Tuple[str, ...] = ("walk", "rear"),
    *,
    src_fps: float = 100.0,
    dst_fps: float = DEFAULT_LABELER_FPS,
    cfg: Optional[Dict[str, float]] = None,
    groom_to_immobile: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Aggregate per-code labels across all bodies of all poses.

    Each pose's K bodies count as additional votes for the same code (a
    code's identity is the same regardless of starting pose). This
    matches the previous ``compute_code_labels`` convention of pooling
    train + test occurrences.

    Codes not present in ``code_ids`` are still emitted in the returned
    dict with zero distributions / UNLABELED label, so downstream consumers
    that index by code id never KeyError.
    """
    n_b = len(BEHAVIORS)
    n_d = len(DIRECTIONS)
    out: Dict[int, Dict[str, Any]] = {}
    for c in range(num_codes):
        out[c] = {
            "behavior": UNLABELED, "direction": UNLABELED, "label": "unlabeled",
            "frames_total": 0, "frames_labeled": 0,
            "behavior_dist": [0] * (n_b + 1),
            "direction_dist": [0] * (n_d + 1),
            "per_body": [],
        }

    for code in code_ids:
        per_body_all: List[Dict[str, Any]] = []
        beh_count = np.zeros(n_b + 1, dtype=np.int64)
        dir_count = np.zeros(n_d + 1, dtype=np.int64)
        span_by_b = np.zeros(n_b + 1, dtype=np.int64)
        span_by_d = np.zeros(n_d + 1, dtype=np.int64)
        frames_total = 0
        frames_labeled = 0

        for pose in poses:
            p = rollouts_data_dir / f"rollout_{pose}_code{int(code):03d}.npz"
            if not p.exists():
                log.warning(f"missing rollout {p}; skipping code {code}/{pose}")
                continue
            d = np.load(p)
            qpos = d["qpos"]
            info = label_code_from_rollout(
                qpos, src_fps=src_fps, dst_fps=dst_fps,
                cfg=cfg, groom_to_immobile=groom_to_immobile,
            )
            beh_count += np.asarray(info["behavior_dist"], dtype=np.int64)
            dir_count += np.asarray(info["direction_dist"], dtype=np.int64)
            for b in info["per_body"]:
                b = {**b, "pose": pose}
                per_body_all.append(b)
                b_int = b["behavior_int"]
                d_int = b["direction_int"]
                b_slot = b_int if 0 <= b_int < n_b else n_b
                d_slot = d_int if 0 <= d_int < n_d else n_d
                span_by_b[b_slot] += int(b["span_frames"])
                span_by_d[d_slot] += int(b["span_frames"])
            frames_total += int(info["frames_total"])
            frames_labeled += int(info["frames_labeled"])

        labeled_b = beh_count[:n_b]
        if labeled_b.sum() > 0:
            max_count = int(labeled_b.max())
            ties = np.where(labeled_b == max_count)[0]
            if ties.size > 1:
                b_mode = int(ties[int(np.argmax(span_by_b[:n_b][ties]))])
            else:
                b_mode = int(ties[0])
        else:
            b_mode = UNLABELED
        labeled_d = dir_count[:n_d]
        if labeled_d.sum() > 0:
            max_count = int(labeled_d.max())
            ties = np.where(labeled_d == max_count)[0]
            if ties.size > 1:
                d_mode = int(ties[int(np.argmax(span_by_d[:n_d][ties]))])
            else:
                d_mode = int(ties[0])
        else:
            d_mode = UNLABELED

        out[int(code)] = {
            "behavior": int(b_mode),
            "direction": int(d_mode),
            "label": code_label_to_string(b_mode, d_mode),
            "frames_total": int(frames_total),
            "frames_labeled": int(frames_labeled),
            "behavior_dist": beh_count.tolist(),
            "direction_dist": dir_count.tolist(),
            "per_body": per_body_all,
        }
    return out


# ---------------------------------------------------------------------------
# Per-cluster aggregation (one vote per code, frame-span tiebreak)
# ---------------------------------------------------------------------------


def cluster_majority_label_rollout(
    cluster_codes: List[int],
    code_labels: Dict[int, Dict[str, Any]],
) -> str:
    """Modal code-label across a cluster's codes.

    Each code casts ONE vote with its own ``label`` string (one-code-one-
    vote — not weighted by body span). Ties broken by total ``frames_total``
    across the codes contributing each label.
    """
    if not cluster_codes:
        return "unlabeled"
    counts: Dict[str, int] = {}
    weights: Dict[str, int] = {}
    for c in cluster_codes:
        info = code_labels.get(int(c))
        if info is None:
            continue
        lab = str(info.get("label", "unlabeled"))
        counts[lab] = counts.get(lab, 0) + 1
        weights[lab] = weights.get(lab, 0) + int(info.get("frames_total", 0))
    if not counts:
        return "unlabeled"
    max_count = max(counts.values())
    ties = [lab for lab, c in counts.items() if c == max_count]
    if len(ties) == 1:
        return ties[0]
    return max(ties, key=lambda lab: weights.get(lab, 0))


def label_based_clusters_rollout(
    code_ids: List[int],
    code_labels: Dict[int, Dict[str, Any]],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Group codes by their (rollout-derived) label string.

    Returns clusters sorted by total ``frames_total`` descending — the
    "label that the model most prolifically produces" comes first.
    """
    bucket: Dict[str, List[int]] = {}
    for c in code_ids:
        info = code_labels.get(int(c))
        lab = str(info["label"]) if info is not None else "unlabeled"
        bucket.setdefault(lab, []).append(int(c))
    items = sorted(
        bucket.items(),
        key=lambda kv: -sum(
            int(code_labels.get(int(c), {}).get("frames_total", 0))
            for c in kv[1]
        ),
    )
    clusters = [codes for _, codes in items]
    info = {
        "strategy": "label_majority_rollout",
        "labels": [lab for lab, _ in items],
        "code_ids": list(code_ids),
        "cutoff_used": None,
    }
    return clusters, info


# ---------------------------------------------------------------------------
# JSON IO (matches code_labels.save_code_labels_json schema)
# ---------------------------------------------------------------------------


def save_code_labels_json(
    path: Path,
    code_labels: Dict[int, Dict[str, Any]],
    *,
    source: str,
) -> None:
    """Serialize rollout-derived per-code labels to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "behaviors": list(BEHAVIORS),
        "directions": list(DIRECTIONS),
        "codes": {
            str(int(c)): {**info} for c, info in sorted(code_labels.items())
        },
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_code_labels_json(path: Path) -> Dict[int, Dict[str, Any]]:
    path = Path(path)
    with path.open() as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload["codes"].items()}
