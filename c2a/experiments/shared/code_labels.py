"""Per-code majority-vote labels from rodent_labels.csv.

For each code id, cross-tab the full train+test code time series against the
labeled behavior/direction annotations to determine the dominant behavioral
context the code occupies. Used by both ``run_code_grid_rollouts`` (for
panel banners + label-based clusters) and ``run_code_grid_dynamics`` (for
DSA / rSLDS plot annotations and label-cluster dynamics).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from c2a.codegen.shared.label_alignment import (
    BEHAVIORS,
    DIRECTIONS,
    UNLABELED,
)

log = logging.getLogger(__name__)


__all__ = [
    "compute_code_labels",
    "code_label_to_string",
    "cluster_majority_label",
    "label_based_clusters",
    "save_code_labels_json",
    "load_code_labels_json",
]


def code_label_to_string(behavior: int, direction: int) -> str:
    """Render (behavior_int, direction_int) as a short display label.

    Full 4 × 3 = 12 cell space. ``"unlabeled"`` if behavior is UNLABELED;
    bare ``"<Behavior>"`` if behavior is labeled but direction is UNLABELED;
    ``"<Behavior>_<DirInitial>"`` (e.g. ``"Walk_L"``, ``"Immobile_S"``)
    when both are labeled.
    """
    if behavior < 0 or behavior >= len(BEHAVIORS):
        return "unlabeled"
    bname = BEHAVIORS[behavior]
    if direction < 0 or direction >= len(DIRECTIONS):
        return bname
    dname = DIRECTIONS[direction]
    return f"{bname}_{dname[0]}"  # Immobile_S, Walk_L, Turn_R, etc.


def compute_code_labels(
    train_codes: np.ndarray,
    train_indices: np.ndarray,
    test_codes: np.ndarray,
    test_indices: np.ndarray,
    behavior_arr: np.ndarray,
    direction_arr: np.ndarray,
    label_clip_indices: np.ndarray,
    num_codes: int,
) -> dict[int, dict[str, Any]]:
    """Majority-vote label per code over the full labeled corpus.

    Args:
        train_codes: ``(n_train_clips, T)`` int per-frame code ids. Rows are
            indexed by position in ``train_indices``.
        train_indices: ``(n_train_clips,)`` clip ids matching ``train_codes`` rows.
        test_codes / test_indices: analogous for the test split.
        behavior_arr: ``(N_labeled_clips, T)`` int32 in {UNLABELED, 0..3}
            from :func:`load_label_frames`.
        direction_arr: ``(N_labeled_clips, T)`` int32 in {UNLABELED, 0..2}.
        label_clip_indices: ``(N_labeled_clips,)`` clip ids for the rows of
            ``behavior_arr`` / ``direction_arr``. Returned by
            :class:`LabelArrays` as ``clip_indices``.
        num_codes: codebook cardinality (highest code id + 1).

    Returns:
        ``{code_id: {
            "behavior": int,              # mode behavior across labeled frames
            "direction": int,             # mode direction across labeled frames
            "label": str,                 # human-readable
            "frames_total": int,          # frames where this code appeared in
                                          # labeled clips
            "frames_labeled": int,        # of those, frames with both labels
            "behavior_dist": list[int],   # length len(BEHAVIORS) + 1; last is UNLABELED
            "direction_dist": list[int],  # length len(DIRECTIONS) + 1; last is UNLABELED
        }}``

    Codes that never appear in any labeled clip get all-zero distributions and
    ``label="unlabeled"``.
    """
    if behavior_arr.shape != direction_arr.shape:
        raise ValueError(
            f"behavior_arr.shape {behavior_arr.shape} != direction_arr.shape "
            f"{direction_arr.shape}"
        )
    if label_clip_indices.shape[0] != behavior_arr.shape[0]:
        raise ValueError(
            f"label_clip_indices ({label_clip_indices.shape[0]}) and label arr "
            f"rows ({behavior_arr.shape[0]}) must agree"
        )
    # Codes and labels must share the per-clip frame count, else the
    # cross-tab silently truncates and labels get misaligned in time.
    T_code = int(train_codes.shape[1]) if train_codes.size else 0
    T_label = int(behavior_arr.shape[1])
    if T_code and T_code != T_label:
        raise ValueError(
            f"train_codes T={T_code} != behavior_arr T={T_label}. "
            "Check ckpt_cfg.env_config.clip_length vs "
            "load_label_frames(n_frames=...) and csv_to_h5_subsample."
        )
    labeled_set = set(int(c) for c in label_clip_indices.tolist())
    split_set = set(int(c) for c in train_indices.tolist()) | set(
        int(c) for c in test_indices.tolist()
    )
    missing = split_set - labeled_set
    if missing:
        frac = len(missing) / max(len(split_set), 1)
        log.warning(
            "compute_code_labels: %d/%d split clips (%.1f%%) are outside "
            "the labeled clip range and will not contribute to label votes.",
            len(missing), len(split_set), 100.0 * frac,
        )

    n_b = len(BEHAVIORS)
    n_d = len(DIRECTIONS)
    # +1 slot for UNLABELED (-1). We map UNLABELED → index n_b / n_d.
    behavior_counts = np.zeros((num_codes, n_b + 1), dtype=np.int64)
    direction_counts = np.zeros((num_codes, n_d + 1), dtype=np.int64)
    frames_total = np.zeros(num_codes, dtype=np.int64)
    frames_labeled = np.zeros(num_codes, dtype=np.int64)

    label_clip_lookup = {int(c): i for i, c in enumerate(label_clip_indices)}

    def _accumulate(codes_arr: np.ndarray, clip_ids: np.ndarray) -> None:
        for i, clip_id in enumerate(clip_ids):
            cid = int(clip_id)
            if cid not in label_clip_lookup:
                continue
            li = label_clip_lookup[cid]
            codes_row = codes_arr[i]
            beh_row = behavior_arr[li]
            dir_row = direction_arr[li]
            T = min(codes_row.shape[0], beh_row.shape[0])
            for t in range(T):
                c = int(codes_row[t])
                if c < 0 or c >= num_codes:
                    continue
                frames_total[c] += 1
                b = int(beh_row[t])
                d = int(dir_row[t])
                bi = b if (0 <= b < n_b) else n_b
                di = d if (0 <= d < n_d) else n_d
                behavior_counts[c, bi] += 1
                direction_counts[c, di] += 1
                if 0 <= b < n_b and 0 <= d < n_d:
                    frames_labeled[c] += 1

    _accumulate(train_codes, train_indices)
    _accumulate(test_codes, test_indices)

    out: dict[int, dict[str, Any]] = {}
    for c in range(num_codes):
        b_counts_lab = behavior_counts[c, :n_b]
        d_counts_lab = direction_counts[c, :n_d]
        if b_counts_lab.sum() > 0:
            b_mode = int(np.argmax(b_counts_lab))
        else:
            b_mode = UNLABELED
        if d_counts_lab.sum() > 0:
            d_mode = int(np.argmax(d_counts_lab))
        else:
            d_mode = UNLABELED
        label = code_label_to_string(b_mode, d_mode)
        out[c] = {
            "behavior": b_mode,
            "direction": d_mode,
            "label": label,
            "frames_total": int(frames_total[c]),
            "frames_labeled": int(frames_labeled[c]),
            "behavior_dist": behavior_counts[c].tolist(),
            "direction_dist": direction_counts[c].tolist(),
        }
    return out


def cluster_majority_label(
    cluster_codes: list[int],
    code_labels: dict[int, dict[str, Any]],
) -> str:
    """Frame-weighted majority label across a cluster's constituent codes.

    Each constituent code casts a vote equal to its ``frames_total`` for its
    own modal label string. The label with the largest summed weight wins.
    Returns ``"unlabeled"`` if all constituents have zero frames or are
    unknown to ``code_labels``.

    Implication: a cluster of {rare-Walk_L code, common-unlabeled code}
    resolves to ``"unlabeled"`` because the unlabeled code contributes more
    frames. This matches the corpus prior — the cluster behaves like an
    unlabeled regime in practice — but if you want unweighted modal vote,
    use a different aggregation.
    """
    if not cluster_codes:
        return "unlabeled"
    weights: dict[str, int] = {}
    for c in cluster_codes:
        info = code_labels.get(int(c))
        if info is None:
            continue
        lab = str(info["label"])
        weights[lab] = weights.get(lab, 0) + int(info["frames_total"])
    if not weights:
        return "unlabeled"
    return max(weights.items(), key=lambda kv: kv[1])[0]


def label_based_clusters(
    code_ids: list[int],
    code_labels: dict[int, dict[str, Any]],
) -> tuple[list[list[int]], dict[str, Any]]:
    """Group codes by their majority-vote label string.

    Args:
        code_ids: subset of code ids to cluster (e.g. the top-N by usage).
        code_labels: output of :func:`compute_code_labels`.

    Returns:
        ``(clusters, info)`` — ``clusters`` is a list of code-id lists sorted
        by total usage descending. ``info`` carries the per-cluster label
        and total-frames-of-occurrence weight.
    """
    bucket: dict[str, list[int]] = {}
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
        "strategy": "label_majority",
        "labels": [lab for lab, _ in items],
        "code_ids": list(code_ids),
        "cutoff_used": None,
    }
    return clusters, info


def save_code_labels_json(
    path: Path,
    code_labels: dict[int, dict[str, Any]],
    *,
    source: str,
) -> None:
    """Serialize code-label info to JSON (int keys → string for portability)."""
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


def load_code_labels_json(path: Path) -> dict[int, dict[str, Any]]:
    """Load code-label info written by :func:`save_code_labels_json`."""
    path = Path(path)
    with path.open() as f:
        payload = json.load(f)
    return {int(k): v for k, v in payload["codes"].items()}
