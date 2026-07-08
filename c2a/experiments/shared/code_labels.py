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
    BEHAVIOR_TO_INT,
    DIRECTIONS,
    DIRECTION_TO_INT,
    UNLABELED,
)
from c2a.codegen.shared.mlb_labels import (
    MLB_COLORS,
    MLB_LABELS,
    MLB_LABEL_TO_INT,
    mlb_code,
)

log = logging.getLogger(__name__)


__all__ = [
    "compute_code_labels",
    "code_label_to_string",
    "cluster_majority_label",
    "label_based_clusters",
    "save_code_labels_json",
    "load_code_labels_json",
    # 9-class (D3) canonical labeling shared across all steering strategies.
    "NINE_CLASS_LABELS",
    "behavior_direction_to_nine",
    "compute_nine_class_labels",
    "nine_class_label_for_code",
]

# Canonical 9-class scheme is *the* cross-strategy behavior vocabulary (D3):
# every steering strategy (kpms / kvae / mlb) colors its codes into one of
# these 9 directional categories. The names/order/colors are owned by
# ``c2a.codegen.shared.mlb_labels`` (the mlb tokenizer's source of truth);
# kpms/kvae codes are *auto-labeled* into them by :func:`compute_nine_class_labels`.
NINE_CLASS_LABELS: tuple[str, ...] = MLB_LABELS


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


# ---------------------------------------------------------------------------
# 9-class (D3) canonical labeling — behavior x kinematic direction.
#
# Unlike :func:`compute_code_labels` (which reads direction from the CSV
# ``direction`` track — annotated ~100% Straight for Walk and therefore useless
# for handedness), the 9-class scheme derives direction *kinematically* from the
# reference body's net heading change over each code occurrence, exactly the way
# the mlb tokenizer does (see :mod:`c2a.codegen.mlb_segmenter`). This keeps the
# global color/legend scheme consistent across all three steering strategies.
# ---------------------------------------------------------------------------


def behavior_direction_to_nine(
    behavior_int: int, direction_idx: int
) -> tuple[str, int]:
    """Map a (modal-behavior, kinematic-direction) pair to a 9-class label.

    Args:
        behavior_int: index into :data:`BEHAVIORS`
            (``0=Immobile, 1=Rear, 2=Turn, 3=Walk``) or :data:`UNLABELED`.
        direction_idx: index into :data:`DIRECTIONS`
            (``0=Left, 1=Right, 2=Straight``). Ignored for ``Immobile``.
            For ``Turn`` it must be Left/Right (the kinematic path guarantees
            this); for ``Walk``/``Rear`` any of the three is valid.

    Returns:
        ``(label, code)`` where ``label`` is one of
        :data:`NINE_CLASS_LABELS` (or ``"unlabeled"``) and ``code`` is the
        matching 0..8 index (or ``-1`` when unlabeled).

    Raises:
        ValueError: ``Turn`` with a non Left/Right direction. Use
            :func:`compute_nine_class_labels` (kinematic) so this never occurs.
    """
    if behavior_int < 0 or behavior_int >= len(BEHAVIORS):
        return "unlabeled", -1
    bname = BEHAVIORS[behavior_int]
    if bname == "Immobile":
        code = MLB_LABEL_TO_INT["Immobile"]
        return MLB_LABELS[code], code
    code = mlb_code(bname, direction_idx)
    return MLB_LABELS[code], code


def nine_class_label_for_code(
    behavior_int: int,
    net_yaws: "np.ndarray | list[float]",
    weights: "np.ndarray | list[float]",
    *,
    deadband: float = 0.10,
) -> tuple[str, int, int]:
    """9-class label for one code from its per-occurrence net-yaw signal.

    This is the per-code primitive: given a code's modal behavior and the
    (sign-applied) net heading change of each of its contiguous occurrences,
    classify a single representative direction (frame-weighted modal vote over
    the per-occurrence Left/Right/Straight calls) and return the 9-class label.

    Direction rules mirror :func:`c2a.codegen.mlb_segmenter._direction_idx`:

    * ``Immobile`` collapses direction (single code, ``"Immobile"``).
    * ``Turn`` is forced Left/Right by net-yaw sign (never Straight).
    * ``Walk`` / ``Rear`` are Straight when ``|net yaw| < deadband`` else L/R.

    Args:
        behavior_int: modal behavior index into :data:`BEHAVIORS`, or
            :data:`UNLABELED`.
        net_yaws: ``(n_occurrences,)`` sign-applied net yaw per occurrence
            (``sign * (yaw[end-1] - yaw[start])``). ``Left := net > 0``.
        weights: ``(n_occurrences,)`` frame counts per occurrence (the vote
            weight). Same length as ``net_yaws``.
        deadband: straight deadband in radians (mlb default 0.10).

    Returns:
        ``(label, code, direction_idx)``. ``direction_idx`` is the resolved
        :data:`DIRECTIONS` index, or ``-1`` for Immobile / unlabeled.
    """
    if behavior_int < 0 or behavior_int >= len(BEHAVIORS):
        return "unlabeled", -1, -1
    bname = BEHAVIORS[behavior_int]
    if bname == "Immobile":
        code = MLB_LABEL_TO_INT["Immobile"]
        return MLB_LABELS[code], code, -1

    net_yaws = np.asarray(net_yaws, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    is_turn = bname == "Turn"
    # Frame-weighted vote over per-occurrence L/R/S calls.
    vote = np.zeros(len(DIRECTIONS), dtype=np.float64)
    for ny, w in zip(net_yaws, weights):
        if is_turn:
            d = DIRECTION_TO_INT["Left"] if ny > 0 else DIRECTION_TO_INT["Right"]
        elif abs(ny) < deadband:
            d = DIRECTION_TO_INT["Straight"]
        else:
            d = DIRECTION_TO_INT["Left"] if ny > 0 else DIRECTION_TO_INT["Right"]
        vote[d] += w
    if vote.sum() <= 0:
        # No kinematic evidence: default to Straight (Walk/Rear) or, for Turn
        # (which has no Straight code), the more populated of L/R is undefined
        # so fall back to Left. This only fires for codes with zero occurrences
        # in the reference, which are already behavior-unlabeled in practice.
        d_idx = DIRECTION_TO_INT["Left"] if is_turn else DIRECTION_TO_INT["Straight"]
    elif is_turn:
        d_idx = int(np.argmax(vote[: DIRECTION_TO_INT["Straight"]]))  # only L/R
    else:
        d_idx = int(np.argmax(vote))
    code = mlb_code(bname, d_idx)
    return MLB_LABELS[code], code, d_idx


def _code_bouts(codes_row: np.ndarray, code: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs where ``codes_row == code``."""
    mask = codes_row == code
    out: list[tuple[int, int]] = []
    s: int | None = None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def compute_nine_class_labels(
    train_codes: np.ndarray,
    train_indices: np.ndarray,
    test_codes: np.ndarray,
    test_indices: np.ndarray,
    behavior_arr: np.ndarray,
    direction_arr: np.ndarray,
    label_clip_indices: np.ndarray,
    ref_yaw: np.ndarray,
    ref_clip_indices: np.ndarray,
    num_codes: int,
    *,
    deadband: float = 0.10,
    sign: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Auto-label every code into one of the 9 canonical directional categories.

    Modal *behavior* is taken from the CSV cross-tab (same as
    :func:`compute_code_labels`); *direction* is derived kinematically from the
    reference body's net yaw over each contiguous code occurrence (the mlb
    convention), so Walk/Rear codes can be Left/Right (the CSV direction track
    cannot supply this — it annotates Walk ~100% Straight).

    Args:
        train_codes / train_indices / test_codes / test_indices: per-clip code
            time series and their clip ids (same as :func:`compute_code_labels`).
        behavior_arr / direction_arr / label_clip_indices: CSV label arrays from
            :func:`c2a.codegen.shared.label_alignment.load_label_frames`.
        ref_yaw: ``(N_ref_clips, T)`` unwrapped reference yaw, e.g. from
            ``c2a.codegen.mlb_segmenter._load_reference_yaw``. Row order matches
            ``ref_clip_indices``.
        ref_clip_indices: ``(N_ref_clips,)`` clip ids for ``ref_yaw`` rows.
        num_codes: codebook cardinality.
        deadband: straight deadband in radians (mlb default 0.10).
        sign: Left-sign convention (``Left := sign * net_yaw > 0``). If ``None``,
            it is calibrated against the CSV Turn Left/Right annotations exactly
            as the mlb tokenizer does (``mlb_segmenter._calibrate_sign``).

    Returns:
        ``{code_id: {**compute_code_labels[code], "nine_label": str,
        "nine_code": int, "nine_direction": int, "color": [r, g, b]}}``. Codes
        with no labeled behavior get ``nine_label="unlabeled"``, ``nine_code=-1``.
    """
    base = compute_code_labels(
        train_codes, train_indices, test_codes, test_indices,
        behavior_arr, direction_arr, label_clip_indices, num_codes,
    )

    if sign is None:
        from c2a.codegen.mlb_segmenter import _calibrate_sign

        # _calibrate_sign indexes behavior/direction/yaw by the SAME row order;
        # align all three to ref_clip_indices.
        ref_pos = {int(c): i for i, c in enumerate(ref_clip_indices)}
        lab_pos = {int(c): i for i, c in enumerate(label_clip_indices)}
        common = [c for c in ref_clip_indices.tolist() if int(c) in lab_pos]
        beh = np.stack([behavior_arr[lab_pos[int(c)]] for c in common])
        dirn = np.stack([direction_arr[lab_pos[int(c)]] for c in common])
        yw = np.stack([ref_yaw[ref_pos[int(c)]] for c in common])
        sign = _calibrate_sign(beh, dirn, yw)

    ref_pos = {int(c): i for i, c in enumerate(ref_clip_indices)}

    def _occurrences(code: int) -> tuple[list[float], list[float]]:
        net_yaws: list[float] = []
        weights: list[float] = []
        for codes_arr, clip_ids in (
            (train_codes, train_indices),
            (test_codes, test_indices),
        ):
            for i, clip_id in enumerate(clip_ids):
                rp = ref_pos.get(int(clip_id))
                if rp is None:
                    continue
                yc = ref_yaw[rp]
                for s, e in _code_bouts(codes_arr[i], code):
                    if e - s < 2:
                        continue
                    net_yaws.append(sign * float(yc[e - 1] - yc[s]))
                    weights.append(float(e - s))
        return net_yaws, weights

    out: dict[int, dict[str, Any]] = {}
    for c in range(num_codes):
        info = dict(base[c])
        b_mode = int(info["behavior"])
        net_yaws, weights = _occurrences(c)
        label, code9, d_idx = nine_class_label_for_code(
            b_mode, net_yaws, weights, deadband=deadband
        )
        info["nine_label"] = label
        info["nine_code"] = int(code9)
        info["nine_direction"] = int(d_idx)
        info["color"] = (
            list(MLB_COLORS[label]) if label in MLB_COLORS else [0.8, 0.8, 0.8]
        )
        out[c] = info
    return out
