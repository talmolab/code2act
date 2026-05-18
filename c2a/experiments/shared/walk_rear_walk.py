"""Find clips with a 3-segment transition like Rear → Walk → Rear.

Scans the per-clip labeled segments from :func:`load_label_frames` and
selects clips where three consecutive segments match a target
``(A, B, A)`` behavior pattern. Caller chooses strict (no behaviors in
between) or loose (short bridges absorbed).

The default pattern is **Rear → Walk → Rear** because the rodent corpus
has 0 Walk→Rear→Walk clips but 82 Rear→Walk→Rear candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from c2a.codegen.shared.label_alignment import BEHAVIORS, UNLABELED


@dataclass
class WRWClip:
    """One transition (A → B → A) in a single clip.

    Field names kept as walk/rear for backward compat; semantically they
    just label the outer and inner behaviors of the (outer, inner, outer)
    sequence.
    """
    clip_id: int
    walk1_start: int       # outer-1 start  (was: walk1 — kept name)
    walk1_end: int
    rear_start: int        # inner start    (was: rear)
    rear_end: int
    walk2_start: int       # outer-2 start
    walk2_end: int

    @property
    def total_frames(self) -> int:
        return int(self.walk2_end - self.walk1_start)

    @property
    def rear_frames(self) -> int:
        return int(self.rear_end - self.rear_start)


def find_walk_rear_walk_clips(
    behavior_arr: np.ndarray,
    label_clip_indices: np.ndarray,
    *,
    outer: str = "Rear",
    inner: str = "Walk",
    strict: bool = True,
    bridge_max_frames: int = 30,
    min_outer_frames: int = 5,
    min_inner_frames: int = 10,
    k: int = 5,
) -> List[WRWClip]:
    """Return up to ``k`` (outer, inner, outer) transitions, sorted by
    total sequence length (longest first).

    Defaults to Rear → Walk → Rear (the only ``A → B → A`` pattern with
    nontrivial population in the corpus).

    Args:
        behavior_arr: ``(N_labeled, T)`` int32 from :func:`load_label_frames`.
        label_clip_indices: ``(N_labeled,)`` clip ids matching the rows.
        outer: outer behavior name (default ``"Rear"``).
        inner: inner behavior name (default ``"Walk"``).
        strict: if True, require the three segments to be consecutive with
            no other labels between them. If False, allow short bridges
            up to ``bridge_max_frames``.
        bridge_max_frames: maximum bridge length allowed when ``strict=False``.
        min_outer_frames / min_inner_frames: per-segment minimum durations.
        k: maximum number of clips to return.

    Returns:
        List of :class:`WRWClip` (≤ k), longest transition first.
    """
    outer_int = BEHAVIORS.index(outer)
    inner_int = BEHAVIORS.index(inner)
    # Legacy variable names retained in the scanning logic below: walk_int
    # is the OUTER, rear_int is the INNER. The names date back to the
    # original Walk→Rear→Walk request — semantics now controlled by the
    # `outer` / `inner` args.
    walk_int = outer_int
    rear_int = inner_int
    immob_int = BEHAVIORS.index("Immobile")

    out: List[WRWClip] = []
    for row in range(behavior_arr.shape[0]):
        clip_id = int(label_clip_indices[row])
        b = behavior_arr[row]
        T = b.shape[0]
        # Build segment list (label, start, end).
        if T == 0:
            continue
        boundaries = np.where(np.diff(b) != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [T]))
        segs = [(int(b[s]), int(s), int(e)) for s, e in zip(starts, ends)]

        # Scan triples (Walk, Rear, Walk) or with bridges.
        if strict:
            for i in range(len(segs) - 2):
                a, b1 = segs[i], segs[i + 1]
                c = segs[i + 2]
                if (a[0] == walk_int and b1[0] == rear_int and c[0] == walk_int
                    and a[2] - a[1] >= min_outer_frames
                    and b1[2] - b1[1] >= min_inner_frames
                    and c[2] - c[1] >= min_outer_frames):
                    out.append(WRWClip(
                        clip_id=clip_id,
                        walk1_start=a[1], walk1_end=a[2],
                        rear_start=b1[1], rear_end=b1[2],
                        walk2_start=c[1], walk2_end=c[2],
                    ))
        else:
            # Loose: walk → (bridge?) → rear → (bridge?) → walk.
            i = 0
            while i < len(segs):
                if segs[i][0] != walk_int:
                    i += 1
                    continue
                walk1 = segs[i]
                # Find next Rear after walk1.
                j = i + 1
                while j < len(segs):
                    s_lbl, s_s, s_e = segs[j]
                    if s_lbl == rear_int:
                        # Check bridge from walk1.end to rear.start is short.
                        bridge_len = s_s - walk1[2]
                        if bridge_len <= bridge_max_frames:
                            break
                        else:
                            j = len(segs)  # break outer "find rear"
                            break
                    if s_lbl == walk_int:
                        # Another walk before rear — restart from this walk.
                        break
                    j += 1
                if j >= len(segs) or segs[j][0] != rear_int:
                    i += 1
                    continue
                rear = segs[j]
                # Find next Walk after rear.
                kk = j + 1
                while kk < len(segs):
                    s_lbl, s_s, s_e = segs[kk]
                    if s_lbl == walk_int:
                        bridge_len = s_s - rear[2]
                        if bridge_len <= bridge_max_frames:
                            break
                        else:
                            kk = len(segs)
                            break
                    if s_lbl == rear_int:
                        break
                    kk += 1
                if kk >= len(segs) or segs[kk][0] != walk_int:
                    i = j + 1
                    continue
                walk2 = segs[kk]
                if (walk1[2] - walk1[1] >= min_outer_frames
                    and rear[2] - rear[1] >= min_inner_frames
                    and walk2[2] - walk2[1] >= min_outer_frames):
                    out.append(WRWClip(
                        clip_id=clip_id,
                        walk1_start=walk1[1], walk1_end=walk1[2],
                        rear_start=rear[1], rear_end=rear[2],
                        walk2_start=walk2[1], walk2_end=walk2[2],
                    ))
                i = kk + 1

    out.sort(key=lambda c: -c.total_frames)
    return out[:k]
