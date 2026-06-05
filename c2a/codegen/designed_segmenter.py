"""Hand-designed multi-scale segmenter: labels CSV → per-frame code stream.

Produces a `best_codes.npz` in the same CodesPayload format the
training pipeline expects, but the codebook is hand-crafted rather than
clustering-derived.  Each code corresponds to a *named pattern* over the
behavior-label CSV (single label, palindromic n-gram, directional n-gram,
duration-anchored long bout).  Longest-match-wins arbitrates overlaps —
when a 5-gram and a nested 3-gram both cover the same frame, the 5-gram
claims it.

Usage:
    python3.12 bash_scripts/_pyrun.py c2a.codegen.designed_segmenter \\
        --pattern_yaml c2a/codegen/patterns/multi_scale_v1.yaml \\
        --output outputs/tokenizer/designed_v1/best_codes.npz \\
        [--csv data/rodent/rodent_labels.csv] \\
        [--splits data/rodent/rodent_balanced_splits.json] \\
        [--n_frames 250] [--csv_to_h5_subsample 2]

The resulting npz contains:
    train_codes        (n_train, n_frames) int32
    test_codes         (n_test,  n_frames) int32
    train_indices      (n_train,)          int64  (original CSV clip ids)
    test_indices       (n_test,)           int64
    num_codes          int (= len(codebook) including UNL)
    code_source        str = "designed"
    provenance         dict (pattern_yaml path, per-code instance + frame counts)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from c2a.codegen.shared.label_alignment import BEHAVIORS, UNLABELED
from c2a.paths import get_repo_root


_BEHAVIOR_LABELS = {"Walk", "Rear", "Turn", "Immobile"}


# ---------------------------------------------------------------------------
# Codebook loading & validation
# ---------------------------------------------------------------------------


def load_codebook(yaml_path: str | Path) -> dict[str, Any]:
    """Read the codebook YAML and verify the schema.

    Returns the parsed dict.  Raises ValueError on any inconsistency.
    """
    with open(yaml_path) as f:
        cb = yaml.safe_load(f)

    if "codebook" not in cb:
        raise ValueError(f"{yaml_path}: missing top-level 'codebook'")
    entries = cb["codebook"]
    if not entries:
        raise ValueError(f"{yaml_path}: empty codebook")

    seen_ids = set()
    for e in entries:
        for required in ("id", "name", "type", "tier", "color"):
            if required not in e:
                raise ValueError(f"codebook entry missing {required!r}: {e}")
        if e["id"] in seen_ids:
            raise ValueError(f"duplicate codebook id {e['id']}")
        seen_ids.add(e["id"])
        if e["type"] in ("fine", "long"):
            if "label" not in e:
                raise ValueError(
                    f"codebook entry id={e['id']} type={e['type']} requires "
                    f"'label' field"
                )
            if e["label"] not in _BEHAVIOR_LABELS:
                raise ValueError(
                    f"codebook entry id={e['id']} label={e['label']!r} "
                    f"not in {_BEHAVIOR_LABELS}"
                )
            if e["type"] == "long" and "min_csv_frames" not in e:
                raise ValueError(
                    f"codebook entry id={e['id']} type=long requires "
                    f"'min_csv_frames'"
                )
        elif e["type"] in ("p3", "d3", "n4", "n5"):
            if "labels" not in e:
                raise ValueError(
                    f"codebook entry id={e['id']} type={e['type']} requires "
                    f"'labels' list"
                )
            n = {"p3": 3, "d3": 3, "n4": 4, "n5": 5}[e["type"]]
            if len(e["labels"]) != n:
                raise ValueError(
                    f"codebook entry id={e['id']} type={e['type']} expects "
                    f"{n} labels, got {len(e['labels'])}"
                )
            for lab in e["labels"]:
                if lab not in _BEHAVIOR_LABELS:
                    raise ValueError(
                        f"codebook entry id={e['id']} label={lab!r} "
                        f"not in {_BEHAVIOR_LABELS}"
                    )
        elif e["type"] == "sentinel":
            pass
        else:
            raise ValueError(
                f"codebook entry id={e['id']} unknown type {e['type']!r}"
            )

    # Find the sentinel ID (used to fill unmatched frames).
    sentinels = [e for e in entries if e["type"] == "sentinel"]
    if len(sentinels) != 1:
        raise ValueError(
            f"codebook must have exactly one sentinel entry; got {len(sentinels)}"
        )
    cb["_sentinel_id"] = sentinels[0]["id"]
    cb["_num_codes"] = max(e["id"] for e in entries) + 1
    return cb


# ---------------------------------------------------------------------------
# Per-clip segment list construction
# ---------------------------------------------------------------------------


def _clip_to_index(name: str) -> int:
    s = Path(name).stem
    if not s.startswith("clip_"):
        return -1
    try:
        return int(s.split("_", 1)[1])
    except ValueError:
        return -1


def build_segment_chains(
    csv_path: str | Path,
) -> dict[int, list[tuple[str, int, int]]]:
    """Per clip_id → list of (label, start_csv_frame, end_csv_frame),
    sorted by start frame.  Behavior track only."""
    chains: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("Track") != "behavior":
                continue
            cid = _clip_to_index(row["Video"])
            if cid < 0:
                continue
            chains[cid].append(
                (row["Label"], int(row["Start_Frame"]), int(row["End_Frame"]))
            )
    for cid in chains:
        chains[cid].sort(key=lambda x: x[1])
    return chains


# ---------------------------------------------------------------------------
# Match enumeration per clip
# ---------------------------------------------------------------------------


def _enumerate_matches(
    chain: list[tuple[str, int, int]], codebook_entries: list[dict[str, Any]],
) -> list[tuple[int, int, int, int]]:
    """For one clip's segment chain, enumerate every (span_start, span_end,
    code_id, span_length) tuple for every codebook entry that matches.

    span_start / span_end are CSV-frame indices (inclusive start, exclusive
    end).  Length is end - start.
    """
    matches = []
    n_segs = len(chain)
    for entry in codebook_entries:
        t = entry["type"]
        if t == "sentinel":
            continue
        if t == "fine":
            target = entry["label"]
            for lab, s, e in chain:
                if lab == target:
                    matches.append((s, e, entry["id"], e - s))
        elif t == "long":
            target = entry["label"]
            min_d = int(entry["min_csv_frames"])
            for lab, s, e in chain:
                if lab == target and (e - s) >= min_d:
                    matches.append((s, e, entry["id"], e - s))
        elif t in ("p3", "d3", "n4", "n5"):
            seq = entry["labels"]
            n = len(seq)
            for i in range(n_segs - n + 1):
                window = chain[i : i + n]
                if all(window[j][0] == seq[j] for j in range(n)):
                    span_s = window[0][1]
                    span_e = window[-1][2]
                    matches.append((span_s, span_e, entry["id"], span_e - span_s))
    return matches


# Type-priority for match resolution.  Higher number = higher priority
# (claims frames first).  Within the same type-priority level, the LONGER
# match wins.  Rationale: composites are the most semantically specific
# (= rarer in the corpus, more information per occurrence); long-bouts
# are next (they encode a duration-anchored single-behavior chunk);
# fine codes only fire when nothing else matches.  Without this layering
# long-codes are entirely absorbed by n-grams (a single Long-Rear ≥1.6s
# Rear segment is shorter than any 3-gram containing that Rear, so
# pure longest-match-wins hides every long-code).
_TYPE_PRIORITY: dict[str, int] = {
    "n5":   5,
    "n4":   4,
    "p3":   3,
    "d3":   3,
    "long": 2,
    "fine": 1,
    "sentinel": 0,
}


def _entry_type_by_id(
    codebook_entries: list[dict[str, Any]],
) -> dict[int, str]:
    return {int(e["id"]): str(e["type"]) for e in codebook_entries}


def _entry_specificity_by_id(
    codebook_entries: list[dict[str, Any]],
) -> dict[int, int]:
    """Per-entry sub-priority within the same type.

    For long-type entries this is ``min_csv_frames`` (more specific →
    higher).  For everything else it's 0 (no within-type ordering — fine
    codes share priority, n-grams already tie-break on span length).
    """
    out: dict[int, int] = {}
    for e in codebook_entries:
        if e["type"] == "long":
            out[int(e["id"])] = int(e.get("min_csv_frames", 0))
        else:
            out[int(e["id"])] = 0
    return out


def _assign_codes_per_csv_frame(
    matches: list[tuple[int, int, int, int]],
    n_csv_frames: int,
    sentinel_id: int,
    type_by_id: dict[int, str],
    specificity_by_id: dict[int, int],
) -> np.ndarray:
    """Resolve overlapping matches via TYPE PRIORITY then specificity then length.

    Sort key (descending priority):
      1. type priority (n5 > n4 > p3=d3 > long > fine > sentinel)
      2. within-type specificity (long: higher min_csv_frames first;
         e.g. Long-Rear ≥4s wins over Long-Rear ≥1.6s on segments ≥4s)
      3. span length (longer match wins ties on the same type)
      4. start position (deterministic tiebreaker)

    First match in that order claims frames; subsequent matches only claim
    unassigned frames.  Frames not covered by any match get the sentinel
    code.  Returns an int32 array of length n_csv_frames.
    """
    out = np.full(n_csv_frames, sentinel_id, dtype=np.int32)
    if not matches:
        return out

    def _key(m):
        s, e, code_id, length = m
        prio = _TYPE_PRIORITY.get(type_by_id.get(code_id, "fine"), 1)
        spec = specificity_by_id.get(code_id, 0)
        return (-prio, -spec, -length, s)

    matches_sorted = sorted(matches, key=_key)
    assigned = np.zeros(n_csv_frames, dtype=bool)
    for s, e, code_id, _ in matches_sorted:
        e_clip = min(e, n_csv_frames)
        if s >= e_clip:
            continue
        sub_mask = ~assigned[s:e_clip]
        if sub_mask.any():
            out_slice = out[s:e_clip]
            out_slice[sub_mask] = code_id
            out[s:e_clip] = out_slice
            assigned[s:e_clip] |= sub_mask
    return out


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_codes(
    pattern_yaml: str | Path,
    csv_path: str | Path,
    splits_json: str | Path,
    n_h5_frames: int = 250,
    csv_to_h5_subsample: int = 2,
) -> dict[str, Any]:
    """Run the segmenter and return arrays + provenance.

    Returns a dict ready for ``np.savez_compressed(**)`` plus extra keys
    ``num_codes`` (int) and ``code_source`` (str) which need to be unpacked
    by the caller.
    """
    cb = load_codebook(pattern_yaml)
    entries = cb["codebook"]
    sentinel_id = cb["_sentinel_id"]
    num_codes = cb["_num_codes"]

    chains = build_segment_chains(csv_path)

    with open(splits_json) as f:
        splits = json.load(f)
    train_indices = np.asarray(
        splits["balanced"]["train_indices"], dtype=np.int64,
    )
    test_indices = np.asarray(
        splits["balanced"]["test_indices"], dtype=np.int64,
    )

    n_csv_frames = n_h5_frames * csv_to_h5_subsample

    type_by_id = _entry_type_by_id(entries)
    spec_by_id = _entry_specificity_by_id(entries)

    def codes_for_clip_subset(indices: np.ndarray) -> np.ndarray:
        out = np.full((len(indices), n_h5_frames), sentinel_id, dtype=np.int32)
        for i, clip_id in enumerate(indices):
            cid = int(clip_id)
            chain = chains.get(cid, [])
            matches = _enumerate_matches(chain, entries)
            csv_codes = _assign_codes_per_csv_frame(
                matches, n_csv_frames, sentinel_id, type_by_id, spec_by_id,
            )
            # Downsample CSV → H5 frame rate.  Use the code at the first
            # CSV frame of each H5 frame (frame i covers CSV frames
            # [i*subsample, (i+1)*subsample]).
            for h in range(n_h5_frames):
                c0 = h * csv_to_h5_subsample
                out[i, h] = csv_codes[c0]
        return out

    train_codes = codes_for_clip_subset(train_indices)
    test_codes = codes_for_clip_subset(test_indices)

    # Provenance: count instances and frames per code.
    all_codes = np.concatenate([train_codes, test_codes], axis=0)
    code_frame_counts = {
        int(c): int(np.sum(all_codes == c)) for c in range(num_codes)
    }
    # Instance count: contiguous runs per code per clip.
    code_instance_counts: dict[int, int] = defaultdict(int)
    for row in all_codes:
        cur = -1
        for c in row:
            if int(c) != cur:
                code_instance_counts[int(c)] += 1
                cur = int(c)
    # Strip the +1 (every clip starts a new "run" of UNL or some code).
    code_instance_counts = {k: v for k, v in code_instance_counts.items()}

    per_code = []
    for e in entries:
        cid = e["id"]
        per_code.append({
            "id": cid,
            "name": e["name"],
            "type": e["type"],
            "tier": e["tier"],
            "instances": int(code_instance_counts.get(cid, 0)),
            "frames": int(code_frame_counts.get(cid, 0)),
        })

    provenance = {
        "pattern_yaml": str(pattern_yaml),
        "priority_rule": cb.get("priority_rule", "longest_match_wins"),
        "n_h5_frames": int(n_h5_frames),
        "csv_to_h5_subsample": int(csv_to_h5_subsample),
        "per_code": per_code,
    }

    return {
        "train_codes": train_codes,
        "test_codes": test_codes,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "num_codes": int(num_codes),
        "code_source": "designed",
        "provenance": provenance,
    }


def save_codes(payload: dict[str, Any], output: str | Path) -> None:
    """Save in CodesPayload-compatible npz format.

    The loader (`_load_kpms_npz` in `code_sources.py`) reads:
      - train_codes / test_codes      (int)
      - train_indices / test_indices  (int)
      - num_states                    (codebook size — IMPORTANT, NOT num_codes)
      - is_merged                     (optional flag; we set False)
      - model_type                    (provenance string)
    Everything else is informational.
    """
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
        model_type=np.array("designed_multi_scale"),
        provenance_json=np.array(json.dumps(payload["provenance"])),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--pattern_yaml", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument(
        "--csv", type=str,
        default=str(get_repo_root() / "data/rodent/rodent_labels.csv"),
    )
    ap.add_argument(
        "--splits", type=str,
        default=str(get_repo_root() / "data/rodent/rodent_balanced_splits.json"),
    )
    ap.add_argument("--n_frames", type=int, default=250)
    ap.add_argument("--csv_to_h5_subsample", type=int, default=2)
    args = ap.parse_args(argv)

    payload = build_codes(
        pattern_yaml=args.pattern_yaml,
        csv_path=args.csv,
        splits_json=args.splits,
        n_h5_frames=args.n_frames,
        csv_to_h5_subsample=args.csv_to_h5_subsample,
    )
    save_codes(payload, args.output)

    print(f"\nSaved {args.output}")
    print(f"  shapes: train_codes={payload['train_codes'].shape}, "
          f"test_codes={payload['test_codes'].shape}")
    print(f"  num_codes: {payload['num_codes']}")
    print(f"\nPer-code coverage (instances = contiguous runs in corpus, "
          f"frames = total covered frames):")
    print(f"  {'id':>3} {'tier':>4} {'type':>8} {'name':<32} "
          f"{'inst':>5} {'frames':>7}")
    for p in payload["provenance"]["per_code"]:
        print(f"  {p['id']:>3} {p['tier']:>4} {p['type']:>8} "
              f"{p['name']:<32} {p['instances']:>5} {p['frames']:>7}")


if __name__ == "__main__":
    main()
