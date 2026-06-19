"""Auto-generate a designed-segmenter codebook YAML from a rule.

Inputs (CLI flags):
    --orders            list[int]  default 1 2 3      orders to include
    --min_component     int        default 30         per-component frame floor
                                                     (H5 frames -- CSV is 2x)
    --min_pattern       int        default 10         valid-instance floor for
                                                     a pattern to be included
    --csv               path       default rodent_labels.csv
    --output            path       output YAML

For order=1 (fine codes) all four behavior labels are always emitted.
For orders>=2 every ordered pattern over the four labels (with the
constraint that adjacent positions are distinct; consecutive runs in the
CSV are by construction distinct labels) is enumerated, valid instances
are counted (every component segment >= min_component H5 frames), and
patterns with >= min_pattern valid instances are written into the
codebook.

The output YAML matches the schema c2a.codegen.designed_segmenter
expects (fields: id, name, type, tier, color; label for fine; labels
for n-grams). Type strings used:
    n1 -> "fine"  (so existing renderer code keeps working)
    n2 -> "n2"
    n3 -> "n3"
    n4 -> "n4"
    n5 -> "n5"
Tier auto-assignment: T1 if valid>=50, T2 if 20..49, T3 if 10..19.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import yaml

from c2a.paths import get_repo_root


BEHAVIORS = ["Walk", "Rear", "Turn", "Immobile"]
SUBSAMPLE = 2  # csv_to_h5_subsample (CSV frame = 2 H5 frames)

# Colors keyed by output type-name.
TYPE_COLOR = {
    "fine": [0.70, 0.70, 0.70],   # pale gray
    "n2":   [0.50, 0.85, 0.50],   # lime
    "n3":   [0.20, 0.65, 0.30],   # green (same scale family as v1 p3)
    "n4":   [0.20, 0.30, 0.80],   # blue
    "n5":   [0.55, 0.30, 0.80],   # purple
}
UNL_COLOR = [0.30, 0.30, 0.30]    # dark gray
ORDER_TO_TYPE = {1: "fine", 2: "n2", 3: "n3", 4: "n4", 5: "n5"}


def load_runs(csv_path: Path) -> dict[str, list[tuple[str, int, int]]]:
    """clip_id -> sorted list of (label, start_frame, end_frame_exclusive),
    behavior track only, CSV-frame units."""
    runs: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("Track") != "behavior":
                continue
            runs[row["Video"]].append((
                row["Label"],
                int(row["Start_Frame"]),
                int(row["End_Frame"]),
            ))
    for k in runs:
        runs[k].sort(key=lambda t: t[1])
    return runs


def enumerate_patterns(order: int) -> list[tuple[str, ...]]:
    """All ordered length-`order` label patterns with the constraint that
    adjacent positions are distinct."""
    out = []
    for combo in product(BEHAVIORS, repeat=order):
        if all(combo[i] != combo[i + 1] for i in range(order - 1)):
            out.append(combo)
    return out


def count_valid(
    runs_by_clip: dict[str, list[tuple[str, int, int]]],
    pattern: tuple[str, ...],
    min_component_csv: int,
) -> int:
    """Count instances of `pattern` where every component segment is at
    least `min_component_csv` CSV frames long."""
    plen = len(pattern)
    n = 0
    for runs in runs_by_clip.values():
        for i in range(len(runs) - plen + 1):
            window = runs[i:i + plen]
            if any(window[k][0] != pattern[k] for k in range(plen)):
                continue
            if all((e - s) >= min_component_csv for _, s, e in window):
                n += 1
    return n


def tier_of(valid: int) -> str:
    if valid >= 50:
        return "T1"
    if valid >= 20:
        return "T2"
    return "T3"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3],
                   help="n-gram orders to include (1=single labels, 2=2-grams, "
                        "3=3-grams, etc.)")
    p.add_argument("--min_component", type=int, default=30,
                   help="per-component frame floor in H5 frames (CSV is 2x)")
    p.add_argument("--min_pattern", type=int, default=10,
                   help="min valid instances for a pattern to be included")
    p.add_argument("--csv", type=Path,
                   default=Path("data/rodent/rodent_labels.csv"))
    p.add_argument("--output", type=Path, required=True,
                   help="path to write the YAML codebook")
    p.add_argument("--name", type=str, default=None,
                   help="codebook name (header comment); defaults to "
                        "the output stem")
    args = p.parse_args()

    repo = Path(get_repo_root())
    csv_path = args.csv if args.csv.is_absolute() else repo / args.csv
    out_path = args.output if args.output.is_absolute() else repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {csv_path}")
    runs = load_runs(csv_path)
    print(f"  clips: {len(runs)}")

    min_component_csv = args.min_component * SUBSAMPLE
    print(f"Rule: min_component={args.min_component} H5 frames "
          f"(={min_component_csv} CSV frames), "
          f"min_pattern_instances={args.min_pattern}")
    print(f"Orders to include: {args.orders}")

    entries: list[dict] = []
    next_id = 0

    # Order 1 always emits all four fine codes (single labels do not need
    # an inclusion filter -- they are the fallback).
    if 1 in args.orders:
        for label in BEHAVIORS:
            entries.append({
                "id": next_id,
                "name": f"fine.{label}",
                "type": "fine",
                "tier": "T1",
                "color": list(TYPE_COLOR["fine"]),
                "label": label,
            })
            next_id += 1

    # Orders >= 2: enumerate, count valid, filter.
    for order in sorted(args.orders):
        if order < 2:
            continue
        if order not in ORDER_TO_TYPE:
            raise ValueError(f"order {order} not supported (max 5)")
        type_name = ORDER_TO_TYPE[order]
        ranked: list[tuple[tuple[str, ...], int]] = []
        for pat in enumerate_patterns(order):
            v = count_valid(runs, pat, min_component_csv)
            if v >= args.min_pattern:
                ranked.append((pat, v))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        for pat, valid in ranked:
            # Abbreviate label tokens in the name (Immobile -> Imm) like v1.
            tok = lambda L: "Imm" if L == "Immobile" else L
            short = "_".join(tok(L) for L in pat)
            entries.append({
                "id": next_id,
                "name": f"{type_name}.{short}",
                "type": type_name,
                "tier": tier_of(valid),
                "color": list(TYPE_COLOR[type_name]),
                "labels": list(pat),
            })
            next_id += 1

    # UNL sentinel always last.
    entries.append({
        "id": next_id,
        "name": "UNL",
        "type": "sentinel",
        "tier": "UNL",
        "color": list(UNL_COLOR),
    })

    n_active = len(entries) - 1
    header = f"""# =============================================================================
# Auto-generated multi-scale codebook (K={n_active} + UNL = {n_active + 1} slots)
# =============================================================================
# Built by c2a.codegen.auto_codebook_generator.
# Source CSV : {csv_path.relative_to(repo) if csv_path.is_relative_to(repo) else csv_path}
# Rule       : ngram_orders={args.orders}
#              min_component_frames={args.min_component} (H5 frames; CSV is 2x)
#              min_pattern_instances={args.min_pattern}
# Tier       : T1 if valid>=50, T2 if 20..49, T3 if 10..19 (auto-assigned).

priority_rule: type_then_length

codebook:
"""

    with open(out_path, "w") as f:
        f.write(header)
        # Write entries grouped by tier-type for readability.
        for e in entries:
            # Dump inline-style for one-line entries (matches v1/v2 style).
            payload = {k: e[k] for k in
                       ("id", "name", "type", "tier", "color")}
            if "label" in e:
                payload["label"] = e["label"]
            if "labels" in e:
                payload["labels"] = e["labels"]
            f.write("  - " + yaml.safe_dump(
                payload, default_flow_style=True, sort_keys=False).rstrip()
                + "\n")

    print(f"\nWrote {out_path}")
    print(f"  total codes: {n_active + 1}  (active: {n_active}, +1 sentinel)")
    by_type = Counter(e["type"] for e in entries)
    for k in ("fine", "n2", "n3", "n4", "n5", "sentinel"):
        if by_type[k]:
            print(f"    {k}: {by_type[k]}")


if __name__ == "__main__":
    main()
