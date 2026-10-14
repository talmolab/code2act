"""Label chain-tree runs and select the demo's verified sequences.

Runs in the project venv (owns the TopoMimic labeler). A chain *passes* from
a start pose when the labeler reads every 2.5 s phase (settled window, last
40%) as the commanded behavior. The demo only offers chains that passed, with
every prefix, from every chosen start pose in every run.

Subcommands::

    # success.json for a chain_tree run (or for the web app's TS replay dump)
    C2A_ROOT=$PWD python -m c2a.webdemo.chain_select label clean
    C2A_ROOT=$PWD python -m c2a.webdemo.chain_select label-ts ts_check
    # choose K start poses maximizing the verified prefix tree over runs (AND)
    C2A_ROOT=$PWD python -m c2a.webdemo.chain_select pick clean noise1 noise2 --k 10
    # final check on fresh runs; prunes failures, writes verified_chains.json
    C2A_ROOT=$PWD python -m c2a.webdemo.chain_select verify clean noise1 noise2 noise3 noise4 ts_check
    # physical criterion on a --phys chain_tree run (or a TS --phys dump)
    python -m c2a.webdemo.chain_select phys phys_clean
    # per start pose: each clip gets its own tree (what the demo uses)
    C2A_ROOT=$PWD python -m c2a.webdemo.chain_select verify-clip clean noise1 ... --held noise11 noise12
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from c2a.paths import get_repo_root

TREE_DIR = get_repo_root() / "outputs/webdemo/chain_tree"
VERIFIED = Path(__file__).with_name("verified_chains.json")
MAX_DEPTH = 5


def _behavior(symbol: str) -> str:
    return symbol.split("_")[0]


def _label_tree_file(path: str) -> tuple[dict, dict]:
    from c2a.experiments.shared.topomimic_labeler import label_rollout_body

    def reads(window: np.ndarray, symbol: str) -> bool:
        return label_rollout_body(window, src_fps=100.0).behavior == _behavior(symbol)

    d = np.load(path)
    nodes = {k.rsplit("|", 1)[0] for k in d.files}
    term, via = {}, {}
    for node in nodes:
        cur, base = node.split(">")[-1], d[f"{node}|base"]
        term[node] = [reads(np.concatenate([base[b], d[f"{node}|term"][b]]), cur)
                      for b in range(len(base))]
        if ">" in node:
            parent = node.rsplit(">", 1)[0]
            pfile = d if f"{parent}|base" in d.files else np.load(
                str(Path(path).with_name(f"sub_{parent}.npz")))
            pbase = pfile[f"{parent}|base"]
            via[node] = [reads(np.concatenate([pbase[b], d[f"{node}|tail"][b]]),
                               parent.split(">")[-1]) for b in range(len(pbase))]
    return term, via


def label(run: str) -> None:
    """success[chain][clip]: the chain's own last phase (sequence ends there)
    and each earlier phase as followed by its actual successor."""
    rdir = TREE_DIR / run
    clips = json.loads((rdir / "config.json").read_text())["clips"]
    files = sorted(glob.glob(str(rdir / "sub_*.npz")))
    term, via = {}, {}
    with ProcessPoolExecutor(min(48, len(files))) as ex:
        for t, v in ex.map(_label_tree_file, files):
            term.update(t)
            via.update(v)
    succ = {}
    for node in term:
        parts, ok = node.split(">"), np.array(term[node])
        for i in range(2, len(parts) + 1):
            ok &= np.array(via[">".join(parts[:i])])
        succ[node] = {str(c): bool(o) for c, o in zip(clips, ok)}
    (rdir / "success.json").write_text(json.dumps(succ))
    print(f"{run}: {len(succ)} chains x {len(clips)} clips")


#: Physical failure: the body "goes crazy" or collapses. Calibrated on rendered
#: examples from the settled windows of 10,260 chain x clip runs (2026-10-14).
CRAZY_SPEED = 2.5  # m/s root speed over one control step (normal motion < ~1)
SINK_Z = -0.010  # m root height, mean over SINK_STEPS (normal still: > -0.005)
STILL_JOINT = 0.1  # rad/s mean joint speed, mean over SINK_STEPS
SINK_STEPS = 100  # 1 s
FLIP_UP, FLIP_Z, FLIP_STEPS = -0.5, 0.03, 25  # upside down near the floor


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.pad(x, [(0, 0), (1, 0)]), axis=1)
    return (c[:, n:] - c[:, :-n]) / n


def physical_failure(sig: np.ndarray) -> np.ndarray:
    """Per body, which failure (0 none, 1 crazy, 2 sinking, 3 flipped) a
    trajectory ``[B, T, 4]`` of chain_tree ``PHYS_CHANNELS`` shows first-checked."""
    speed, z, up, joint = (sig[..., i].astype(np.float64) for i in range(4))
    crazy = (speed > CRAZY_SPEED).any(1) | ~np.isfinite(sig).all((1, 2))
    sink = ((_rolling_mean(z, SINK_STEPS) < SINK_Z)
            & (_rolling_mean(joint, SINK_STEPS) < STILL_JOINT)).any(1)
    flip = (_rolling_mean(((up < FLIP_UP) & (z < FLIP_Z)).astype(float), FLIP_STEPS) == 1).any(1)
    return np.select([crazy, sink, flip], [1, 2, 3], 0)


def _phys_trajectories(run: str) -> dict[str, np.ndarray]:
    """chain -> full trajectory [B, T, 4] from a --phys chain_tree run."""
    seg: dict[str, np.ndarray] = {}
    for f in glob.glob(str(TREE_DIR / run / "sub_*.npz")):
        d = np.load(f)
        seg.update({k: d[k] for k in d.files})
    chains = sorted({k.rsplit("|", 1)[0] for k in seg}, key=lambda c: (c.count(">"), c))
    out, body = {}, {}
    for c in chains:  # body = trajectory up to the end of c's main part
        if ">" in c:
            body[c] = np.concatenate([body[c.rsplit(">", 1)[0]], seg[f"{c}|tail"],
                                      seg[f"{c}|main"]], 1)
        else:
            body[c] = seg[f"{c}|main"]
        out[c] = np.concatenate([body[c], seg[f"{c}|term"]], 1)
    return out


def phys(run: str) -> None:
    """success.json from the physical criterion (no labeler), plus reasons."""
    rdir = TREE_DIR / run
    clips = json.loads((rdir / "config.json").read_text())["clips"]
    succ, reasons, vmax = {}, {}, {}
    for c, traj in _phys_trajectories(run).items():
        why = physical_failure(traj)
        succ[c] = {str(k): bool(w == 0) for k, w in zip(clips, why)}
        reasons[c] = {str(k): int(w) for k, w in zip(clips, why)}
        vmax[c] = {str(k): round(float(v), 3) for k, v in zip(clips, traj[..., 0].max(1))}
    (rdir / "success.json").write_text(json.dumps(succ))
    (rdir / "reasons.json").write_text(json.dumps(reasons))
    (rdir / "vmax.json").write_text(json.dumps(vmax))  # peak root speed, m/s
    n = np.bincount([w for r in reasons.values() for w in r.values()], minlength=4)
    print(f"{run}: {len(succ)} chains x {len(clips)} clips; ok {n[0]}, crazy {n[1]}, "
          f"sinking {n[2]}, flipped {n[3]}")


def _phys_ts_file(path: str) -> list[tuple[str, int, int]]:
    meta = json.loads(Path(path).read_text())
    sig = np.fromfile(path.replace(".json", ".bin"), np.float32).reshape(-1, 4)
    return [(r["chain"], r["clip"],
             int(physical_failure(sig[None, r["offset"]:r["offset"] + meta.get("hold", 250)
                                      * r["blocks"]])[0]))
            for r in meta["index"]]


def phys_ts(run: str) -> None:
    """success.json (physical criterion) for a PHYS=1 dump of tests/verify_chains.test.ts."""
    rdir = TREE_DIR / run
    succ: dict[str, dict[str, bool]] = {}
    n = np.zeros(4, int)
    with ProcessPoolExecutor(32) as ex:
        for rows in ex.map(_phys_ts_file, sorted(glob.glob(str(rdir / "ts_*.json")))):
            for chain, clip, why in rows:
                succ.setdefault(chain, {})[str(clip)] = why == 0
                n[why] += 1
    (rdir / "success.json").write_text(json.dumps(succ))
    print(f"{run}: ok {n[0]}, crazy {n[1]}, sinking {n[2]}, flipped {n[3]}")


def _label_ts_file(path: str) -> list[tuple[str, int, bool]]:
    from c2a.experiments.shared.topomimic_labeler import label_rollout_body

    meta = json.loads(Path(path).read_text())
    q = np.fromfile(path.replace(".json", ".bin"), np.float32).reshape(-1, meta["nq"])
    out = []
    for r in meta["index"]:
        ok = all(label_rollout_body(q[r["offset"] + 100 * i:r["offset"] + 100 * (i + 1)],
                                    src_fps=100.0).behavior == _behavior(s)
                 for i, s in enumerate(r["chain"].split(">")))
        out.append((r["chain"], r["clip"], ok))
    return out


def label_ts(run: str) -> None:
    """success.json for a dump of the web app's tests/verify_chains.test.ts."""
    rdir = TREE_DIR / run
    succ: dict[str, dict[str, bool]] = {}
    with ProcessPoolExecutor(32) as ex:
        for rows in ex.map(_label_ts_file, sorted(glob.glob(str(rdir / "ts_*.json")))):
            for chain, clip, ok in rows:
                succ.setdefault(chain, {})[str(clip)] = ok
    (rdir / "success.json").write_text(json.dumps(succ))
    print(f"{run}: {len(succ)} chains")


def _combined(runs: list[str], clips: list[int] | None = None):
    """AND over runs: chains x clips bool matrix (clips present in every run)."""
    succ = [json.loads((TREE_DIR / r / "success.json").read_text()) for r in runs]
    chains = sorted(set(succ[0]).intersection(*succ[1:]), key=lambda c: (c.count(">"), c))
    if clips is None:
        clips = sorted(set.intersection(*[set(map(int, s[chains[0]])) for s in succ]))
    S = np.array([[all(s[c][str(k)] for s in succ) for k in clips] for c in chains])
    return chains, clips, S


def _prefix_closed(chains: list[str], ok: np.ndarray) -> np.ndarray:
    pos = {c: i for i, c in enumerate(chains)}  # chains sorted by depth
    ok = ok.copy()
    for i, c in enumerate(chains):
        if ">" in c:
            ok[i] &= ok[pos[c.rsplit(">", 1)[0]]]
    return ok


def pick(runs: list[str], k: int, exclude: list[int], trials: int = 200) -> None:
    """Greedy + swap search for the K clips with the largest verified tree,
    weighted toward full-length chains. ``exclude``: start poses never picked."""
    chains, clips, S = _combined(runs)
    keep_cols = [i for i, c in enumerate(clips) if c not in exclude]
    clips, S = [clips[i] for i in keep_cols], S[:, keep_cols]
    depth = np.array([c.count(">") + 1 for c in chains])

    def score(cols: list[int]) -> int:
        ok = _prefix_closed(chains, S[:, cols].all(1))
        return int((ok & (depth == MAX_DEPTH)).sum() * 10 + ok.sum())

    rng, best = random.Random(0), None
    for trial in range(trials):
        cols = [rng.randrange(len(clips))] if trial else []
        while len(cols) < k:
            cols.append(max((m for m in range(len(clips)) if m not in cols),
                            key=lambda m: (score(cols + [m]), rng.random())))
        improved = True
        while improved:
            improved, base = False, score(cols)
            for i in range(k):
                for m in range(len(clips)):
                    cand = cols[:i] + [m] + cols[i + 1:]
                    if m not in cols and (sc := score(cand)) > base:
                        cols, base, improved = cand, sc, True
        if best is None or score(cols) > best[0]:
            best = (score(cols), sorted(cols))
    cols = best[1]
    ok = _prefix_closed(chains, S[:, cols].all(1))
    picked = {"clips": [clips[m] for m in cols], "chains": [c for c, o in zip(chains, ok) if o],
              "runs": runs}
    (TREE_DIR / f"pick_{k}.json").write_text(json.dumps(picked))
    print(f"clips {picked['clips']}: {len(picked['chains'])} chains, by depth "
          f"{np.bincount(depth[ok], minlength=MAX_DEPTH + 1)[1:].tolist()}")


def verify(runs: list[str], k: int) -> None:
    """Re-check the picked chains on further runs; keep only chains (with all
    prefixes) that pass every run from every picked clip."""
    picked = json.loads((TREE_DIR / f"pick_{k}.json").read_text())
    chains, clips, S = _combined(runs, picked["clips"])
    keep = set(picked["chains"])
    ok = _prefix_closed(chains, S.all(1) & np.array([c in keep for c in chains]))
    final = [c for c, o in zip(chains, ok) if o]
    for c in sorted(keep - set(final)):
        print("  dropped", c)
    VERIFIED.write_text(json.dumps({"clips": clips, "chains": final, "runs": runs,
                                    "hold": 250, "max_blocks": MAX_DEPTH}, indent=1) + "\n")
    print(f"{len(final)} of {len(keep)} chains verified on {len(runs)} runs -> {VERIFIED.name}")


def verify_clip(runs: list[str], held: list[str], k: int, clips: list[int] | None = None) -> None:
    """Per start pose: a clip offers a chain when the chain (with all prefixes)
    passed every run from that clip. ``held``: runs only used to estimate the
    failure rate. A chain missing from a selection run (e.g. a TS replay of a
    subset) counts as failed; a held run is scored on the pairs it covers."""
    clips = clips or json.loads((TREE_DIR / f"pick_{k}.json").read_text())["clips"]
    hold = json.loads((TREE_DIR / runs[0] / "config.json").read_text()).get("hold", 250)
    succ = {r: json.loads((TREE_DIR / r / "success.json").read_text()) for r in runs + held}
    chains = sorted(succ[runs[0]], key=lambda c: (c.count(">"), c))

    def passed(r: str, c: str, clip: int) -> bool:
        return succ[r].get(c, {}).get(str(clip), False)

    trees, fails, total = {}, 0, 0
    for clip in clips:
        ok: list[str] = []
        for c in chains:
            parent = c.rsplit(">", 1)[0] if ">" in c else None
            if (parent is None or parent in ok) and all(passed(r, c, clip) for r in runs):
                ok.append(c)
        trees[str(clip)] = ok
        # a held run may cover a sample of the pairs (e.g. a TS replay): count what it has
        tried = [(r, c) for r in held for c in ok if str(clip) in succ[r].get(c, {})]
        f = sum(not passed(r, c, clip) for r, c in tried)
        fails, total = fails + f, total + len(tried)
        print(f"clip {clip}: {len(ok)} chains, by depth "
              f"{np.bincount([c.count('>') + 1 for c in ok], minlength=MAX_DEPTH + 1)[1:].tolist()}"
              + (f", held-out fails {f}/{len(tried)}" if held else ""))
    if held:
        print(f"held-out: {fails}/{total} chain x clip runs failed ({fails / max(total, 1):.2%})")
    VERIFIED.write_text(json.dumps({"clips": clips, "chains": trees, "runs": runs, "held": held,
                                    "held_fails": [fails, total], "hold": hold,
                                    "max_blocks": MAX_DEPTH}, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("label").add_argument("run")
    sub.add_parser("label-ts").add_argument("run")
    sub.add_parser("phys").add_argument("run")
    sub.add_parser("phys-ts").add_argument("run")
    for name in ("pick", "verify"):
        p = sub.add_parser(name)
        p.add_argument("runs", nargs="+")
        p.add_argument("--k", type=int, default=10)
        p.add_argument("--exclude", default="", help="pick: comma-separated start poses to skip")
    p = sub.add_parser("verify-clip")
    p.add_argument("runs", nargs="+")
    p.add_argument("--held", nargs="*", default=[])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--clips", default="", help="comma-separated start poses (default: pick_K.json)")
    args = ap.parse_args()
    if args.cmd == "label":
        label(args.run)
    elif args.cmd == "phys":
        phys(args.run)
    elif args.cmd == "phys-ts":
        phys_ts(args.run)
    elif args.cmd == "label-ts":
        label_ts(args.run)
    elif args.cmd == "pick":
        pick(args.runs, args.k, [int(c) for c in args.exclude.split(",") if c])
    elif args.cmd == "verify-clip":
        verify_clip(args.runs, args.held, args.k, [int(c) for c in args.clips.split(",") if c])
    else:
        verify(args.runs, args.k)


if __name__ == "__main__":
    main()
