"""Exhaustive chain tree for the demo's verified sequences (MuJoCo C + ONNX).

Runs in ``.venv-webdemo``. Simulates every chain of up to ``--depth`` palette
symbols (2.5 s holds) over a candidate switch graph, from each start pose.
A chain shares its prefix's simulation: the state is copied bit-exactly at the
step where the 5-step code window first sees the next symbol, so every leaf
matches a straight ``c_sim.rollout`` of that chain exactly.

Per tree node (a chain prefix) it saves, for every body:

* ``<chain>|base``: qpos of the last block, steps [150, 245) (settled window
  minus its tail);
* ``<chain>|term``: steps [245, 250) when the sequence ends here (window padded
  with the last symbol);
* ``<chain>|tail``: steps [245, 250) of the *parent's* block when this node's
  symbol follows.

With ``--phys`` it instead saves per-step body signals (``PHYS_CHANNELS``) for
the whole block, so a chain's full trajectory is ``main`` of its first block,
then ``tail`` + ``main`` of each later block, then ``term``: keys
``<chain>|main`` (steps [0, 245)), ``<chain>|term`` and ``<chain>|tail``.

``c2a.webdemo.chain_select`` labels and selects from these.

``--search`` instead grows one tree per start pose directly: each clip is a
batch of ``--replicas`` bodies (replica 0 clean, the rest with action noise
seeded by clip, chain and segment, so a prefix always replays identically).
A chain is kept if no replica loses control (``chain_select.physical_failure``)
and is only extended then, over the switches not in
``chain_select.WRONG_BEHAVIOR``. It writes ``success.json`` for the chains it
tried, for ``chain_select verify-clip``.

Usage::

    .venv-webdemo/bin/python -m c2a.webdemo.chain_tree RUN [--noise-seed 1] [--clips 1,10,...]
        [--phys] [--all-switches] [--hold 400] [--palette Walk_L,Rear_L,...]
    .venv-webdemo/bin/python -m c2a.webdemo.chain_tree RUN --search --hold 400 --depth 8 --clips ...
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import mujoco
import numpy as np

from c2a.webdemo.build_onnx import OUT
from c2a.webdemo.c_sim import Body, Controller, ObsIndex

TREE_DIR = OUT / "chain_tree"
HOLD, STACK, SETTLE0 = 250, 5, 150
BRANCH = HOLD - STACK  # first step whose window sees the next block
PALETTE = ["Walk_L", "Walk_R", "Rear_L", "Rear_R", "Turn_R", "Immobile"]
#: Switches left out of the search: the fragile ones in c2a/webdemo/README.md.
FRAGILE = {("Walk_L", "Rear_L"), ("Walk_L", "Rear_R"), ("Walk_R", "Rear_L"),
           ("Walk_R", "Rear_R"), ("Rear_R", "Walk_L"), ("Immobile", "Turn_R"),
           ("Walk_L", "Turn_R"), ("Walk_R", "Turn_R"), ("Turn_R", "Rear_L")}
ALL_EDGES = {(a, b) for a in PALETTE for b in PALETTE if a != b}
EDGES = ALL_EDGES - FRAGILE  # replaced by ALL_EDGES with --all-switches
NOISE_STD = 1e-4  # action noise of the replicas; stands in for float drift
#: --phys signals per control step: root speed (m/s) and mean joint speed
#: (rad/s), both from the qpos change over the step; root height (m); z
#: component of the root frame's up axis.
PHYS_CHANNELS = ("speed", "z", "up", "joint_speed")


def _configure(hold: int, palette: list[str]) -> None:
    """Set the block length and palette (module globals) in this process."""
    global HOLD, BRANCH, PALETTE
    HOLD, BRANCH, PALETTE = hold, hold - STACK, palette


def successors(a: str) -> list[str]:
    return [b for b in PALETTE if (a, b) in EDGES]


class Sim:
    """A batch of bodies (one per clip) that can be snapshotted and restored."""

    def __init__(self, clips: list[int], noise_seed: int, phys: bool = False):
        self.phys = phys
        meta = json.loads((OUT / "symbols.json").read_text())
        self.names = [s["name"] for s in meta["symbols"]]
        starts = np.asarray(json.loads((OUT / "starts.json").read_text())["qpos"])[clips]
        self.m = mujoco.MjModel.from_xml_path(str(OUT / "rodent_c2a.xml"))
        self.ctrl = Controller(str(OUT / "controller.onnx"), threads=1)
        idx = ObsIndex.from_model(self.m)
        self.bodies = [Body(self.m, idx, q) for q in starts]
        self.hidden = self.ctrl.zeros(len(self.bodies))
        self.rng = np.random.RandomState(noise_seed) if noise_seed else None
        self.mask = None  # [B, 1] noise scale per body (None: all bodies)

    def snapshot(self):
        state = []
        for b in self.bodies:
            d = mujoco.MjData(self.m)
            mujoco.mj_copyData(d, self.m, b.d)
            state.append((d, b.action.copy(), b.prev_action.copy()))
        rng = None if self.rng is None else self.rng.get_state()
        return state, [h.copy() for h in self.hidden], rng

    def restore(self, snap) -> None:
        state, hidden, rng = snap
        for b, (d, a, pa) in zip(self.bodies, state):
            mujoco.mj_copyData(b.d, self.m, d)
            b.action, b.prev_action = a.copy(), pa.copy()
        self.hidden = [h.copy() for h in hidden]
        if rng is not None:
            self.rng.set_state(rng)

    def run(self, windows: list[tuple[str, ...]]) -> np.ndarray:
        """Step through per-step code windows; returns qpos ``[B, T, nq]``, or
        with ``phys`` the ``PHYS_CHANNELS`` signals ``[B, T, 4]``."""
        width = len(PHYS_CHANNELS) if self.phys else self.m.nq
        q = np.zeros((len(self.bodies), len(windows), width), np.float32)
        for t, w in enumerate(windows):
            codes = np.repeat([[self.names.index(s) for s in w]], len(self.bodies), 0)
            proprio = np.stack([b.obs() for b in self.bodies])
            a, self.hidden, _ = self.ctrl(codes, proprio, self.hidden)
            if self.rng is not None:
                noise = self.rng.normal(0, NOISE_STD, a.shape).astype(np.float32)
                a = a + (noise if self.mask is None else noise * self.mask)
            for k, b in enumerate(self.bodies):
                if not self.phys:
                    q[k, t] = b.d.qpos
                    b.step(a[k])
                    continue
                before = b.d.qpos.copy()
                b.step(a[k])
                qp = b.d.qpos
                q[k, t] = (np.linalg.norm(qp[:3] - before[:3]) * 100, qp[2],
                           1 - 2 * (qp[4] ** 2 + qp[5] ** 2),
                           np.abs(qp[7:] - before[7:]).mean() * 100)
        return q


def window(cur: str, nxt: str | None, t: int) -> tuple[str, ...]:
    """Code window at step ``t`` of a ``cur`` block followed by ``nxt`` (None: end)."""
    return tuple(cur if t + j < HOLD or nxt is None else nxt for j in range(STACK))


def subtree(job: tuple) -> str:
    """Simulate the one-block chain ``first`` (``second`` None) or the whole
    subtree under ``first > second``, and save its nodes."""
    first, second, depth, clips, seed, run, phys, edges, hold, palette = job
    _configure(hold, palette)
    sim, out = Sim(clips, seed, phys), {}
    if phys:
        return _subtree_phys(sim, first, second, depth, run, edges)
    if second is None:
        q = sim.run([window(first, None, t) for t in range(HOLD)])
        out[f"{first}|base"], out[f"{first}|term"] = q[:, SETTLE0:BRANCH], q[:, BRANCH:]
        tag = first
    else:
        sim.run([window(first, None, t) for t in range(BRANCH)])

        def expand(chain: list[str], snap) -> None:
            parent, cur = chain[-2], chain[-1]
            sim.restore(snap)
            key = ">".join(chain)
            out[f"{key}|tail"] = sim.run([window(parent, cur, t) for t in range(BRANCH, HOLD)])
            out[f"{key}|base"] = sim.run([window(cur, None, t) for t in range(BRANCH)])[:, SETTLE0:]
            here = sim.snapshot()
            out[f"{key}|term"] = sim.run([window(cur, None, t) for t in range(BRANCH, HOLD)])
            if len(chain) < depth:
                for nxt in successors(cur):
                    expand(chain + [nxt], here)

        expand([first, second], sim.snapshot())
        tag = f"{first}__{second}"
    np.savez(TREE_DIR / run / f"sub_{tag}.npz", **out)
    return f"{tag}: {len(out) // 3} nodes"


def _subtree_phys(sim: Sim, first: str, second: str | None, depth: int, run: str,
                  edges: set) -> str:
    out = {}
    out[f"{first}|main"] = sim.run([window(first, None, t) for t in range(BRANCH)])
    if second is None:
        out[f"{first}|term"] = sim.run([window(first, None, t) for t in range(BRANCH, HOLD)])
        tag = first
    else:

        def expand(chain: list[str], snap) -> None:
            parent, cur = chain[-2], chain[-1]
            sim.restore(snap)
            key = ">".join(chain)
            out[f"{key}|tail"] = sim.run([window(parent, cur, t) for t in range(BRANCH, HOLD)])
            out[f"{key}|main"] = sim.run([window(cur, None, t) for t in range(BRANCH)])
            here = sim.snapshot()
            out[f"{key}|term"] = sim.run([window(cur, None, t) for t in range(BRANCH, HOLD)])
            if len(chain) < depth:
                for nxt in (b for b in PALETTE if (cur, b) in edges):
                    expand(chain + [nxt], here)

        expand([first, second], sim.snapshot())
        tag = f"{first}__{second}"
    np.savez(TREE_DIR / run / f"sub_{tag}.npz", **out)
    return f"{tag}: {len(out) // 3} nodes"


def search(job: tuple) -> dict[str, bool]:
    """Simulate ``prefix`` from one clip with ``reps`` replicas, then grow its
    subtree up to ``depth`` symbols while no replica loses control. Returns
    chain -> passed for every chain tried (prefix nodes included)."""
    from c2a.webdemo.chain_select import physical_failure

    clip, prefix, depth, hold, palette, edges, reps = job
    _configure(hold, palette)
    sim, out = Sim([clip] * reps, noise_seed=1, phys=True), {}
    sim.mask = np.ones((reps, 1), np.float32)
    sim.mask[0] = 0

    def seg(key: str, name: str, windows: list) -> np.ndarray:
        sim.rng = np.random.RandomState(zlib.crc32(f"{clip}|{key}|{name}".encode()))
        return sim.run(windows)

    def block(chain: list[str], nxt: str) -> np.ndarray:  # tail of chain[-1], then nxt's main
        key = ">".join(chain + [nxt])
        return np.concatenate([seg(key, "tail", [window(chain[-1], nxt, t) for t in range(BRANCH, HOLD)]),
                               seg(key, "main", [window(nxt, None, t) for t in range(BRANCH)])], 1)

    def passed(chain: list[str], body: np.ndarray) -> bool:  # the sequence ending here
        key, here = ">".join(chain), sim.snapshot()
        term = seg(key, "term", [window(chain[-1], None, t) for t in range(BRANCH, HOLD)])
        sim.restore(here)
        out[key] = bool((physical_failure(np.concatenate([body, term], 1)) == 0).all())
        return out[key]

    chain, body = [prefix[0]], seg(prefix[0], "main", [window(prefix[0], None, t) for t in range(BRANCH)])
    if not passed(chain, body):
        return out
    for nxt in prefix[1:]:
        body = np.concatenate([body, block(chain, nxt)], 1)
        chain = chain + [nxt]
        if not passed(chain, body):
            return out

    def expand(chain: list[str], body: np.ndarray) -> None:
        if len(chain) >= depth:
            return
        here = sim.snapshot()
        for nxt in (b for b in PALETTE if (chain[-1], b) in edges):
            sim.restore(here)
            b = np.concatenate([body, block(chain, nxt)], 1)
            if passed(chain + [nxt], b):
                expand(chain + [nxt], b)

    expand(chain, body)
    return out


def run_search(args, clips: list[int]) -> None:
    from c2a.webdemo.chain_select import WRONG_BEHAVIOR

    edges = ALL_EDGES - WRONG_BEHAVIOR
    rdir = TREE_DIR / args.run
    (rdir / "config.json").write_text(json.dumps(
        {"search": True, "depth": args.depth, "clips": clips, "replicas": args.replicas,
         "noise_std": NOISE_STD, "edges": sorted(edges), "hold": HOLD, "palette": PALETTE}))
    prefixes = [(a, b, c) for a in PALETTE for b in PALETTE for c in PALETTE
                if (a, b) in edges and (b, c) in edges]
    jobs = [(k, p, args.depth, HOLD, PALETTE, edges, args.replicas) for k in clips for p in prefixes]
    random.Random(0).shuffle(jobs)  # spread the deep subtrees over the workers
    succ: dict[str, dict[str, bool]] = {}
    with ProcessPoolExecutor(min(len(jobs), args.workers)) as ex:
        futs = {ex.submit(search, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            clip = str(futs[f][0])
            for chain, ok in f.result().items():
                prev = succ.setdefault(chain, {}).setdefault(clip, ok)
                assert prev == ok, f"{chain} from clip {clip} replayed differently"
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} jobs, {sum(len(v) for v in succ.values())} chain x clip tried",
                      flush=True)
    (rdir / "success.json").write_text(json.dumps(succ))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="output name under outputs/webdemo/chain_tree/")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--noise-seed", type=int, default=0, help="0: clean rollout")
    ap.add_argument("--clips", default="", help="comma-separated start poses (default: all)")
    ap.add_argument("--phys", action="store_true", help="save per-step body signals instead of qpos")
    ap.add_argument("--all-switches", action="store_true", help="also search the FRAGILE switches")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--search", action="store_true", help="grow per-clip trees with replicas")
    ap.add_argument("--replicas", type=int, default=17, help="--search: bodies per clip (1 clean)")
    ap.add_argument("--hold", type=int, default=HOLD, help="control steps per block")
    ap.add_argument("--palette", default="", help="comma-separated symbols (default: PALETTE); "
                    "implies all switches")
    args = ap.parse_args()
    edges = ALL_EDGES if args.all_switches else EDGES
    if args.palette:
        _configure(args.hold, args.palette.split(","))
        edges = {(a, b) for a in PALETTE for b in PALETTE if a != b}
    _configure(args.hold, PALETTE)
    n_starts = len(json.loads((OUT / "starts.json").read_text())["qpos"])
    clips = [int(c) for c in args.clips.split(",")] if args.clips else list(range(n_starts))
    (TREE_DIR / args.run).mkdir(parents=True, exist_ok=True)
    if args.search:
        return run_search(args, clips)
    (TREE_DIR / args.run / "config.json").write_text(json.dumps(
        {"depth": args.depth, "noise_seed": args.noise_seed, "clips": clips,
         "edges": sorted(edges), "phys": args.phys, "hold": HOLD, "palette": PALETTE}))
    jobs = [(a, None) for a in PALETTE] + [(a, b) for a in PALETTE for b in PALETTE
                                           if (a, b) in edges]
    jobs = [(a, b, args.depth, clips, args.noise_seed, args.run, args.phys, edges, HOLD, PALETTE)
            for a, b in jobs]
    with ProcessPoolExecutor(min(len(jobs), args.workers)) as ex:
        for msg in ex.map(subtree, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
