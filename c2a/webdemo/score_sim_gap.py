"""Score MuJoCo-C rollouts against the MJX website rollouts (GO / NO-GO).

Runs in the project venv (owns the TopoMimic labeler). For each preset it
reports, over the same 48 start poses:

* labeler phase agreement (``run_sequences.phase_agreement``) — median / mean;
* falls: bodies rolled past 45 deg or belly-up for >= 1 s continuously;
* mean root speed.

MJX reference: ``outputs/website/sequences/<preset>.npz``.
C rollouts:    ``outputs/webdemo/sim_gap/<preset>_<convention>.npz``.

Usage::

    C2A_ROOT=$PWD python -m c2a.webdemo.score_sim_gap [--set demo]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from c2a.paths import get_repo_root
from c2a.website.run_sequences import PRESETS, phase_agreement

ROOT = get_repo_root()
GAP = ROOT / "outputs/webdemo/sim_gap"
MJX = ROOT / "outputs/website/sequences"


def down_mask(q: np.ndarray) -> np.ndarray:
    """Rolled past ~45 deg, or belly-up without being reared (``[B, T]``)."""
    w, x, y, z = (q[..., i] for i in (3, 4, 5, 6))
    xz, yz, zz = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    return (np.abs(yz) > 0.7) | ((zz < -0.3) & (np.abs(xz) < 0.7))


def longest_run(mask: np.ndarray) -> np.ndarray:
    best = np.zeros(mask.shape[0], int)
    run = np.zeros(mask.shape[0], int)
    for t in range(mask.shape[1]):
        run = np.where(mask[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def score(q: np.ndarray, seq) -> dict:
    ok = np.isfinite(q).all(axis=(1, 2))
    agree = np.array([phase_agreement(q[b], seq) for b in np.flatnonzero(ok)])
    fell = longest_run(down_mask(q[ok])) >= 100
    speed = np.linalg.norm(np.diff(q[ok][:, :, :2], axis=1), axis=-1).mean(1) * 100
    return {"n": int(ok.sum()), "nonfinite": int((~ok).sum()),
            "agree_median": float(np.median(agree)), "agree_mean": float(agree.mean()),
            "fell_1s": float(fell.mean()), "speed": float(np.median(speed))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=["website", "demo"], default="website")
    args = ap.parse_args()
    rows = {}
    if args.set == "website":
        for key, (_, _, seq) in PRESETS.items():
            rows[(key, "mjx")] = score(np.load(MJX / f"{key}.npz")["qpos"], seq)
            for conv in ("ctrl", "mocap"):
                path = GAP / f"{key}_{conv}.npz"
                if path.exists():
                    rows[(key, f"C/{conv}")] = score(np.load(path)["qpos"], seq)
    else:
        meta = json.loads((ROOT / "outputs/webdemo/symbols.json").read_text())
        for key, p in meta["presets"].items():
            seq = [(s, int(h)) for s, h in p["blocks"]]
            rows[(key, "C/ctrl")] = score(np.load(GAP / f"demo_{key}_ctrl.npz")["qpos"], seq)
    print(f"{'preset':8s} {'engine':8s} {'n':>3s} {'agree med':>9s} {'mean':>6s} "
          f"{'fell>=1s':>8s} {'speed':>6s}")
    for (key, eng), r in rows.items():
        print(f"{key:8s} {eng:8s} {r['n']:3d} {r['agree_median']:9.3f} "
              f"{r['agree_mean']:6.3f} {r['fell_1s']:8.3f} {r['speed']:6.3f}")
    (GAP / f"summary_{args.set}.json").write_text(json.dumps(
        {f"{k}/{e}": r for (k, e), r in rows.items()}, indent=1))


if __name__ == "__main__":
    main()
