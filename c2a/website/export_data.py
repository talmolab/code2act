"""Export the numbers behind the project page's interactive charts.

Every chart on the page is drawn from the same quantities as the manuscript's
figures, computed with the figure scripts' own helpers:

* ``roundtrip_bars.json``  per-symbol recall decomposition (Fig 3a, 6), from
  ``render_roundtrip_figure._decompose``.
* ``perturbation.json``    displacement recovered and pose deviation per
  behavior vs impulse, with the unperturbed floor and foreign-code control
  (Fig 3b,c, 7), from ``render_perturbation_recovery``.
* ``hidden3d.json``        GRU hidden-state trajectories of held MLB symbols on
  the first three principal components (Fig 2b's representation, drawn in 3D).

Usage::

    C2A_ROOT=$PWD python -m c2a.website.export_data
"""

from __future__ import annotations

import json
import logging

import numpy as np
from sklearn.decomposition import PCA

from c2a.figures.paper.render_roundtrip_figure import _decompose
from c2a.figures.render_perturbation_recovery import (
    _beh_mean_recovery, _recovery_deviation,
)
from c2a.paths import get_repo_root
from c2a.website.render_lib import mlb_rgb
from c2a.website.render_videos import DATA, MLB_SHORT, SITE

#: Large payloads are fetched at runtime rather than bundled into the page.
PUBLIC_DATA = SITE / "public" / "data"

log = logging.getLogger(__name__)
ROOT = get_repo_root()

#: (strategy, run dir under code_roundtrip, condition key in its summary)
ROUNDTRIP = [("mlb", "mlb", "mlb"), ("kpms", "default", "kpms"),
             ("kvae", "best_vae", "best_vae")]
PERTURB = [("mlb", "mlb"), ("kpms", "kpms"), ("kvae", "best_vae")]
BEHAVIORS = ["Immobile", "Rear", "Walk", "Turn"]


def export_roundtrip() -> None:
    out = {}
    for name, run, cond in ROUNDTRIP:
        summ = json.load(open(ROOT / "outputs/ic_contraction/code_roundtrip" / run
                              / "roundtrip_summary.json"))
        cc = summ["conditions"][cond]
        rows = _decompose(cc)
        n = sum(r["n"] for r in rows)
        out[name] = {
            "rows": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows],
            "symbol_acc": round(sum(r["exact"] * r["n"] for r in rows) / n, 4),
            "behavior_acc": round(sum((r["exact"] + r["same"]) * r["n"] for r in rows) / n, 4),
        }
        log.info(f"roundtrip {name}: {out[name]['symbol_acc']:.3f} / "
                 f"{out[name]['behavior_acc']:.3f}")
    (DATA / "roundtrip_bars.json").write_text(json.dumps(out, indent=1))


def export_perturbation() -> None:
    out = {}
    for name, run in PERTURB:
        rdir = ROOT / "outputs/perturbation/perturbation_recovery" / run
        summ = json.load(open(rdir / "perturbation_summary.json"))
        amps = [float(a) for a in summ["amplitudes"]]
        codes = [int(c) for c in summ["codes"]]
        beh_of = {int(c): summ["codes"][c].get("behavior") for c in summ["codes"]}
        wrong = {c: codes[(i + 1) % len(codes)] for i, c in enumerate(codes)}
        own, ctrl, _ = _recovery_deviation(rdir, codes, amps, wrong)
        curves = []
        for b in BEHAVIORS:
            bc = [c for c in codes if beh_of[c] == b]
            if not bc:
                continue
            rec = _beh_mean_recovery(summ, bc, len(amps))
            dv = np.nanmean(np.array([own[c] for c in bc if c in own]), axis=0)
            curves.append({
                "behavior": b,
                "recovery": [None if not np.isfinite(x) else round(float(x), 4) for x in rec],
                "deviation": [None if not np.isfinite(x) else round(float(x), 4) for x in dv],
            })
        out[name] = {
            "amps": amps, "curves": curves,
            "floor": round(float(np.nanmean([own[c][0] for c in codes if c in own])), 4),
            "control": round(float(np.nanmean([ctrl[c] for c in codes if c in ctrl])), 4),
        }
        log.info(f"perturbation {name}: {len(curves)} behaviors")
    (DATA / "perturbation.json").write_text(json.dumps(out, indent=1))


def export_hidden3d(n_bodies: int = 6, stride: int = 6, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    gdir = ROOT / "outputs/code_grid/code_grid_rollouts/mlb/data"
    per_code = []
    for c in range(9):
        d = np.load(gdir / f"rollout_random_code{c:03d}.npz")
        ok = np.flatnonzero(np.isfinite(d["qpos"]).all(axis=(1, 2)))
        bodies = rng.choice(ok, n_bodies, replace=False)
        per_code.append(d["hidden"][bodies, ::stride])          # (B, T', H)
    pca = PCA(n_components=3, random_state=seed).fit(
        np.concatenate([h[:, 30:].reshape(-1, h.shape[-1]) for h in per_code]))
    z_all = [pca.transform(h.reshape(-1, h.shape[-1])).reshape(*h.shape[:2], 3)
             for h in per_code]
    scale = float(np.percentile(np.abs(np.concatenate([z.reshape(-1, 3) for z in z_all])), 99))
    out = {"explained": np.round(pca.explained_variance_ratio_, 3).tolist(),
           "dt": stride / 100.0, "codes": []}
    for c, z in enumerate(z_all):
        out["codes"].append({"code": c, "name": MLB_SHORT[c], "color": list(mlb_rgb(c)),
                             "lines": np.round(z / scale, 3).tolist()})
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DATA / "hidden3d.json").write_text(json.dumps(out))
    log.info(f"hidden3d: explained {out['explained']}")


def export_transitions(run: str = "mlb_s208_len500") -> None:
    """Bout-level MLB transition counts of the corpus (Fig 13's inventory)."""
    meta = json.load(open(ROOT / "outputs/behavior/impossible_transitions" / run
                          / "transitions.json"))
    (DATA / "transitions.json").write_text(json.dumps(
        {"counts": meta["transition_counts"], "pairs": meta["pairs"]}))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    DATA.mkdir(parents=True, exist_ok=True)
    export_roundtrip()
    export_transitions()
    export_hidden3d()
    export_perturbation()


if __name__ == "__main__":
    main()
