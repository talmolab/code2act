"""Round-trip recovery for the perturbation experiment (Fig 5 b,c).

Replaces the manifold-distance recovery curves with a code-space round-trip:
re-tokenize the perturbation rollouts with the trained K=100 KPMS and ask
whether, after an external impulse, the body's re-inferred KPMS code (and its
behavior class) returns to the *held* code.

Per (code, amplitude) we re-tokenize the rollout qpos -> KPMS codes and measure,
over time:
  * exact-recovery(t) — fraction of bodies whose windowed modal re-inferred code
                        equals the held (driving) KPMS code id.
  * beh-recovery(t)   — same on the behavior class.
A wrong-code control scores recovery against a *different* code's behavior.

Panel (b): recovery(t) at a representative amplitude (drops at the impulse,
returns to its pre-impulse floor); the wrong-code control stays low.
Panel (c): post-recovery fraction vs impulse magnitude (the basin), in code space.

Run (after the KPMS perturbation rollouts exist)::

    python -m c2a.experiments.perturbation.run_perturbation_roundtrip \
        --codes 1 4 2 5 --n-bodies 40
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from c2a.experiments.ic_contraction.run_code_roundtrip import (
    GROUP_ORDER,
    behavior_map_from_labels,
    retokenize,
    _windowed_modal,
)
from c2a.paths import get_repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="kpms")
    ap.add_argument("--base",
                    default="outputs/perturbation/perturbation_recovery")
    ap.add_argument("--codes", nargs="+", type=int, default=[1, 4, 2, 5])
    ap.add_argument("--kpms-model-dir",
                    default="outputs/tokenizer/kpms_sweep/"
                            "s100_k1e+02_l10_arhmm/seed4")
    ap.add_argument("--kpms-model-name",
                    default="keypoint_arhmm_states100_seed4")
    ap.add_argument("--behavior-labels",
                    default="outputs/code_grid/code_grid_rollouts/kpms/"
                            "labels/code_labels.json")
    ap.add_argument("--fk-h5", default="data/rodent/rodent_reference_clips.h5")
    ap.add_argument("--stac-xml", default="c2a/assets/rodent.xml")
    ap.add_argument("--n-bodies", type=int, default=40)
    ap.add_argument("--num-iters", type=int, default=50)
    ap.add_argument("--n-grid", type=int, default=60)
    ap.add_argument("--out",
                    default="outputs/perturbation/perturbation_roundtrip/default")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    root = get_repo_root()

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    run_dir = _abs(args.base) / args.run
    summary = json.loads((run_dir / "perturbation_summary.json").read_text())
    t_perturb = int(summary["t_perturb"])
    T = int(summary["hold_frames"])
    beh_map = behavior_map_from_labels(_abs(args.behavior_labels))

    out_dir = _abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_dir = out_dir / "recoded"
    rec_dir.mkdir(exist_ok=True)
    tmp_dir = out_dir / "tmp"

    # discover available amplitudes
    files = sorted((run_dir / "data").glob("rollout_code*_amp*.npz"))
    amps = sorted({float(re.search(r"amp([\d.]+)\.npz", f.name).group(1)
                         .rstrip(".")) for f in files})

    grid = np.linspace(0, T - 1, args.n_grid).astype(int)
    half = max(5, int(0.02 * T))
    payload = {"grid": grid, "T": T, "t_perturb": t_perturb,
               "amps": np.array(amps)}
    out_summary = {"t_perturb": t_perturb, "T": T, "amps": amps,
                   "codes": {}, "behaviors": GROUP_ORDER}

    def _load(cid, amp):
        # filenames use code{cid:03d}_amp{amp:g} with a trailing dot variant
        for cand in [f"rollout_code{cid:03d}_amp{amp:g}.npz",
                     f"rollout_code{cid:03d}_amp{amp}.npz"]:
            p = run_dir / "data" / cand
            if p.exists():
                return np.asarray(np.load(p)["qpos"])[: args.n_bodies]
        raise FileNotFoundError(f"no rollout for code {cid} amp {amp}")

    for cid in args.codes:
        held_beh = beh_map.get(cid, "Other")
        per_amp = {}
        for amp in amps:
            cache = rec_dir / f"code{cid:03d}_amp{amp:g}.npz"
            if args.reuse and cache.exists():
                rc = np.asarray(np.load(cache)["codes"], dtype=np.int32)
            else:
                q = _load(cid, amp)
                rc = retokenize(q, model_dir=str(_abs(args.kpms_model_dir)),
                                model_name=args.kpms_model_name,
                                fk_h5=str(_abs(args.fk_h5)),
                                stac_xml=str(_abs(args.stac_xml)),
                                num_iters=args.num_iters, seed=0,
                                tmp_dir=tmp_dir, tag=f"pb_{cid}_{amp:g}")
                np.savez_compressed(cache, codes=rc, code=cid, amp=amp)
                print(f"  code {cid} amp {amp:g}: retokenized {rc.shape}",
                      flush=True)
            # exact / behavior recovery curves
            ex = np.array([float(np.mean(_windowed_modal(rc, int(t), half) == cid))
                           for t in grid])
            modal_beh = lambda t: np.array(
                [beh_map.get(c, "Other") for c in _windowed_modal(rc, int(t), half)])
            bh = np.array([float(np.mean(modal_beh(t) == held_beh)) for t in grid])
            payload[f"exact_{cid}_{amp:g}"] = ex
            payload[f"beh_{cid}_{amp:g}"] = bh
            per_amp[f"{amp:g}"] = {
                "floor_exact": float(np.mean(ex[grid < t_perturb])),
                "floor_beh": float(np.mean(bh[grid < t_perturb])),
                "final_exact": float(np.mean(ex[grid > t_perturb + 0.6 * (T - t_perturb)])),
                "final_beh": float(np.mean(bh[grid > t_perturb + 0.6 * (T - t_perturb)])),
                "trough_beh": float(np.min(bh[(grid >= t_perturb)
                                              & (grid < t_perturb + 150)])),
            }
        out_summary["codes"][str(cid)] = {"behavior": held_beh, "amps": per_amp}

    np.savez(out_dir / "curves.npz", **payload)
    (out_dir / "perturbation_roundtrip_summary.json").write_text(
        json.dumps(out_summary, indent=2))
    print(f"wrote {out_dir}/perturbation_roundtrip_summary.json")


if __name__ == "__main__":
    main()
