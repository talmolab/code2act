"""Round-trip classification: re-tokenize held-code rollouts with the trained KPMS.

Code-space counterpart of ``run_ic_contraction`` and replacement for the
hard-to-read manifold-distance metric. Same experiment (the held-code rollouts:
``K`` clips/bodies from diverse initial conditions, one code held constant,
decoder-only); only the *metric* changes.

We run the **same trained Keypoint-MoSeq tokenizer the KPMS decoder was trained
with** (``K=100`` syllables) forward over each clip's rolled-out motion (qpos ->
forward kinematics -> keypoints -> ``apply_model``, in an x64 subprocess), reduce
each clip to a **single re-inferred code** (its modal syllable over the converged
window), and score the round trip as a **classification** problem: the driving
code (its behavior class) is the ground-truth label and the re-inferred code (its
behavior class) is the prediction. From the (driving, re-inferred) pairs we build
a **confusion matrix** and report **accuracy / recall / precision / F1**.

The same ``K=100`` KPMS tokenizer re-tokenizes both the KPMS-decoder and the
designed-codebook-decoder rollouts, so the round-trip quality is a property of the
controller, not of a per-decoder tokenizer.

Run (after the held-code rollouts exist for each condition)::

    python -m c2a.experiments.ic_contraction.run_code_roundtrip \
        --conditions kpms multiscale --n-bodies 60

Outputs to ``outputs/ic_contraction/code_roundtrip/<run>/``:
    recoded_<cond>/code<NNN>.npz   — cached (N, T) re-inferred KPMS codes
    clip_preds_<cond>.npz          — per-clip (driving, predicted) labels + matrices
    roundtrip_summary.json         — confusion matrices + accuracy/recall/prec/F1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from c2a.paths import get_repo_root

GROUP_ORDER = ["Immobile", "Rear", "Turn", "Walk"]
LABELS = GROUP_ORDER  # confusion is over the four behavior classes only


# --------------------------------------------------------------------------- #
# KPMS re-tokenization (x64 subprocess worker)
# --------------------------------------------------------------------------- #
def retokenize(qpos: np.ndarray, *, model_dir: str, model_name: str,
               fk_h5: str, stac_xml: str, num_iters: int, seed: int,
               tmp_dir: Path, tag: str) -> np.ndarray:
    """qpos ``(N, T, nq)`` -> re-inferred KPMS codes ``(N, T)`` int32.

    Runs ``c2a.experiments.tokenizer._roundtrip_kpms_worker`` in a separate
    process because ``keypoint_moseq`` forces (irreversible) x64."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_npz = tmp_dir / f"rt_in_{tag}.npz"
    out_npz = tmp_dir / f"rt_out_{tag}.npz"
    np.savez(in_npz, qpos=np.asarray(qpos, dtype=np.float32))
    pyrun = Path(get_repo_root()) / "bash_scripts" / "_pyrun.py"
    cmd = [
        sys.executable, str(pyrun),
        "c2a.experiments.tokenizer._roundtrip_kpms_worker",
        "--qpos-npz", str(in_npz), "--out-npz", str(out_npz),
        "--kpms-model-dir", model_dir, "--kpms-model-name", model_name,
        "--num-iters", str(int(num_iters)),
        "--fk-h5", fk_h5, "--stac-xml", stac_xml,
        "--seeds", str(int(seed)),
    ]
    subprocess.run(cmd, check=True)
    out = np.load(out_npz)
    codes = np.asarray(out["codes"], dtype=np.int32)[0]  # (N, T), seed 0
    in_npz.unlink(missing_ok=True)
    out_npz.unlink(missing_ok=True)
    return codes


# --------------------------------------------------------------------------- #
# Clip-level reduction + behavior maps
# --------------------------------------------------------------------------- #
def _windowed_modal(codes: np.ndarray, t: int, half: int) -> np.ndarray:
    """Per-clip modal code in a window ``[t-half, t+half]`` -> ``(N,)``.

    Retained for the perturbation round-trip, which scores recovery in a window
    around the impulse."""
    T = codes.shape[1]
    lo, hi = max(0, t - half), min(T, t + half + 1)
    win = codes[:, lo:hi]
    out = np.empty(win.shape[0], dtype=np.int32)
    for i, row in enumerate(win):
        vals, cnt = np.unique(row, return_counts=True)
        out[i] = int(vals[np.argmax(cnt)])
    return out


def clip_modal_code(codes: np.ndarray, settle_frac: float) -> np.ndarray:
    """One re-inferred code per clip: the modal KPMS syllable over the converged
    window ``[settle_frac*T, T)`` (the initial transient, while the controller
    drives the body out of its random initial pose, is dropped). ``(N,)``."""
    N, T = codes.shape
    lo = int(settle_frac * T)
    out = np.empty(N, dtype=np.int32)
    for i in range(N):
        vals, cnt = np.unique(codes[i, lo:], return_counts=True)
        out[i] = int(vals[np.argmax(cnt)])
    return out


def behavior_map_from_labels(labels_json: Path) -> dict[int, str]:
    """KPMS syllable -> behavior class, from the kpms rollout auto-labels
    (top-usage syllables are labeled; the rest map to 'Other')."""
    d = json.loads(labels_json.read_text())
    out: dict[int, str] = {}
    for cid, info in d["codes"].items():
        lab = info.get("label", "unlabeled").split("_")[0]
        out[int(cid)] = lab if lab in GROUP_ORDER else "Other"
    return out


_DESIGNED_TOKEN = {"Walk": "Walk", "Rear": "Rear", "Turn": "Turn",
                   "Imm": "Immobile", "Immobile": "Immobile"}


def designed_behavior_map(codes_path: Path) -> dict[int, str]:
    """Designed code id -> defined (dominant/first) behavior, from the codebook
    provenance (so labels match the other designed-codebook figures)."""
    d = np.load(codes_path, allow_pickle=True)
    prov = json.loads(str(d["provenance_json"]))
    out: dict[int, str] = {}
    for pc in prov["per_code"]:
        if pc.get("type") == "sentinel":
            continue
        first = pc["name"].split(".", 1)[1].split("_")[0]
        out[int(pc["id"])] = _DESIGNED_TOKEN.get(first, "Other")
    return out


# --------------------------------------------------------------------------- #
# Classification metrics from a confusion matrix
# --------------------------------------------------------------------------- #
def classification_metrics(conf: np.ndarray) -> dict:
    """``conf`` is ``(4, 5)``: rows = ground-truth driving behavior (the four
    classes), cols = predicted behavior (four classes + Other). Returns
    accuracy, macro-F1, and per-class recall / precision / F1."""
    nb = len(GROUP_ORDER)
    total = conf.sum()
    diag = float(sum(conf[i, i] for i in range(nb)))
    acc = diag / total if total else float("nan")
    recall, precision, f1 = {}, {}, {}
    for i, g in enumerate(GROUP_ORDER):
        rsum = conf[i, :].sum()
        csum = conf[:, i].sum()
        r = conf[i, i] / rsum if rsum else float("nan")
        p = conf[i, i] / csum if csum else float("nan")
        recall[g] = float(r)
        precision[g] = float(p)
        f1[g] = float(2 * p * r / (p + r)) if (p + r) else 0.0
    macro_f1 = float(np.nanmean([f1[g] for g in GROUP_ORDER]))
    return {"accuracy": acc, "macro_f1": macro_f1,
            "recall": recall, "precision": precision, "f1": f1}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["kpms", "multiscale"])
    ap.add_argument("--rollouts-base",
                    default="outputs/code_grid/code_grid_rollouts")
    ap.add_argument("--pose", default="random")
    ap.add_argument("--kpms-model-dir",
                    default="outputs/tokenizer/kpms_sweep/"
                            "s100_k1e+02_l10_arhmm/seed4")
    ap.add_argument("--kpms-model-name",
                    default="keypoint_arhmm_states100_seed4")
    ap.add_argument("--behavior-labels",
                    default="outputs/code_grid/code_grid_rollouts/kpms/"
                            "labels/code_labels.json",
                    help="kpms rollout labels -> KPMS-syllable behavior map "
                         "(shared across conditions; same K=100 tokenizer)")
    ap.add_argument("--designed-codes",
                    default="outputs/tokenizer/designed_v3/best_codes.npz",
                    help="designed codebook npz; non-kpms conditions take their "
                         "driving behavior from its provenance, not auto-labels")
    ap.add_argument("--fk-h5", default="data/rodent/rodent_reference_clips.h5")
    ap.add_argument("--stac-xml", default="c2a/assets/rodent.xml")
    ap.add_argument("--n-bodies", type=int, default=60)
    ap.add_argument("--num-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle-frac", type=float, default=0.2,
                    help="drop this leading fraction of each clip (the initial "
                         "drive-in transient) before taking the modal code")
    ap.add_argument("--out", default="outputs/ic_contraction/code_roundtrip/default")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached recoded npz if present")
    ap.add_argument("--max-codes", type=int, default=0,
                    help="cap codes per condition (0 = all); for quick tests")
    args = ap.parse_args()

    root = get_repo_root()

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    out_dir = _abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    beh_map = behavior_map_from_labels(_abs(args.behavior_labels))
    is_kpms_decoder = {c: (c == "kpms") for c in args.conditions}
    lab_idx = {b: i for i, b in enumerate(LABELS)}

    summary = {"conditions": {}, "n_bodies": args.n_bodies,
               "num_iters": args.num_iters, "settle_frac": args.settle_frac,
               "kpms_model": args.kpms_model_name, "behaviors": GROUP_ORDER,
               "labels": LABELS}

    for cond in args.conditions:
        base = _abs(args.rollouts_base) / cond
        data_dir = base / "data"
        files = sorted(data_dir.glob(f"rollout_{args.pose}_code*.npz"))
        if args.max_codes:
            files = files[: args.max_codes]
        if not files:
            print(f"[skip] no rollouts for condition {cond} at {data_dir}")
            continue
        # driving-code behavior: designed codebook uses its provenance, learned
        # tokenizers (kpms) use the rollout auto-label.
        if is_kpms_decoder[cond]:
            clp = base / "labels" / "code_labels.json"
            cond_beh = behavior_map_from_labels(clp) if clp.exists() else {}
        else:
            dc = _abs(args.designed_codes)
            cond_beh = designed_behavior_map(dc) if dc.exists() else {}

        rec_dir = out_dir / f"recoded_{cond}"
        rec_dir.mkdir(parents=True, exist_ok=True)

        # confusion matrix: GT driving behavior (4) x predicted behavior (4+Other)
        conf = np.zeros((len(GROUP_ORDER), len(LABELS)), dtype=np.int64)
        per_code = {}
        clip_rows = []  # (driving_code, gt_idx, pred_beh_idx, pred_code)
        for f in files:
            d = np.load(f)
            cid = int(d["code"])
            gt = cond_beh.get(cid, "Other")
            if gt not in GROUP_ORDER:
                continue  # only score codes with a defined target behavior
            cache = rec_dir / f"code{cid:03d}.npz"
            if args.reuse and cache.exists():
                rc = np.asarray(np.load(cache)["codes"], dtype=np.int32)
            else:
                q = np.asarray(d["qpos"])[: args.n_bodies]
                t0 = time.time()
                rc = retokenize(
                    q, model_dir=str(_abs(args.kpms_model_dir)),
                    model_name=args.kpms_model_name,
                    fk_h5=str(_abs(args.fk_h5)),
                    stac_xml=str(_abs(args.stac_xml)),
                    num_iters=args.num_iters, seed=args.seed,
                    tmp_dir=tmp_dir, tag=f"{cond}_{cid}")
                np.savez_compressed(cache, codes=rc, code=cid)
                print(f"  [{cond}] code {cid:3d}: retokenized {rc.shape} "
                      f"in {time.time() - t0:.0f}s", flush=True)

            pred_code = clip_modal_code(rc, args.settle_frac)   # (N,)
            pred_beh = [beh_map.get(c, "Other") for c in pred_code]
            gi = lab_idx[gt]
            for pb in pred_beh:                       # behavior confusion (aux)
                if pb in lab_idx:
                    conf[gi, lab_idx[pb]] += 1
            for pc in pred_code:                      # code-level clip records
                clip_rows.append((cid, int(pc)))
            vc, cnt = np.unique(pred_code, return_counts=True)
            code_purity = float(cnt.max() / len(pred_code))      # clips agree?
            behavior_recall = float(np.mean([b == gt for b in pred_beh]))
            code_recon = (float(np.mean(pred_code == cid))       # recover code c?
                          if is_kpms_decoder[cond] else None)
            per_code[str(cid)] = {
                "driving_behavior": gt, "n_clips": int(len(pred_code)),
                "code_purity": code_purity, "behavior_recall": behavior_recall,
                "code_recon": code_recon,
                "modal_pred_code": int(vc[np.argmax(cnt)]),
            }

        met = classification_metrics(conf.astype(float))
        cond_rec = {
            "n_codes": len(per_code), "n_clips": int(conf.sum()),
            "behavior_confusion": conf.tolist(), **met,
            "mean_code_purity": float(np.mean(
                [v["code_purity"] for v in per_code.values()])),
            "codes": per_code,
        }
        # code-level reconstruction confusion: the round trip proper — drive a
        # syllable, re-segment, do we get THAT syllable back? Rows/cols are the
        # analyzed driving codes, ordered by behavior so within-behavior blocks
        # are visible (KPMS decoder only; designed codes are not KPMS syllables).
        if is_kpms_decoder[cond]:
            order_i = sorted((int(c) for c in per_code), key=lambda c: (
                GROUP_ORDER.index(per_code[str(c)]["driving_behavior"]), c))
            idx_of = {c: i for i, c in enumerate(order_i)}
            K = len(order_i)
            ccode = np.zeros((K, K), dtype=np.int64)
            off = 0
            for dc, pc in clip_rows:
                if pc in idx_of:
                    ccode[idx_of[dc], idx_of[pc]] += 1
                else:
                    off += 1                            # re-inferred outside set
            tot = int(ccode.sum())
            cond_rec.update({
                "code_confusion": ccode.tolist(),
                "code_order": order_i,
                "code_behaviors": [per_code[str(c)]["driving_behavior"]
                                   for c in order_i],
                "code_accuracy": float(np.trace(ccode) / tot) if tot else float("nan"),
                "code_offset_frac": float(off / (tot + off)) if (tot + off) else 0.0,
            })
        np.savez(out_dir / f"clip_preds_{cond}.npz",
                 behavior_confusion=conf, labels=np.array(LABELS),
                 clip_rows=np.array(clip_rows, dtype=np.int64))
        summary["conditions"][cond] = cond_rec
        print(f"[{cond}] behavior acc={met['accuracy']:.2f} F1={met['macro_f1']:.2f} "
              f"code-purity={cond_rec['mean_code_purity']:.2f}"
              + (f" code-acc={cond_rec['code_accuracy']:.2f}"
                 if is_kpms_decoder[cond] else ""), flush=True)

    (out_dir / "roundtrip_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/roundtrip_summary.json")


if __name__ == "__main__":
    main()
