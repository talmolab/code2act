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

from c2a.codegen.shared.label_alignment import BEHAVIOR_TO_INT, DIRECTION_TO_INT
from c2a.codegen.shared.mlb_labels import MLB_LABELS, MLB_LABEL_TO_INT
from c2a.experiments.shared.code_labels import behavior_direction_to_nine
from c2a.experiments.shared.code_labels_from_rollouts import (
    augment_code_labels_with_nine_class,
    load_code_labels_json,
)
from c2a.experiments.shared.topomimic_labeler import label_rollout_body
from c2a.paths import get_repo_root

GROUP_ORDER = ["Immobile", "Rear", "Turn", "Walk"]
LABELS = GROUP_ORDER  # confusion is over the four behavior classes only

# 9 directional categories (mlb codebook order) for the directional round-trip
# confusion. Left := sign * net_yaw > 0; Straight inside the deadband (Walk/Rear
# only); Turn forced Left/Right; Immobile direction-free.
NINE_LABELS: list[str] = list(MLB_LABELS)
NINE_SIGN: float = 1.0
NINE_DEADBAND: float = 0.10


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


def _settled_net_yaw(qpos: np.ndarray, settle_frac: float,
                     sign: float = NINE_SIGN) -> np.ndarray:
    """Per-body net yaw over the converged window ``[settle_frac*T, T)`` -> ``(N,)``.

    Same net-heading-change primitive the mlb tokenizer / nine-class labeler use
    (root quaternion -> yaw -> unwrap -> ``sign * (yaw[-1] - yaw[0])``), restricted
    to the settled window so the initial drive-in transient does not contaminate
    the measured direction of *this* rollout body."""
    T = qpos.shape[1]
    lo = int(settle_frac * T)
    q = qpos[:, lo:, 3:7]  # root quaternion (w, x, y, z)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw = np.unwrap(yaw, axis=1)
    return sign * (yaw[:, -1] - yaw[:, 0])


def _instance_nine(beh: str, net_yaw: float,
                   deadband: float = NINE_DEADBAND) -> tuple[str, int] | None:
    """9-class category of a single re-tokenized rollout instance.

    Crosses the instance's re-inferred 4-class *behavior* (from the re-tokenizer)
    with the kinematic *direction* measured from THAT body's settled-window net
    yaw (same deadband / sign convention as the mlb labeler): Turn is forced
    Left/Right, Immobile is direction-free, Walk/Rear are Straight inside the
    deadband else Left/Right. Returns ``(label, code)`` in ``0..8`` or ``None``
    when the behavior is 'Other' (re-inferred outside the four named classes)."""
    if beh == "Immobile":
        c = MLB_LABEL_TO_INT["Immobile"]
        return MLB_LABELS[c], c
    if beh not in ("Walk", "Rear", "Turn"):
        return None
    if beh == "Turn":
        d = DIRECTION_TO_INT["Left"] if net_yaw > 0 else DIRECTION_TO_INT["Right"]
    elif abs(net_yaw) < deadband:
        d = DIRECTION_TO_INT["Straight"]
    else:
        d = DIRECTION_TO_INT["Left"] if net_yaw > 0 else DIRECTION_TO_INT["Right"]
    return behavior_direction_to_nine(BEHAVIOR_TO_INT[beh], d)


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
    # Re-tokenizer selection. "kpms" (default) re-segments rollouts with the
    # K=100 KPMS AR-HMM in an x64 subprocess. "vae" re-segments with a frozen
    # kinematics-VAE + K-Means tokenizer in-process (qpos -> VAE mean -> PCA ->
    # K-Means), the *own* tokenizer of the kvae decoder. "mlb" re-tokenizes with
    # the *own* tokenizer of the mlb decoder = the kinematic behavior+direction
    # auto-labeler (TopoMimic behavior over each body's settled window x measured
    # net-yaw direction), so each strategy is re-tokenized by its own tokenizer
    # (paper D5). For "mlb" the driving category is the FIXED definitional 9-class
    # category of the held mlb code (id 0..8 -> MLB_LABELS), NOT a rollout label.
    ap.add_argument("--segmenter", choices=["kpms", "vae", "mlb"], default="kpms")
    ap.add_argument("--vae-ckpt-dir",
                    default="outputs/tokenizer/trained_kinematics_vae_ckpts/"
                            "vae_sweep/kl_0p001/step_0030000")
    ap.add_argument("--vae-tokenizer-pkl",
                    default="outputs/tokenizer/trained_kinematics_vae_ckpts/"
                            "vae_sweep/kl_0p001/tokenize_k25/tokenizer.pkl")
    ap.add_argument("--vae-num-codes", type=int, default=25)
    ap.add_argument("--vae-n-joints", type=int, default=67)
    ap.add_argument("--vae-window-len", type=int, default=10)
    ap.add_argument("--vae-horizon", type=int, default=5)
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
    # "own-tokenizer" conditions: rollout codes share the re-tokenizer's
    # alphabet, so per-frame code-level reconstruction (drive code c, recover
    # code c) is meaningful and driving behavior comes from the rollout
    # auto-labels. For the vae segmenter every condition is its own tokenizer
    # (the kvae decoder re-tokenized by the kvae tokenizer); for kpms only the
    # kpms condition is.
    if args.segmenter == "vae":
        is_kpms_decoder = {c: True for c in args.conditions}
    elif args.segmenter == "mlb":
        # mlb is re-tokenized by the kinematic auto-labeler, not a KPMS/VAE code
        # alphabet, so there is no per-frame code to reconstruct: skip the
        # code-level reconstruction confusion entirely.
        is_kpms_decoder = {c: False for c in args.conditions}
    else:
        is_kpms_decoder = {c: (c == "kpms") for c in args.conditions}

    # Build the in-process VAE segmenter once (deterministic; no subprocess).
    vae_seg = None
    if args.segmenter == "vae":
        from c2a.experiments.shared.segmenters import AESegmenter

        vae_seg = AESegmenter(
            "vae", "vae", str(_abs(args.vae_ckpt_dir)),
            num_codes=int(args.vae_num_codes),
            n_joints=int(args.vae_n_joints),
            use_qvel=False,
            window_len=int(args.vae_window_len),
            horizon=int(args.vae_horizon),
            tokenizer_pkl=str(_abs(args.vae_tokenizer_pkl)),
        )
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

        # Driving 9-class category per held code: modal behavior x kinematic
        # net-yaw direction of the held-code rollout bodies (the mlb convention),
        # from the run's rollout code-labels. None when this condition has no
        # rollout code-labels (e.g. provenance-only designed codebooks).
        nine_labels_path = base / "labels" / "code_labels.json"
        nine_drive: dict[int, dict] = {}
        if nine_labels_path.exists():
            _cl = augment_code_labels_with_nine_class(
                load_code_labels_json(nine_labels_path), data_dir,
                sign=NINE_SIGN, deadband=NINE_DEADBAND)
            nine_drive = {int(c): info for c, info in _cl.items()}

        # confusion matrix: GT driving behavior (4) x predicted behavior (4+Other)
        conf = np.zeros((len(GROUP_ORDER), len(LABELS)), dtype=np.int64)
        # directional round trip: driving 9-class (rows) x re-inferred 9-class
        # (cols). Re-inferred category = re-tokenized behavior x measured net-yaw
        # direction of that rollout body.
        cat_conf = np.zeros((len(NINE_LABELS), len(NINE_LABELS)), dtype=np.int64)
        per_code = {}
        clip_rows = []  # (driving_code, gt_idx, pred_beh_idx, pred_code)
        for f in files:
            d = np.load(f)
            cid = int(d["code"])

            # --- mlb re-tokenization: own tokenizer = kinematic auto-labeler ---
            # Driving category = FIXED definitional 9-class label of the held mlb
            # code (id 0..8 -> MLB_LABELS); re-inferred category = TopoMimic
            # auto-labeler behavior over each body's settled window x that body's
            # measured net-yaw direction (via _instance_nine). No KPMS/VAE encode.
            if args.segmenter == "mlb":
                if 0 <= cid < len(NINE_LABELS):
                    drow = cid
                    nine_label = NINE_LABELS[cid]
                    gt = nine_label.split("_")[0]
                else:
                    # validation fallback (non-mlb codebook with >9 codes): take
                    # the driving category from the rollout labels so the
                    # re-inference path can be exercised on e.g. kvae rollouts.
                    drv9 = nine_drive.get(cid)
                    if not drv9 or int(drv9.get("nine_code", -1)) < 0:
                        continue
                    drow = int(drv9["nine_code"])
                    nine_label = drv9.get("nine_label")
                    gt = (nine_label or "").split("_")[0]
                if gt not in GROUP_ORDER:
                    continue
                gi = lab_idx[gt]
                q = np.asarray(d["qpos"])[: args.n_bodies]
                T = q.shape[1]
                lo = int(args.settle_frac * T)
                nyaw = _settled_net_yaw(q, args.settle_frac)
                pred_beh: list[str] = []
                reinf_codes: list[int] = []
                n_hit = 0
                for bi in range(q.shape[0]):
                    bl = label_rollout_body(
                        np.asarray(q[bi, lo:], dtype=np.float64),
                        src_fps=100.0, groom_to_immobile=True)
                    pb = bl.behavior
                    pred_beh.append(pb)
                    if pb in lab_idx:
                        conf[gi, lab_idx[pb]] += 1
                    r = _instance_nine(pb, float(nyaw[bi]))
                    if r is not None:
                        cat_conf[drow, r[1]] += 1
                        reinf_codes.append(int(r[1]))
                        if r[1] == drow:
                            n_hit += 1
                n = len(pred_beh)
                behavior_recall = (float(np.mean([b == gt for b in pred_beh]))
                                   if n else float("nan"))
                category_recall = float(n_hit / n) if n else None
                if reinf_codes:
                    vc, cnt = np.unique(reinf_codes, return_counts=True)
                    code_purity = float(cnt.max() / len(reinf_codes))
                    modal_pred_code = int(vc[np.argmax(cnt)])
                else:
                    code_purity = 0.0
                    modal_pred_code = -1
                per_code[str(cid)] = {
                    "driving_behavior": gt, "n_clips": int(n),
                    "code_purity": code_purity, "behavior_recall": behavior_recall,
                    "code_recon": None, "modal_pred_code": modal_pred_code,
                    "nine_label": nine_label, "category_recall": category_recall,
                }
                cr_str = f"{category_recall:.2f}" if category_recall is not None else "n/a"
                print(f"  [{cond}] mlb code {cid:3d} ({nine_label}): "
                      f"category-recall={cr_str} "
                      f"consistency={code_purity:.2f} (n={n})", flush=True)
                continue

            gt = cond_beh.get(cid, "Other")
            if gt not in GROUP_ORDER:
                continue  # only score codes with a defined target behavior
            cache = rec_dir / f"code{cid:03d}.npz"
            if args.reuse and cache.exists():
                rc = np.asarray(np.load(cache)["codes"], dtype=np.int32)
            else:
                q = np.asarray(d["qpos"])[: args.n_bodies]
                t0 = time.time()
                if vae_seg is not None:
                    rc = vae_seg.encode(q, seed=args.seed).astype(np.int32)
                else:
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

            # Directional (9-class) round trip for this held code. Driving
            # category = code's modal 9-class label; re-inferred category =
            # per-body re-tokenized behavior x measured net-yaw direction of
            # that body's settled window.
            drv9 = nine_drive.get(cid)
            nine_label = drv9.get("nine_label") if drv9 else None
            category_recall = None
            if drv9 and int(drv9.get("nine_code", -1)) >= 0:
                drow = int(drv9["nine_code"])
                q_dir = np.asarray(d["qpos"])[: len(pred_code)]
                nyaw = _settled_net_yaw(q_dir, args.settle_frac)
                n_hit = 0
                for bi, pb in enumerate(pred_beh):
                    r = _instance_nine(pb, float(nyaw[bi]))
                    if r is None:
                        continue
                    cat_conf[drow, r[1]] += 1
                    if r[1] == drow:
                        n_hit += 1
                category_recall = float(n_hit / len(pred_beh))

            per_code[str(cid)] = {
                "driving_behavior": gt, "n_clips": int(len(pred_code)),
                "code_purity": code_purity, "behavior_recall": behavior_recall,
                "code_recon": code_recon,
                "modal_pred_code": int(vc[np.argmax(cnt)]),
                "nine_label": nine_label, "category_recall": category_recall,
            }

        met = classification_metrics(conf.astype(float))
        cond_rec = {
            "n_codes": len(per_code), "n_clips": int(conf.sum()),
            "behavior_confusion": conf.tolist(), **met,
            "mean_code_purity": float(np.mean(
                [v["code_purity"] for v in per_code.values()])),
            "codes": per_code,
        }
        # directional 9-class round trip (rows = driving category, cols =
        # re-inferred category). Row-normalize at plot time to recall.
        if cat_conf.sum() > 0:
            cat_total = int(cat_conf.sum())
            cond_rec.update({
                "category_confusion": cat_conf.tolist(),
                "category_labels": list(NINE_LABELS),
                "category_accuracy": float(np.trace(cat_conf) / cat_total),
            })
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
                 if is_kpms_decoder[cond] else "")
              + (f" category-acc={cond_rec['category_accuracy']:.2f}"
                 if "category_accuracy" in cond_rec else ""), flush=True)

    (out_dir / "roundtrip_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/roundtrip_summary.json")


if __name__ == "__main__":
    main()
