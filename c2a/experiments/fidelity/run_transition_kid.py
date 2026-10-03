"""KID for commanded transition arms, in Table 1's feature space.

Answers: does commanding a symbol transition that never occurs in the corpus
degrade the realism of the generated motion?

Everything is held identical to the Table 1 run (``run_inception_distance`` with
``moseq_inception_distance_mlb``) so the numbers are directly comparable:

* same controller (step 208), same 450-control-step survival threshold and
  225-mocap-frame truncation, same ``exclude_xy`` / ``handle_rotation`` /
  joint-normalisation preprocessing;
* same "real" distribution -- the 100 saved mimic-mjx rollouts;
* the **same cached VAE feature extractor** (3 seeds), reused rather than
  retrained, so the arms are embedded in an identical feature space.

Arms are subsampled to ``--n-clips`` (default 100) so sample count matches the
Table 1 methods too; otherwise an arm with 1800 trajectories would enjoy a
variance advantage the published rows do not have.

Usage::

    python3.12 bash_scripts/_pyrun.py c2a.experiments.fidelity.run_transition_kid \\
        --impossible outputs/behavior/impossible_transitions/mlb_s208_len500 \\
        --observed  outputs/behavior/impossible_transitions/mlb_s208_len500_observed
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np

from c2a.paths import get_repo_root
from c2a.experiments.fidelity.run_inception_distance import (
    _compute_vae_cache_key, _load_cached_vae, compute_joint_normalization,
    compute_kid, extract_features, filter_and_truncate, get_joint_start_index,
    normalize_joints, preprocess_data,
)

log = logging.getLogger(__name__)

REF_RUN = "outputs/fidelity/moseq_inception_distance_mlb"


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else get_repo_root() / p)


def load_saved_method(run_dir: Path, name: str) -> list[np.ndarray]:
    """Raw control-rate rollouts for one saved Table 1 method."""
    d = np.load(run_dir / f"rollouts_{name}.npz")
    return [d[k] for k in d.files if k.startswith("raw_")]


def load_arm_by_pair(arm_dir: Path) -> dict[str, list[np.ndarray]]:
    """Raw rollouts grouped by ``A__B`` pair tag, for the breakdown tables."""
    out: dict[str, list[np.ndarray]] = {}
    for p in sorted(glob.glob(str(arm_dir / "rollout_*.npz"))):
        tag = Path(p).stem.replace("rollout_", "")
        q = np.load(p, allow_pickle=True)["qpos"]
        out[tag] = [q[i] for i in range(q.shape[0])]
    return out


def load_arm(arm_dir: Path) -> list[np.ndarray]:
    """Flatten every (pair, body) rollout of a transition arm to a raw list."""
    out = []
    for p in sorted(glob.glob(str(arm_dir / "rollout_*.npz"))):
        q = np.load(p, allow_pickle=True)["qpos"]     # (bodies, T, nq)
        out.extend(q[i] for i in range(q.shape[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--impossible", required=True)
    ap.add_argument("--observed", default=None,
                    help="optional matched-structure control arm")
    ap.add_argument("--ref-run", default=REF_RUN)
    ap.add_argument("--survival-threshold", type=int, default=450)
    ap.add_argument("--steps-per-frame", type=int, default=2)
    ap.add_argument("--n-clips", type=int, default=100,
                    help="subsample each arm to match Table 1's sample count")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--breakdown", action="store_true",
                    help="also emit per-pair and per-source-family KID")
    ap.add_argument("--family-n", type=int, default=120,
                    help="trajectories per family, matched across families")
    ap.add_argument("--out", default="outputs/fidelity/transition_kid_mlb")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    ref = Path(_abs(args.ref_run))
    out_dir = Path(_abs(args.out)); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    def prep(raw: list[np.ndarray], tag: str) -> np.ndarray:
        arr, kept, total = filter_and_truncate(
            raw, args.survival_threshold, args.steps_per_frame)
        log.info(f"  {tag:<26} {kept}/{total} survived -> {arr.shape}")
        return arr

    # real = the same mimic-mjx rollouts the Table 1 VAE was trained on
    real_qpos = prep(load_saved_method(ref, "mimic_mjx"), "mimic_mjx (real)")

    arms: dict[str, np.ndarray] = {}
    for name in ("decoder_original_codes", "uniform_random"):
        arms[name] = prep(load_saved_method(ref, name), name)
    arms["impossible_transitions"] = prep(
        load_arm(Path(_abs(args.impossible))), "impossible_transitions")
    if args.observed:
        arms["observed_transitions"] = prep(
            load_arm(Path(_abs(args.observed))), "observed_transitions")

    # equalise sample counts so no arm gets a variance advantage
    for k, v in arms.items():
        if len(v) > args.n_clips:
            arms[k] = v[rng.choice(len(v), args.n_clips, replace=False)]
            log.info(f"  subsampled {k} -> {args.n_clips}")

    real = preprocess_data(real_qpos, True, True)
    js = get_joint_start_index(True, True)
    jm, jstd = compute_joint_normalization(real, js)
    real = normalize_joints(real, js, jm, jstd)
    fake = {k: normalize_joints(preprocess_data(v, True, True), js, jm, jstd)
            for k, v in arms.items()}
    log.info(f"\nreal {real.shape}; feature space = Table 1 cached VAE")

    input_size = int(np.prod(real.shape[1:]))
    key = _compute_vae_cache_key(
        real.shape, 64, [512, 256], 5000, 1.0, True, True, True,
        0.001, 0.00001, 0.1, True)
    cache_dir = ref / "vae_cache"

    per_seed: dict[str, list[float]] = {k: [] for k in arms}
    for sd in args.seeds:
        got = _load_cached_vae(cache_dir, key, sd, input_size, 64,
                               (512, 256), 0.1, True)
        if got is None:
            raise RuntimeError(
                f"no cached VAE for key={key} seed={sd} in {cache_dir}. The "
                f"cache key encodes real_data_shape, so this means the arm "
                f"truncation does not match the Table 1 run.")
        params, (_vae, _init, _apply, encode_fn) = got
        mu_real = extract_features(params, encode_fn, real)
        for k, v in fake.items():
            kid, _ = compute_kid(mu_real, extract_features(params, encode_fn, v))
            per_seed[k].append(float(kid))
        log.info(f"  seed {sd} done")

    res = {k: {"kid_mean": float(np.mean(v)), "kid_std": float(np.std(v)),
               "per_seed": v, "n_clips": int(len(arms[k]))}
           for k, v in per_seed.items()}
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))

    log.info(f"\n{'arm':<28}{'KID':>9}{'std':>8}{'n':>6}")
    log.info("-" * 51)
    for k in sorted(res, key=lambda x: res[x]["kid_mean"]):
        r = res[k]
        log.info(f"{k:<28}{r['kid_mean']:>9.3f}{r['kid_std']:>8.3f}{r['n_clips']:>6}")
    log.info(f"\nwrote {out_dir/'results.json'}")

    if not args.breakdown:
        return

    # ---- per-pair and per-source-family KID -------------------------------
    by_pair = load_arm_by_pair(Path(_abs(args.impossible)))
    fam: dict[str, list[np.ndarray]] = {}
    pair_raw: dict[str, np.ndarray] = {}
    for tag, raw in by_pair.items():
        arr, _, _ = filter_and_truncate(
            raw, args.survival_threshold, args.steps_per_frame)
        pair_raw[tag] = arr
        fam.setdefault(tag.split("__")[0].split("_")[0], []).append(arr)
    fam_arr = {k: np.concatenate(v, axis=0) for k, v in fam.items()}
    for k, v in fam_arr.items():
        if len(v) > args.family_n:
            fam_arr[k] = v[rng.choice(len(v), args.family_n, replace=False)]

    norm = lambda a: normalize_joints(preprocess_data(a, True, True), js, jm, jstd)
    pair_d = {k: norm(v) for k, v in pair_raw.items()}
    fam_d = {k: norm(v) for k, v in fam_arr.items()}

    pair_s: dict[str, list[float]] = {k: [] for k in pair_d}
    fam_s: dict[str, list[float]] = {k: [] for k in fam_d}
    for sd in args.seeds:
        params, (_v, _i, _a, enc) = _load_cached_vae(
            cache_dir, key, sd, input_size, 64, (512, 256), 0.1, True)
        mu_r = extract_features(params, enc, real)
        for k, v in pair_d.items():
            pair_s[k].append(float(compute_kid(mu_r, extract_features(params, enc, v))[0]))
        for k, v in fam_d.items():
            fam_s[k].append(float(compute_kid(mu_r, extract_features(params, enc, v))[0]))

    bd = {"per_pair": {k: {"kid_mean": float(np.mean(v)), "kid_std": float(np.std(v)),
                           "n": int(len(pair_d[k]))} for k, v in pair_s.items()},
          "per_source_family": {k: {"kid_mean": float(np.mean(v)),
                                    "kid_std": float(np.std(v)),
                                    "n": int(len(fam_d[k]))} for k, v in fam_s.items()}}
    (out_dir / "breakdown.json").write_text(json.dumps(bd, indent=2))

    log.info(f"\n{'source family':<18}{'KID':>9}{'std':>8}{'n':>6}")
    log.info("-" * 41)
    for k in sorted(bd["per_source_family"], key=lambda x: bd["per_source_family"][x]["kid_mean"]):
        r = bd["per_source_family"][k]
        log.info(f"{k:<18}{r['kid_mean']:>9.3f}{r['kid_std']:>8.3f}{r['n']:>6}")

    log.info(f"\n{'transition':<26}{'KID':>9}{'std':>8}{'n':>6}")
    log.info("-" * 49)
    for k in sorted(bd["per_pair"], key=lambda x: bd["per_pair"][x]["kid_mean"]):
        r = bd["per_pair"][k]
        log.info(f"{k.replace('__',' -> '):<26}{r['kid_mean']:>9.3f}{r['kid_std']:>8.3f}{r['n']:>6}")
    log.info(f"\nwrote {out_dir/'breakdown.json'}")


if __name__ == "__main__":
    main()
