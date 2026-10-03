"""Command transitions the reference corpus never contains.

Motivation
----------
Existing generation controls compare a structured symbol stream against a
*scrambled* one, which only shows that behavior is not driven by arbitrary
codebook activation. It does not separate a controller that generalizes from
one that replayed the transition grammar it was trained on.

This experiment closes that gap by commanding, for every ordered symbol pair
``(A, B)`` whose **bout-level** transition count in the reference corpus is
exactly zero, a two-phase stream: hold ``A`` for ``--hold-frames``, then hold
``B`` for ``--hold-frames``. The controller is asked for something strictly
outside its training distribution but composed of symbols it knows.

Three outcomes are distinguished per pair:

* the command lands (the labeler reads ``B`` in the settled window of phase 2),
* the command lands only after an **unrequested bridge** (a behavior appears
  between the two phases that was never commanded, e.g. a descent inserted
  between Rear and Walk), or
* it fails (body falls, terminates, or never reaches ``B``).

Bodies
------
``--n-diverse`` bodies start from unconstrained test clips (matching the
"diverse-start" convention used by the contraction and distinctness
experiments), plus one extra body started from a **walking** pose, reserved for
rendering so the figure grid matches the static-render style used elsewhere in
the manuscript.

Notes
-----
``kpms_code`` is a *forward window* of ``code_stack_size`` symbols, so at the
switch the controller legitimately sees the upcoming symbol a few frames early,
exactly as during training. The window is therefore built by slicing the
two-phase sequence, not by broadcasting a single symbol.

Outputs (``--out``):
    ``transitions.json``       impossible-pair inventory + endpoint support
    ``rollout_<A>_<B>.npz``    qpos / proprio / hidden / rewards per pair
    ``qualitative.csv``        deliverable (1): per-pair labeler outcome
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

from c2a.paths import get_repo_root

from c2a.codegen.code_sources import load_codes
from c2a.codegen.shared.label_alignment import load_label_frames
from c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled import (
    _build_pose_env, _pick_start_clips,
)
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint, make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.decoder_only import assert_decoder_only_settings
from c2a.experiments.shared.topomimic_labeler import label_rollout_body

log = logging.getLogger(__name__)

#: Canonical mlb symbol order (c2a.codegen.shared.mlb_labels).
MLB_NAMES = [
    "Walk_L", "Walk_R", "Walk_S", "Rear_L", "Rear_R", "Rear_S",
    "Turn_L", "Turn_R", "Immobile",
]


def _abs(p) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else get_repo_root() / p)


# ---------------------------------------------------------------------------
# Impossible-transition inventory
# ---------------------------------------------------------------------------

def bout_transition_counts(codes: np.ndarray, num_codes: int) -> np.ndarray:
    """Bout-level (run-length collapsed) symbol transition counts.

    Frame-level counts are dominated by self-transitions because syllables are
    sticky, so runs of one symbol are collapsed before counting.

    Args:
        codes: ``[n_clips, n_frames]`` integer symbol stream.
        num_codes: Vocabulary size ``K``.

    Returns:
        ``[K, K]`` counts, ``T[a, b]`` = number of bouts of ``a`` followed
        immediately by a bout of ``b``.
    """
    T = np.zeros((num_codes, num_codes), dtype=np.int64)
    for row in codes:
        bouts = [int(row[0])]
        for x in row[1:]:
            if int(x) != bouts[-1]:
                bouts.append(int(x))
        for a, b in zip(bouts[:-1], bouts[1:]):
            T[a, b] += 1
    return T


def impossible_pairs(
    T: np.ndarray, min_support: int,
) -> tuple[list[tuple[int, int]], dict]:
    """Ordered pairs with zero bout transitions, filtered by endpoint support.

    A zero count is only interpretable when both endpoints are themselves well
    attested; a symbol that barely occurs produces zeros everywhere purely by
    undersampling. Pairs whose source or target has fewer than ``min_support``
    bouts are reported separately rather than silently included.

    Args:
        T: ``[K, K]`` bout-level transition counts.
        min_support: Minimum outgoing/incoming bouts for an endpoint to count.

    Returns:
        ``(pairs, meta)`` where ``pairs`` are the supported zero-count pairs.
    """
    K = T.shape[0]
    out_support = T.sum(axis=1)
    in_support = T.sum(axis=0)
    supported, undersampled = [], []
    for a in range(K):
        for b in range(K):
            if a == b or T[a, b] != 0:
                continue
            if out_support[a] >= min_support and in_support[b] >= min_support:
                supported.append((a, b))
            else:
                undersampled.append((a, b))
    meta = {
        "out_support": out_support.tolist(),
        "in_support": in_support.tolist(),
        "n_supported": len(supported),
        "n_undersampled": len(undersampled),
        "undersampled_pairs": [[MLB_NAMES[a], MLB_NAMES[b]] for a, b in undersampled],
        "min_support": min_support,
    }
    return supported, meta


def observed_pairs(
    T: np.ndarray, min_count: int, min_support: int,
) -> tuple[list[tuple[int, int]], dict]:
    """Ordered pairs that DO occur, used as the matched control arm.

    The impossible-transition result is only interpretable against the
    controller's performance on transitions it was actually trained on, so the
    same two-phase protocol is run over observed pairs.

    Args:
        T: ``[K, K]`` bout-level transition counts.
        min_count: Minimum observed bout transitions for the pair itself.
        min_support: Minimum endpoint support, as for the impossible arm.

    Returns:
        ``(pairs, meta)``.
    """
    K = T.shape[0]
    out_support, in_support = T.sum(axis=1), T.sum(axis=0)
    pairs = [
        (a, b) for a in range(K) for b in range(K)
        if a != b and T[a, b] >= min_count
        and out_support[a] >= min_support and in_support[b] >= min_support
    ]
    return pairs, {"n_observed": len(pairs), "min_count": min_count,
                   "counts": {f"{MLB_NAMES[a]}->{MLB_NAMES[b]}": int(T[a, b])
                              for a, b in pairs}}


# ---------------------------------------------------------------------------
# Two-phase rollout
# ---------------------------------------------------------------------------

def make_code_windows(
    code_a: int, code_b: int, hold_frames: int, code_stack_size: int,
) -> np.ndarray:
    """Per-timestep forward windows for the two-phase stream ``A`` then ``B``.

    Args:
        code_a: Symbol held for the first ``hold_frames`` steps.
        code_b: Symbol held for the second ``hold_frames`` steps.
        hold_frames: Steps per phase.
        code_stack_size: Forward-context length ``N`` the network expects.

    Returns:
        ``[2 * hold_frames, N]`` float32; row ``t`` is ``seq[t : t + N]``
        with the tail clamped to the final symbol.
    """
    seq = np.concatenate([
        np.full(hold_frames, code_a, dtype=np.int32),
        np.full(hold_frames, code_b, dtype=np.int32),
    ])
    total = seq.shape[0]
    padded = np.concatenate([seq, np.full(code_stack_size, seq[-1], np.int32)])
    windows = np.stack([padded[t:t + code_stack_size] for t in range(total)])
    return windows.astype(np.float32)


def rollout_sequence(
    env, inf_fn, params, ppo_networks,
    code_windows: jnp.ndarray, clip_indices: jnp.ndarray, base_keys: jnp.ndarray,
) -> dict[str, jax.Array]:
    """Roll K bodies through a per-timestep symbol stream.

    Mirrors ``run_code_grid_rollouts._batched_rollout_for_code`` (vmap over
    bodies, ``lax.scan`` over time, identical PRNG threading and recorded
    quantities) except that the code window is scanned rather than held fixed.

    Args:
        code_windows: ``[T, N]`` forward windows, one row per timestep.
        clip_indices: ``[K]`` env-internal reset clip index per body.
        base_keys: ``[K, 2]`` PRNG key per body.

    Returns:
        ``{hidden, qpos, proprio, code, reward}``, each leading ``(K, T, ...)``.
    """
    def single_body(base_key, clip_idx):
        key, reset_key = jax.random.split(base_key)
        state0 = env.reset(reset_key, clip_idx=clip_idx)
        hidden0 = ppo_networks.policy_network.init_hidden(1)

        def step(carry, window):
            state, hidden, key = carry
            new_obs = OrderedDict(state.obs)
            new_obs["kpms_code"] = window
            state = state.replace(obs=new_obs)
            key, subkey = jax.random.split(key)
            batched_obs = jax.tree.map(lambda x: x[None], state.obs)
            action, extras, hidden = inf_fn(params, batched_obs, hidden, subkey)
            action = jax.tree.map(lambda x: x[0], action)
            out = {
                "hidden": hidden[-1][0],
                "qpos": state.data.qpos,
                "proprio": state.obs["proprioception"],
                "code": jnp.asarray(extras["code_idx"]).reshape(-1)[0].astype(jnp.int32),
            }
            new_state = env.step(state, action)
            out["reward"] = new_state.reward
            return (new_state, hidden, key), out

        _, outs = jax.lax.scan(step, (state0, hidden0, key), code_windows)
        return outs

    return jax.vmap(single_body)(base_keys, clip_indices)


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def label_two_phase(
    qpos: np.ndarray, hold_frames: int, settle_frac: float, ctrl_hz: float,
    base_a: str, base_b: str,
) -> dict:
    """Label one body's two-phase trajectory.

    Args:
        qpos: ``[T, nq]`` for a single body.
        hold_frames: Steps per phase.
        settle_frac: Trailing fraction of each phase treated as settled.
        ctrl_hz: Env control rate, passed to the labeler for resampling.

    Returns:
        ``{phase_a, phase_b, bridge}`` behavior names; ``bridge`` is the set of
        behaviors observed strictly between the settled windows that equal
        neither settled label.
    """
    settle = max(1, int(round(hold_frames * settle_frac)))
    a_win = qpos[hold_frames - settle:hold_frames]
    b_win = qpos[2 * hold_frames - settle:]
    a_lab = label_rollout_body(a_win, src_fps=ctrl_hz).behavior
    b_lab = label_rollout_body(b_win, src_fps=ctrl_hz).behavior
    mid = qpos[hold_frames - settle:2 * hold_frames - settle]
    mid_lab = label_rollout_body(mid, src_fps=ctrl_hz)
    # BodyLabel exposes `behavior_segments` as (name, start, end) tuples.
    # Exclude the *commanded* base behaviors, not this body's settled labels:
    # keying off the settled labels counts the target itself as a bridge
    # whenever phase B fails to settle, which conflates an inserted
    # intermediate behavior with a transient, non-persisting success.
    seen = {name for (name, _s, _e) in (mid_lab.behavior_segments or [])}
    bridge = sorted(x for x in seen if x and x not in (base_a, base_b))
    return {"phase_a": a_lab, "phase_b": b_lab, "bridge": bridge}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="mlb decoder checkpoint dir (pin the manuscript step)")
    ap.add_argument("--codes-path",
                    default="outputs/tokenizer/mlb_v1/best_codes.npz")
    ap.add_argument("--reference-data",
                    default="data/rodent/rodent_reference_clips.h5")
    ap.add_argument("--balanced-split",
                    default="data/rodent/rodent_balanced_splits.json")
    ap.add_argument("--labels-path", default="data/rodent/rodent_labels.csv")
    ap.add_argument("--hold-frames", type=int, default=100)
    ap.add_argument("--n-diverse", type=int, default=50)
    ap.add_argument("--n-render", type=int, default=10,
                    help="walk-pose bodies reserved for rendering; a pool lets "
                         "the figure show a body that actually achieved the "
                         "transition without varying the initial pose")
    ap.add_argument("--min-support", type=int, default=10,
                    help="min endpoint bouts for a zero count to be meaningful")
    ap.add_argument("--settle-frac", type=float, default=0.3)
    ap.add_argument("--probe-frame", type=int, default=20)
    ap.add_argument("--probe-frame-max", type=int, default=230,
                    help="Walk is rare (~2%% of frames), so the render-pose "
                         "probe searches a frame range rather than one frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/behavior/impossible_transitions/mlb")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--observed", action="store_true",
                    help="control arm: run transitions that DO occur")
    ap.add_argument("--min-count", type=int, default=1,
                    help="observed mode: minimum bout count for a pair")
    ap.add_argument("--inventory-only", action="store_true",
                    help="compute the impossible-pair inventory and stop")
    args = ap.parse_args()

    # A library import configures the root logger before we get here, so
    # basicConfig alone is a no-op and INFO records are dropped.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    out_dir = Path(_abs(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- inventory -------------------------------------------------------
    d = np.load(_abs(args.codes_path), allow_pickle=True)
    all_codes = d["all_codes"]
    K = int(all_codes.max()) + 1
    T = bout_transition_counts(all_codes, K)
    if args.observed:
        pairs, meta = observed_pairs(T, args.min_count, args.min_support)
        meta["arm"] = "observed"
    else:
        pairs, meta = impossible_pairs(T, args.min_support)
        meta["arm"] = "impossible"
    meta["transition_counts"] = T.tolist()
    meta["pairs"] = [[MLB_NAMES[a], MLB_NAMES[b]] for a, b in pairs]
    (out_dir / "transitions.json").write_text(json.dumps(meta, indent=2))
    log.info(f"{len(pairs)} {meta['arm']} transitions")
    for a, b in pairs:
        log.info(f"  {MLB_NAMES[a]:>9} -> {MLB_NAMES[b]:<9} "
                 f"(count={T[a, b]}, out={T.sum(1)[a]}, in={T.sum(0)[b]})")
    if args.inventory_only:
        return

    # --- controller ------------------------------------------------------
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
        _abs(args.checkpoint))
    ckpt_cfg.kpms_config.codes_path = _abs(args.codes_path)
    assert_decoder_only_settings(0.0, deterministic=True)
    inf_fn = make_inference_fn(ppo_networks, use_rnn=True,
                               deterministic=True, z_e_scale=0.0)
    params = (norm_state, policy_params)
    clip_len = int(ckpt_cfg.env_config.clip_length)
    stack = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    ctrl_hz = 1.0 / float(ckpt_cfg.env_config.ctrl_dt)

    splits = load_balanced_splits(_abs(args.balanced_split))
    train_idx = np.asarray(splits["balanced"]["train_indices"], np.int64)
    test_idx = np.asarray(splits["balanced"]["test_indices"], np.int64)
    payload = load_codes(ckpt_cfg, train_idx, test_idx, clip_len)

    labels = load_label_frames(_abs(args.labels_path), clip_range=(0, 841),
                               n_frames=clip_len, csv_to_h5_subsample=2)
    rng = np.random.RandomState(args.seed)
    diverse = rng.choice(test_idx, size=args.n_diverse, replace=False)
    try:
        walk_clips, probe = _pick_start_clips(
            "Walk", test_idx, labels.behavior, probe_frame=args.probe_frame,
            probe_frame_max=args.probe_frame_max, K=args.n_render,
            seed=args.seed)
    except RuntimeError as e:
        # Walk-pose clips are scarce; shrink the render pool rather than abort.
        log.warning(f"render pool: {e}")
        for k in range(args.n_render - 1, 0, -1):
            try:
                walk_clips, probe = _pick_start_clips(
                    "Walk", test_idx, labels.behavior,
                    probe_frame=args.probe_frame,
                    probe_frame_max=args.probe_frame_max, K=k, seed=args.seed)
                log.warning(f"render pool reduced to K={k}")
                break
            except RuntimeError:
                continue
        else:
            raise
    start_clips = np.concatenate([diverse, walk_clips])  # last = render body
    n_render = len(walk_clips)
    log.info(f"{len(start_clips)} bodies ({args.n_diverse} diverse + {n_render} "
             f"walk-pose @ frame {probe}, render pool = indices "
             f"{args.n_diverse}..{len(start_clips) - 1})")

    # test_codes rows are indexed by position within the test split, not by
    # global clip id, so map through the split index (same as code_grid).
    test_pos = {int(c): i for i, c in enumerate(test_idx)}
    codes_for_clips = np.stack(
        [payload.test_codes[test_pos[int(c)]] for c in start_clips], axis=0)
    env = _build_pose_env(ckpt_cfg, _abs(args.reference_data), clip_len,
                          keep_clips_idx=start_clips, test_codes=codes_for_clips,
                          code_stack_size=stack, probe_frame=probe)

    n_bodies = len(start_clips)
    clip_indices = jnp.arange(n_bodies, dtype=jnp.int32)

    @jax.jit
    def _run(windows, base_keys):
        return rollout_sequence(env, inf_fn, params, ppo_networks,
                                windows, clip_indices, base_keys)

    rows = []
    for pi, (a, b) in enumerate(pairs):
        tag = f"{MLB_NAMES[a]}__{MLB_NAMES[b]}"
        npz = out_dir / f"rollout_{tag}.npz"
        if args.reuse and npz.exists():
            agg = dict(np.load(npz, allow_pickle=True))
        else:
            windows = jnp.asarray(make_code_windows(a, b, args.hold_frames, stack))
            base_keys = jnp.stack([
                jax.random.PRNGKey(args.seed + 1_000_003 * (pi + 1) + k)
                for k in range(n_bodies)])
            outs = _run(windows, base_keys)
            agg = {k: np.asarray(v) for k, v in outs.items()}
            agg["start_clips"] = start_clips
            agg["code_a"], agg["code_b"] = np.int32(a), np.int32(b)
            np.savez_compressed(npz, **agg)

        base_a, base_b = MLB_NAMES[a].split("_")[0], MLB_NAMES[b].split("_")[0]
        n_all = agg["qpos"].shape[0]
        labs_all = [label_two_phase(agg["qpos"][i], args.hold_frames,
                                    args.settle_frac, ctrl_hz, base_a, base_b)
                    for i in range(n_all)]
        # Per-body outcomes drive figure selection (deliverable 3): the render
        # script picks a pool body that actually achieved the transition.
        (out_dir / f"labels_{tag}.json").write_text(json.dumps({
            "code_a": a, "code_b": b, "base_a": base_a, "base_b": base_b,
            "n_diverse": int(args.n_diverse),
            "render_pool": list(range(int(args.n_diverse), n_all)),
            "bodies": [{"idx": i, **labs_all[i]} for i in range(n_all)],
        }, indent=2))
        labs = labs_all[:args.n_diverse]   # stats over diverse bodies only
        reached = np.mean([l["phase_b"] == MLB_NAMES[b].split("_")[0]
                           for l in labs])
        held_a = np.mean([l["phase_a"] == MLB_NAMES[a].split("_")[0]
                          for l in labs])
        bridged = np.mean([len(l["bridge"]) > 0 for l in labs])
        bridges = {}
        for l in labs:
            for x in l["bridge"]:
                bridges[x] = bridges.get(x, 0) + 1
        rows.append({
            "from": MLB_NAMES[a], "to": MLB_NAMES[b],
            "pair_count": int(T[a, b]),
            "src_out_bouts": int(T.sum(1)[a]), "tgt_in_bouts": int(T.sum(0)[b]),
            "held_A": round(float(held_a), 3),
            "reached_B": round(float(reached), 3),
            "bridge_rate": round(float(bridged), 3),
            "bridge_behaviors": ";".join(
                f"{k}:{v}" for k, v in sorted(bridges.items(), key=lambda x: -x[1])),
            "mean_reward": round(float(np.mean(agg["reward"])), 3),
        })
        log.info(f"  {tag:<24} held_A={held_a:.2f} reached_B={reached:.2f} "
                 f"bridge={bridged:.2f}")

    import csv
    with open(out_dir / "qualitative.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {out_dir/'qualitative.csv'} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
