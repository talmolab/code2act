"""Chained-symbol rollouts and data for the project page's sequence viewer.

Rolls the paper's MLB controller (step 208, decoder-only, deterministic) through
a few preset symbol sequences with the same rollout function as the
unobserved-pair experiment (``run_impossible_transitions.rollout_sequence``),
from diverse test-clip start poses. For each preset the rendered body is the
one with the highest labeler agreement over phases (ties broken by reward);
the exported JSON records both its agreement and the median over bodies.

Exports, per preset, a trail video plus a JSON with the commanded symbol per
frame, the independent labeler's readout, root speed and height, and the GRU
hidden state projected onto PCs fit to the held-code rollouts of every MLB
symbol (the same basis drawn behind the trace).

Usage::

    LD_LIBRARY_PATH=<glvnd libEGL dir> C2A_ROOT=$PWD \\
        python -m c2a.website.run_sequences
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

from c2a.codegen.code_sources import load_codes
from c2a.experiments.behavior.run_impossible_transitions import rollout_sequence
from c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled import (
    _build_pose_env,
)
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint, make_inference_fn,
)
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.shared.decoder_only import assert_decoder_only_settings
from c2a.experiments.shared.topomimic_labeler import label_rollout_body
from c2a.paths import get_repo_root
from c2a.website.render_lib import (
    TrailStyle, build_env, build_scene, mlb_rgb, render_trail_video,
    write_mp4, write_poster,
)
from c2a.website.export_data import PUBLIC_DATA
from c2a.website.render_videos import DATA, MEDIA, MLB_SHORT

log = logging.getLogger(__name__)
ROOT = get_repo_root()
CKPT = "model_checkpoints/260629_212713_669229_step208"
CODES = "outputs/tokenizer/mlb_v1/best_codes.npz"
N = {n: i for i, n in enumerate(MLB_SHORT)}

#: Preset sequences: (title, blurb, [(symbol, hold steps at 100 Hz), ...]).
PRESETS = {
    "tour": ("A short tour",
             "Six symbols the controller executes reliably, chained with switches "
             "that all occur in the corpus.",
             [("Walk_L", 300), ("Rear_L", 300), ("Walk_L", 300), ("Immobile", 300),
              ("Rear_R", 300), ("Turn_R", 300)]),
    "unseen": ("Transitions the corpus never contains",
               "Every switch in this chain has zero bout-level count in the data.",
               [("Rear_L", 250), ("Walk_R", 250), ("Rear_S", 250), ("Walk_L", 250),
                ("Rear_R", 250), ("Walk_S", 250), ("Rear_S", 250)]),
    "quick": ("Rapid switching",
              "Short holds: the body has about one second per symbol.",
              [("Walk_S", 100), ("Rear_S", 100), ("Walk_L", 100), ("Immobile", 100),
               ("Turn_R", 100), ("Walk_S", 100), ("Rear_L", 100), ("Walk_R", 100),
               ("Immobile", 100), ("Walk_S", 100)]),
}


def windows_for(seq: list[tuple[str, int]], stack: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-step forward code windows ``[T, stack]`` and the commanded stream."""
    codes = np.concatenate([np.full(h, N[s], np.int32) for s, h in seq])
    padded = np.concatenate([codes, np.full(stack, codes[-1], np.int32)])
    win = np.stack([padded[t:t + stack] for t in range(len(codes))])
    return win.astype(np.float32), codes


def phase_agreement(qpos: np.ndarray, seq, settle: float = 0.4) -> float:
    """Fraction of phases whose settled window the labeler reads as commanded."""
    hits, t = 0, 0
    for s, h in seq:
        w = qpos[t + int(h * (1 - settle)):t + h]
        lab = label_rollout_body(w, src_fps=100.0)
        hits += int(lab.behavior == s.split("_")[0])
        t += h
    return hits / len(seq)


def hidden_basis(n_per_code: int = 12, seed: int = 0) -> tuple[PCA, list[dict]]:
    """PCA basis of GRU states from MLB held-code rollouts, plus code clouds."""
    rng = np.random.RandomState(seed)
    gdir = ROOT / "outputs/code_grid/code_grid_rollouts/mlb/data"
    pool, clouds = [], []
    for c in range(9):
        d = np.load(gdir / f"rollout_random_code{c:03d}.npz")
        h, q = d["hidden"], d["qpos"]
        ok = np.flatnonzero(np.isfinite(q).all(axis=(1, 2)))
        bodies = rng.choice(ok, n_per_code, replace=False)
        hs = h[bodies, 300::10].reshape(-1, h.shape[-1])
        pool.append(hs)
        clouds.append((c, hs))
    pca = PCA(n_components=3, random_state=seed).fit(np.concatenate(pool))
    out = []
    for c, hs in clouds:
        z = pca.transform(hs[::3])
        out.append({"code": c, "name": MLB_SHORT[c], "color": list(mlb_rgb(c)),
                    "points": np.round(z[:, :2], 3).tolist()})
    return pca, out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reuse", action="store_true",
                    help="re-select and re-render from saved rollouts")
    reuse = ap.parse_args().reuse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    cfg, norm_state, policy_params, nets = load_moseq_checkpoint(str(ROOT / CKPT))
    cfg.kpms_config.codes_path = str(ROOT / CODES)
    assert_decoder_only_settings(0.0, deterministic=True)
    inf_fn = make_inference_fn(nets, use_rnn=True, deterministic=True, z_e_scale=0.0)
    params = (norm_state, policy_params)
    clip_len = int(cfg.env_config.clip_length)
    stack = int(cfg.network_config.get("code_stack_size", 1))

    splits = load_balanced_splits(str(ROOT / "data/rodent/rodent_balanced_splits.json"))
    train_idx = np.asarray(splits["balanced"]["train_indices"], np.int64)
    test_idx = np.asarray(splits["balanced"]["test_indices"], np.int64)
    payload = load_codes(cfg, train_idx, test_idx, clip_len)
    rng = np.random.RandomState(0)
    starts = rng.choice(test_idx, size=48, replace=False)
    test_pos = {int(c): i for i, c in enumerate(test_idx)}
    codes_for = np.stack([payload.test_codes[test_pos[int(c)]] for c in starts])
    env = _build_pose_env(cfg, str(ROOT / "data/rodent/rodent_reference_clips.h5"),
                          clip_len, keep_clips_idx=starts, test_codes=codes_for,
                          code_stack_size=stack, probe_frame=20)
    clip_indices = jnp.arange(len(starts), dtype=jnp.int32)

    pca, clouds = hidden_basis()
    renv = build_env()
    style = TrailStyle(n_ghosts=5, ghost_every=30, cam_distance=0.95)
    scene = build_scene(renv, style.n_ghosts + 1, 1280, 720)

    out_dir = ROOT / "outputs/website/sequences"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for pi, (key, (title, blurb, seq)) in enumerate(PRESETS.items()):
        win, cmd = windows_for(seq, stack)
        cache = out_dir / f"{key}.npz"
        if reuse and cache.exists():
            agg = {k: v for k, v in np.load(cache).items()
                   if k not in ("commanded", "starts")}
        else:
            keys = jnp.stack([jax.random.PRNGKey(1000 * (pi + 1) + k)
                              for k in range(len(starts))])
            outs = jax.jit(lambda w, k: rollout_sequence(
                env, inf_fn, params, nets, w, clip_indices, k))(jnp.asarray(win), keys)
            agg = {k: np.asarray(v) for k, v in outs.items()}
            np.savez_compressed(cache, **agg, commanded=cmd, starts=starts)

        q_all = agg["qpos"]
        ok = np.flatnonzero(np.isfinite(q_all).all(axis=(1, 2)))
        agree = np.array([phase_agreement(q_all[b], seq) for b in ok])
        rew = agg["reward"][ok].mean(1)
        # Highest command/readout agreement, ties broken by reward.
        order = np.lexsort((-rew, -agree))
        b = int(ok[order[0]])
        log.info(f"[{key}] agreement median {np.median(agree):.2f} (min {agree.min():.2f}, "
                 f"max {agree.max():.2f}); body {b} agreement {agree[order[0]]:.2f}")

        q = q_all[b]
        T = len(cmd)
        frames = render_trail_video(scene, q, np.array([mlb_rgb(c) for c in cmd]), style,
                                    range(0, T, 2))
        rel = f"sequences/{key}"
        write_mp4(frames, MEDIA / f"{rel}.mp4", 50, crf=24)
        write_poster(frames[0], MEDIA / f"{rel}.jpg")

        lab = label_rollout_body(q, src_fps=100.0)
        readout = [{"behavior": n, "start": s / 30.0, "end": e / 30.0}
                   for n, s, e in lab.behavior_segments]
        z = pca.transform(agg["hidden"][b][::2])[:, :2]
        speed = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) * 100.0
        speed = np.convolve(np.r_[speed, speed[-1]], np.ones(15) / 15, mode="same")
        segs, t = [], 0
        for s, h in seq:
            segs.append({"name": s, "code": N[s], "start": t / 100.0, "end": (t + h) / 100.0,
                         "color": list(mlb_rgb(N[s]))})
            t += h
        payload_json = {
            "key": key, "title": title, "blurb": blurb, "fps": 50, "duration": T / 100.0,
            "src": f"media/{rel}.mp4", "poster": f"media/{rel}.jpg",
            "segments": segs, "readout": readout,
            "agreement": round(float(agree[order[0]]), 3),
            "agreement_median": round(float(np.median(agree)), 3), "n_bodies": int(len(ok)),
            "hidden": np.round(z, 3).tolist(),
            "speed": np.round(speed[::2], 3).tolist(),
            "height": np.round(q[::2, 2], 4).tolist(),
        }
        (PUBLIC_DATA / f"sequence_{key}.json").write_text(json.dumps(payload_json))
        index.append({"key": key, "title": title})
    (PUBLIC_DATA / "sequence_clouds.json").write_text(json.dumps({
        "explained": np.round(pca.explained_variance_ratio_[:2], 3).tolist(),
        "clouds": clouds}))
    (DATA / "sequences.json").write_text(json.dumps(index))
    scene.close()


if __name__ == "__main__":
    main()
