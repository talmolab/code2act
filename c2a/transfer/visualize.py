"""Composite visualization for Option-B code-transfer policies.

Layout (matches the design sketch):

    +----------------------------+-------+
    |                            | code  |   right column = "popular code"
    |        CENTER              | tile  |   tiles: each is a PRECOMPUTED clip of
    |   (policy-driven rodent)   | tile  |   the rodent HOLDING that code for
    |                            | tile  |   `hold_frames`. DIM by default,
    +----------------------------+ ...   |   BRIGHT (+colored border) on the
    |  emitted-code timeline bar |       |   frame the center animal emits it.
    +----------------------------+-------+

Two entry points:

* ``precompute_code_tiles`` — render each popular code held for ``hold_frames``
  (decoder-only) ONCE and cache the frames to an ``.npz``. Run before training /
  before visualizing; the viz just loads and plays them.
* ``render_composite`` — roll out the trained policy, render the center, stitch
  the bottom code-bar + right code-tile column, write an mp4. Returns the path so
  it can be ``wandb.Video``-logged either post-training (this module's ``main``)
  or live during training (via :func:`make_video_logger`, droppable into a brax
  ``policy_params_fn``).
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import c2a.transfer  # noqa: F401  (record_scalar shim)

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from c2a.paths import get_repo_root
from c2a.rendering_utils import add_code_transition_bar, get_nature_colormap
from c2a.transfer.decoder_adapter import (
    decode_step,
    extract_decoder_proprio,
    load_frozen_decoder,
)
from c2a.transfer.networks import make_categorical_ppo_networks, make_inference_fn

RENDER_CAMERA = "close_profile"
DEFAULT_DECODER = "model_checkpoints/260603_180852_140475"
DEFAULT_LABELS = "outputs/code_grid/code_grid_rollouts/best_vae/labels/code_labels.json"


# --------------------------------------------------------------------------
# env builders
# --------------------------------------------------------------------------
def _bare_rearing(repo: Path):
    """Bare rearing env on the decoder's training body (rodent.xml)."""
    from vnl_playground.tasks.rodent import consts, rearing

    cfg = rearing.default_config()
    cfg.walker_xml_path = consts.RODENT_XML_PATH
    return rearing.Rearing(config=cfg)


def _render_states(env, states, width, height, camera=RENDER_CAMERA):
    """env.render with a free-camera fallback if the named camera is missing."""
    try:
        return list(env.render(states, camera=camera, height=height, width=width))
    except Exception:  # noqa: BLE001
        return list(env.render(states, height=height, width=width))


# --------------------------------------------------------------------------
# popular codes
# --------------------------------------------------------------------------
def popular_codes(frozen, repo: Path, top_n: int, must_include=None) -> list[int]:
    """Top-N codes by data usage, unioned with ``must_include`` (e.g. the codes
    the policy actually emits, so the lighting-up is guaranteed visible)."""
    import json

    from c2a.codegen.code_sources import load_codes

    split = repo / "data/rodent/rodent_balanced_splits.json"
    with open(split) as f:
        sp = json.load(f)["balanced"]
    payload = load_codes(
        frozen.cfg,
        np.asarray(sp["train_indices"]),
        np.asarray(sp["test_indices"]),
        int(frozen.cfg.env_config.clip_length),
    )
    allc = np.concatenate(
        [payload.train_codes.reshape(-1), payload.test_codes.reshape(-1)]
    )
    counts = np.bincount(allc, minlength=frozen.num_codes)
    ranked = [int(c) for c in np.argsort(-counts) if counts[c] > 0]
    chosen = ranked[:top_n]
    for c in must_include or []:
        if int(c) not in chosen:
            chosen.append(int(c))
    return chosen


def load_code_labels(repo: Path) -> dict[int, str]:
    import json

    p = repo / DEFAULT_LABELS
    if not p.exists():
        return {}
    d = json.load(open(p)).get("codes", {})
    return {int(k): v.get("label", str(k)) for k, v in d.items()}


# --------------------------------------------------------------------------
# precompute per-code held tiles
# --------------------------------------------------------------------------
def precompute_code_tiles(
    decoder_ckpt: str,
    codes: list[int],
    repo: Path,
    out_npz: str | Path,
    hold_frames: int = 500,
    width: int = 200,
    height: int = 150,
) -> dict:
    """Render each code held for ``hold_frames`` (decoder-only) and cache frames.

    Saved npz: ``tile_{code}`` -> uint8 ``[hold_frames, height, width, 3]`` plus
    ``codes`` and ``hold_frames``.
    """
    frozen = load_frozen_decoder(str(repo / decoder_ckpt))
    env = _bare_rearing(repo)
    jreset = jax.jit(env.reset)
    jstep = jax.jit(env.step)
    out: dict = {"codes": np.asarray(codes, dtype=np.int32), "hold_frames": hold_frames}
    key = jax.random.PRNGKey(0)
    for c in codes:
        key, rk = jax.random.split(key)
        state = jreset(rk)
        hidden = frozen.init_hidden(1)
        states = [state]
        for _t in range(hold_frames - 1):
            key, sk = jax.random.split(key)
            proprio = extract_decoder_proprio(state.obs)
            torque, _e, hidden = decode_step(frozen, jnp.int32(c), proprio, hidden, sk)
            state = jstep(state, torque)
            states.append(state)
        frames = _render_states(env, states, width, height)
        out[f"tile_{int(c)}"] = np.asarray(frames, dtype=np.uint8)
        print(f"  precomputed tile code {c}: {len(frames)} frames", flush=True)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)
    print(f"saved code tiles -> {out_npz}", flush=True)
    return out


def load_tiles(npz_path: str | Path) -> tuple[dict[int, np.ndarray], list[int]]:
    d = np.load(npz_path)
    codes = [int(c) for c in d["codes"]]
    return {c: d[f"tile_{c}"] for c in codes}, codes


# --------------------------------------------------------------------------
# center rollout (trained policy)
# --------------------------------------------------------------------------
def rollout_center(env, make_policy, params, key, max_steps, width, height):
    """Roll the transfer env with the trained policy; return (frames, codes)."""
    policy = make_policy(params, deterministic=True)
    jstep = jax.jit(env.step)
    jreset = jax.jit(env.reset)
    jpol = jax.jit(policy)
    state = jreset(key)
    states, codes = [state], []
    for _t in range(max_steps):
        key, k = jax.random.split(key)
        act, _ = jpol(state.obs, k)
        codes.append(int(act))
        state = jstep(state, act)
        states.append(state)
    frames = _render_states(env, states[:-1], width, height)
    return frames, codes


# --------------------------------------------------------------------------
# compositing
# --------------------------------------------------------------------------
def _tile_with_chrome(tile_img, active, color, label, tile_w, tile_h):
    img = cv2.resize(tile_img, (tile_w, tile_h)).astype(np.float32)
    if not active:
        img = img * 0.45  # dim when idle (still readable)
    img = img.astype(np.uint8).copy()
    bc = tuple(int(x) for x in (color if active else (90, 90, 90)))
    th = 3 if active else 1
    cv2.rectangle(img, (0, 0), (tile_w - 1, tile_h - 1), bc[::-1], th)  # BGR
    txt_color = (255, 255, 255) if active else (160, 160, 160)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                txt_color[::-1], 1, cv2.LINE_AA)
    return img


def compose(center_frames, codes, tiles, code_list, labels, colors,
            bar_height=48, tile_w=200):
    """Stitch center + bottom code-bar + right code-tile column per frame."""
    T = len(center_frames)
    Hc, Wc, _ = center_frames[0].shape
    col_h = Hc + bar_height
    N = len(code_list)
    tile_h = col_h // N
    out = []
    for t in range(T):
        # center + bottom emitted-code timeline (reuse the v9 bar)
        left = add_code_transition_bar(
            center_frames[t], t, np.asarray(codes), colors,
            bar_height=bar_height, show_code_label=True,
        )
        if left.shape[0] < col_h:  # pad to exact column height
            left = np.vstack([left, np.zeros((col_h - left.shape[0], Wc, 3), np.uint8)])
        # right code-tile column
        col = np.zeros((col_h, tile_w, 3), np.uint8)
        cur = codes[t]
        for i, c in enumerate(code_list):
            tf = tiles[c][t % len(tiles[c])]
            chrome = _tile_with_chrome(
                tf, c == cur, colors[c], f"c{c} {labels.get(c, '')}", tile_w, tile_h
            )
            col[i * tile_h:(i + 1) * tile_h] = chrome
        out.append(np.hstack([left[:col_h], col]))
    return out


def write_mp4(frames, path, fps=50):
    import imageio.v2 as imageio

    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    return str(path)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def render_composite(decoder_ckpt, policy_params, tiles_npz, out_mp4, repo: Path,
                     max_steps=250, width=600, height=480, seed=0, fps=50):
    frozen = load_frozen_decoder(str(repo / decoder_ckpt))
    from c2a.transfer.train import make_env

    env, _, _ = make_env(decoder_ckpt, "mamba", 2, repo)
    nets = make_categorical_ppo_networks((frozen.proprio_size + 52 + frozen.num_codes,),
                                         frozen.num_codes)
    make_policy = make_inference_fn(nets)
    params = (policy_params[0], policy_params[1])  # (normalizer, policy)

    tiles, code_list = load_tiles(tiles_npz)
    labels = load_code_labels(repo)
    colors = get_nature_colormap(frozen.num_codes)

    frames, codes = rollout_center(
        env, make_policy, params, jax.random.PRNGKey(seed), max_steps, width, height
    )
    comp = compose(frames, codes, tiles, code_list, labels, colors)
    path = write_mp4(comp, out_mp4, fps=fps)
    print(f"wrote composite video -> {path}", flush=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_ckpt", default=DEFAULT_DECODER)
    p.add_argument("--policy_params", default="outputs/transfer/walkrear_mamba_v1/ppo_params.pkl")
    p.add_argument("--tiles_npz", default="outputs/transfer/walkrear_mamba_v1/code_tiles.npz")
    p.add_argument("--out", default="outputs/transfer/walkrear_mamba_v1/composite.mp4")
    p.add_argument("--top_n", type=int, default=12)
    p.add_argument("--hold_frames", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=250)
    p.add_argument("--precompute_only", action="store_true")
    p.add_argument("--wandb_run", default=None, help="wandb run path to log the video to")
    args = p.parse_args()

    repo = Path(get_repo_root())
    frozen = load_frozen_decoder(str(repo / args.decoder_ckpt))

    # determine popular codes (union with the policy's emitted codes)
    must = []
    if Path(args.policy_params).exists():
        params = pickle.load(open(args.policy_params, "rb"))
        from c2a.transfer.train import make_env

        env, _, _ = make_env(args.decoder_ckpt, "mamba", 2, repo)
        nets = make_categorical_ppo_networks(
            (frozen.proprio_size + 52 + frozen.num_codes,), frozen.num_codes
        )
        mp = make_inference_fn(nets)
        _f, emitted = rollout_center(
            env, mp, (params[0], params[1]), jax.random.PRNGKey(1),
            args.max_steps, 200, 160
        )
        must = sorted(set(emitted))
    codes = popular_codes(frozen, repo, args.top_n, must_include=must)
    print(f"popular codes (+policy-emitted): {codes}", flush=True)

    if not Path(args.tiles_npz).exists():
        precompute_code_tiles(args.decoder_ckpt, codes, repo, args.tiles_npz,
                              hold_frames=args.hold_frames)
    if args.precompute_only:
        return

    params = pickle.load(open(args.policy_params, "rb"))
    path = render_composite(
        args.decoder_ckpt, params, args.tiles_npz, args.out, repo,
        max_steps=args.max_steps
    )
    if args.wandb_run:
        import wandb

        run = wandb.init(id=args.wandb_run.split("/")[-1], resume="allow",
                         project="code2act-transfer")
        run.log({"videos/composite": wandb.Video(path, format="mp4")})
        run.finish()


if __name__ == "__main__":
    main()
