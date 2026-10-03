"""Render the project-page videos from the paper's saved rollouts.

Every video is re-rendered from rollouts the manuscript's figures were made
from, so the motion on the page is the motion behind the reported numbers:

* ``atlas``     held-code rollouts (Fig 2c, 11, 12): one body per code.
* ``roundtrip`` the same rollouts from step 0: four bodies started from
  different poses converge onto the commanded symbol (Fig 3a).
* ``kick``      perturbation rollouts (Fig 8): unperturbed and kicked twins.
* ``pairs``     unobserved symbol pairs (Fig 13): hold A, then B.
* ``streams``   KID rollouts (Table 1): original, GPT and random code streams.

Bodies are chosen by rule, not by eye: the body whose labeler readout matches
the code's modal behavior and whose path length is closest to the median of
those bodies (``atlas``), a seeded random draw (``roundtrip``), the body with
the median kick displacement (``kick``), and the body the paper's Fig 13 uses
(``pairs``).

Usage::

    LD_LIBRARY_PATH=<glvnd libEGL dir> C2A_ROOT=$PWD \\
        python -m c2a.website.render_videos atlas roundtrip kick pairs streams
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from c2a.codegen.shared.mlb_labels import MLB_LABELS
from c2a.paths import get_repo_root
from c2a.website.render_lib import (
    TrailStyle, behavior_rgb, build_env, build_scene, mlb_rgb,
    render_trail_video, write_mp4, write_poster,
)

log = logging.getLogger(__name__)

ROOT = get_repo_root()
SITE = ROOT / "Code2Act-website"
MEDIA = SITE / "public" / "media"
DATA = SITE / "src" / "data"

#: Paper run directories per steering strategy.
RUNS = {"mlb": "mlb", "kpms": "kpms", "kvae": "best_vae"}
MLB_SHORT = ["Walk_L", "Walk_R", "Walk_S", "Rear_L", "Rear_R", "Rear_S",
             "Turn_L", "Turn_R", "Immobile"]
FPS = 50          # playback rate; rollouts are 100 Hz, so every 2nd step
STEP = 2


def _grid_dir(strategy: str) -> Path:
    return ROOT / "outputs/code_grid/code_grid_rollouts" / RUNS[strategy]


def _labels(strategy: str) -> dict:
    return json.load(open(_grid_dir(strategy) / "labels/code_labels.json"))


def _rollout(strategy: str, code: int) -> dict:
    return dict(np.load(_grid_dir(strategy) / f"data/rollout_random_code{code:03d}.npz"))


def _path_len(q: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(q[..., :2], axis=-2), axis=-1).sum(-1)


def _finite_bodies(q: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isfinite(q).all(axis=(1, 2)))


def typical_body(qpos: np.ndarray, rec: dict) -> int:
    """Body whose readout matches the modal label, with median path length."""
    beh, _, d = rec["label"].partition("_")
    want_dir = {"L": "Left", "R": "Right", "S": "Straight"}.get(d)
    ok = set(_finite_bodies(qpos).tolist())
    cands = [p["body_idx"] for p in rec["per_body"]
             if p["behavior"] == beh and (beh == "Immobile" or p["direction"] == want_dir)
             and p["body_idx"] in ok]
    if not cands:
        cands = sorted(ok)
    L = _path_len(qpos[cands])
    return int(cands[int(np.argmin(np.abs(L - np.median(L))))])


def _write_manifest(name: str, payload) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(payload, indent=1))


class _Scenes:
    """Cache of compiled scenes keyed by (copies, width, height)."""

    def __init__(self):
        self.env = build_env()
        self._c: dict = {}

    def get(self, n_copies: int, w: int, h: int):
        k = (n_copies, w, h)
        if k not in self._c:
            self._c[k] = build_scene(self.env, n_copies, w, h)
        return self._c[k]


def _top_codes(strategy: str, n: int) -> list[int]:
    from c2a.figures.render_code_atlas_grouped import load_run_meta

    top, _, _ = load_run_meta(_grid_dir(strategy), n)
    return [int(c) for c in top]


# ---------------------------------------------------------------- atlas
def render_atlas(sc: _Scenes, top: int = 15) -> None:
    """One video per code: a typical body holding it (Fig 2c / 11 / 12)."""
    # Locomotor codes cover ~2 m in the window; postural ones stay in place.
    styles = {"Walk": TrailStyle(cam_distance=1.15, follow_window=80),
              "Turn": TrailStyle(cam_distance=1.0)}
    default = TrailStyle(cam_distance=0.62)
    W, H = 960, 640
    out = {}
    for strategy in ("mlb", "kpms", "kvae"):
        labels = _labels(strategy)["codes"]
        codes = list(range(9)) if strategy == "mlb" else _top_codes(strategy, top)
        usage = np.load(_grid_dir(strategy) / "data/usage_rank.npz")
        use = {int(c): int(u) for c, u in zip(usage["ranked_codes"], usage["usage"])} \
            if "ranked_codes" in usage.files else {}
        items = []
        for c in codes:
            rec = labels[str(c)]
            q = _rollout(strategy, c)["qpos"]
            b = typical_body(q, rec)
            color = mlb_rgb(c) if strategy == "mlb" else behavior_rgb(rec["label"])
            style = styles.get(rec["label"].split("_")[0], default)
            frames = render_trail_video(
                sc.get(style.n_ghosts + 1, W, H), q[b],
                np.tile(color, (q.shape[1], 1)), style, range(400, 1000, STEP))
            rel = f"atlas/{strategy}/code{c:03d}"
            write_mp4(frames, MEDIA / f"{rel}.mp4", FPS, crf=25)
            write_poster(frames[len(frames) // 2], MEDIA / f"{rel}.jpg")
            items.append({
                "code": c,
                "name": MLB_SHORT[c] if strategy == "mlb" else f"code {c}",
                "readout": rec["label"],
                "behavior": rec["label"].split("_")[0],
                "color": [round(x, 3) for x in color],
                "usage": use.get(c),
                "body": b,
                "src": f"media/{rel}.mp4",
                "poster": f"media/{rel}.jpg",
            })
            log.info(f"atlas {strategy} code {c} body {b}")
        out[strategy] = items
    _write_manifest("atlas", out)


# ------------------------------------------------------------ roundtrip
def _tile(panels: list[list[np.ndarray]], cols: int, gap: int = 6) -> list[np.ndarray]:
    n = len(panels)
    rows = (n + cols - 1) // cols
    h, w, _ = panels[0][0].shape
    out = []
    for t in range(len(panels[0])):
        canvas = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3),
                         255, np.uint8)
        for i, p in enumerate(panels):
            r, c = divmod(i, cols)
            canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = p[t]
        out.append(canvas)
    return out


def roundtrip_readout(q: np.ndarray, settle_frac: float = 0.2) -> str:
    """Directional readout of one held-code body, exactly as Fig 3a scores it.

    ``run_code_roundtrip`` (``--segmenter mlb``) drops the leading
    ``settle_frac`` of the rollout, labels the rest with the auto-labeler, and
    crosses the behavior with that body's settled-window net yaw.
    """
    from c2a.experiments.ic_contraction.run_code_roundtrip import (
        _instance_nine, _settled_net_yaw,
    )
    from c2a.experiments.shared.topomimic_labeler import label_rollout_body

    lo = int(settle_frac * q.shape[0])
    beh = label_rollout_body(np.asarray(q[lo:], np.float64), src_fps=100.0,
                             groom_to_immobile=True).behavior
    r = _instance_nine(beh, float(_settled_net_yaw(q[None], settle_frac)[0]))
    return MLB_SHORT[r[1]] if r is not None else beh


def render_roundtrip(sc: _Scenes, n_bodies: int = 4, seed: int = 0,
                     scored: int = 60) -> None:
    """Four bodies from different start poses converging on each MLB symbol.

    Bodies are a seeded draw from the ``scored`` bodies the round trip
    evaluates, and each readout is computed with the round trip's own rule.
    """
    style = TrailStyle(n_ghosts=3, ghost_every=30, cam_distance=0.85, follow_window=60)
    W, H = 640, 440
    rng = np.random.RandomState(seed)
    out = []
    for c in range(9):
        q = _rollout("mlb", c)["qpos"]
        pool = _finite_bodies(q[:scored])
        bodies = rng.choice(pool, n_bodies, replace=False)
        panels = [render_trail_video(
            sc.get(style.n_ghosts + 1, W, H), q[b], np.tile(mlb_rgb(c), (q.shape[1], 1)),
            style, range(0, 800, STEP)) for b in bodies]
        frames = _tile(panels, 2)
        rel = f"roundtrip/{MLB_SHORT[c]}"
        write_mp4(frames, MEDIA / f"{rel}.mp4", FPS, crf=25)
        write_poster(frames[-1], MEDIA / f"{rel}.jpg")
        readouts = [roundtrip_readout(q[b]) for b in bodies]
        out.append({"code": c, "name": MLB_SHORT[c], "color": list(mlb_rgb(c)),
                    "bodies": [int(b) for b in bodies], "readouts": readouts,
                    "src": f"media/{rel}.mp4", "poster": f"media/{rel}.jpg"})
        log.info(f"roundtrip {MLB_SHORT[c]} bodies {bodies.tolist()} -> {readouts}")
    _write_manifest("roundtrip", out)


# ----------------------------------------------------------------- kick
KICK_PICKS = {"mlb": {"Rear": 4, "Walk": 0}}
RED = np.array([0.85, 0.12, 0.10])


def _posture_gap(qa: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Per-step posture distance between kicked and unperturbed twins.

    Joint angles plus (scaled) root height; heading and position are ignored,
    so a body that recovers its behavior facing a new direction counts as
    recovered.
    """
    dz = 20.0 * (qa[..., 2] - q0[..., 2])
    dj = qa[..., 7:] - q0[..., 7:]
    return np.sqrt(dz ** 2 + (dj ** 2).sum(-1))


def render_kick(sc: _Scenes) -> None:
    """Unperturbed and kicked twins at every impulse size (Fig 8).

    For each amplitude the body shown is the best-recovering one among those
    that took at least a median-sized shove: the smallest end-of-episode
    posture gap to its unperturbed twin relative to its peak gap.
    """
    style = TrailStyle(n_ghosts=4, ghost_every=30, cam_distance=1.1, follow_window=60)
    W, H = 800, 600
    out = {}
    # The page shows the MLB controller only.
    for strategy, run in (("mlb", RUNS["mlb"]),):
        pdir = ROOT / "outputs/perturbation/perturbation_recovery" / run
        summ = json.load(open(pdir / "perturbation_summary.json"))
        tp = int(summ["t_perturb"])
        amps = [float(a) for a in summ["amplitudes"] if float(a) > 0]
        beh_of = {int(c): v["behavior"] for c, v in summ["codes"].items()}
        items = []
        # The KPMS and KVAE "walk" codes barely translate in these rollouts
        # (median end speed ~0.02 m/s), so only MLB gets a walk example.
        for beh in (("Rear", "Walk") if strategy == "mlb" else ("Rear",)):
            c = KICK_PICKS.get(strategy, {}).get(beh)
            if c is None:
                c = next(cc for cc in sorted(beh_of) if beh_of[cc] == beh)
            q0 = np.load(pdir / f"data/rollout_code{c:03d}_amp0.npz")["qpos"]
            base = np.array(mlb_rgb(c) if strategy == "mlb"
                            else behavior_rgb(f"{beh}_S"))
            T = q0.shape[1]
            w = np.zeros(T)
            k = np.arange(T) - tp
            w[k >= 0] = np.exp(-k[k >= 0] / 120.0)
            col_kick = base[None] * (1 - w[:, None]) + RED[None] * w[:, None]
            col_base = np.tile(base, (T, 1))
            rng = range(tp - 200, T, STEP)
            name = MLB_SHORT[c] if strategy == "mlb" else f"code {c}"
            item = {"behavior": beh, "code": c, "name": name, "kick_time": 2.0,
                    "duration": len(rng) / FPS, "amps": {}}
            for a in amps:
                qa = np.load(pdir / f"data/rollout_code{c:03d}_amp{a:g}.npz")["qpos"]
                ok = np.intersect1d(_finite_bodies(q0), _finite_bodies(qa))
                gap = _posture_gap(qa[ok], q0[ok])
                peak = gap[:, tp:tp + 120].max(1)
                final = gap[:, -100:].mean(1)
                strong = peak >= np.median(peak)
                # Stay on the 20 m floor (an a = 4 kick can launch a body off it).
                on_floor = np.abs(qa[ok, :, :2]).max(axis=(1, 2)) < 9.0
                if beh == "Walk":
                    # A recovered walker is walking steadily again at its twin's
                    # pace; a per-frame posture gap would penalize a mere
                    # gait-phase offset.
                    def _gait(q):
                        xy = q[:, -300:, :2]
                        step = np.linalg.norm(xy[:, 10::10] - xy[:, :-10:10], axis=-1) * 10.0
                        return step.mean(1), (step > 0.15).mean(1)
                    (v0, m0), (va, ma) = _gait(q0[ok]), _gait(qa[ok])
                    walking = on_floor & (m0 >= 0.8) & (ma >= 0.8) & (v0 > 0.25)
                    ratio = np.where(walking, np.abs(va - v0) / np.maximum(v0, 1e-6),
                                     np.inf)
                else:
                    ratio = np.where(strong & on_floor,
                                     final / np.maximum(peak, 1e-6), np.inf)
                if not np.isfinite(ratio).any():
                    log.info(f"kick {strategy} {beh} a={a:g}: no clean example, skipped")
                    continue
                b = int(ok[int(np.argmin(ratio))])
                left = render_trail_video(sc.get(style.n_ghosts + 1, W, H), q0[b],
                                          col_base, style, rng)
                right = render_trail_video(sc.get(style.n_ghosts + 1, W, H), qa[b],
                                           col_kick, style, rng)
                frames = _tile([left, right], 2)
                rel = f"kick/{strategy}_{beh.lower()}_a{a:g}"
                write_mp4(frames, MEDIA / f"{rel}.mp4", FPS, crf=25)
                write_poster(frames[260 // STEP], MEDIA / f"{rel}.jpg")
                item["amps"][f"{a:g}"] = {
                    "src": f"media/{rel}.mp4", "poster": f"media/{rel}.jpg", "body": b,
                    "recovered": round(float(1 - final[ok == b][0] / peak[ok == b][0]), 3)
                    if np.isfinite(ratio[ok == b][0]) else None,
                }
                log.info(f"kick {strategy} {beh} code {c} a={a:g} body {b} "
                         f"gap {peak[ok == b][0]:.2f}->{final[ok == b][0]:.2f}")
            items.append(item)
        out[strategy] = items
    _write_manifest("kick", out)


# ---------------------------------------------------------------- pairs
def _phase_clarity(q: np.ndarray, base: str) -> float:
    """How clearly a settled window shows its behavior (higher is clearer)."""
    v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).mean() * 100.0
    if base == "Rear":
        return float(q[:, 2].mean())
    if base == "Walk":
        return float(v)
    if base == "Turn":
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return float(abs(yaw[-1] - yaw[0]))
    return float(-v)  # Immobile: stillness


def pick_pair_body(qpos: np.ndarray, labels: dict | None, base_a: str, base_b: str,
                   settle: float = 0.4) -> int:
    """Body that executes both phases most clearly.

    Candidates are bodies the labeler reads as ``A`` then ``B`` with no
    unrequested bridge behavior; among them, the best summed rank of
    per-phase clarity (rear height, walk speed, turn yaw, immobile stillness)
    over each phase's settled window. Falls back to Fig 13's body.
    """
    n, T = qpos.shape[:2]
    if not labels:
        return n - 1
    ok = set(_finite_bodies(qpos).tolist())
    cands = [b["idx"] for b in labels["bodies"]
             if b["phase_a"] == base_a and b["phase_b"] == base_b and not b["bridge"]
             and b["idx"] in ok]
    if not cands:
        return n - 1
    half = T // 2
    w = int(half * settle)
    ca = np.array([_phase_clarity(qpos[i, half - w:half], base_a) for i in cands])
    cb = np.array([_phase_clarity(qpos[i, T - w:], base_b) for i in cands])
    rank = ca.argsort().argsort() + cb.argsort().argsort()
    return int(cands[int(np.argmax(rank))])


def render_pairs(sc: _Scenes, run: str = "mlb_s208_len500") -> None:
    """Every unobserved MLB pair, hold A then B (Fig 13).

    The body shown is chosen by :func:`pick_pair_body`.
    """
    style = TrailStyle(n_ghosts=5, ghost_every=30, cam_distance=0.9)
    W, H = 960, 640
    rdir = ROOT / "outputs/behavior/impossible_transitions" / run
    out = []
    for f in sorted(rdir.glob("rollout_*.npz")):
        d = np.load(f)
        a, b = int(d["code_a"]), int(d["code_b"])
        lab = rdir / f.name.replace("rollout_", "labels_").replace(".npz", ".json")
        readout = json.load(open(lab)) if lab.exists() else None
        body = pick_pair_body(d["qpos"], readout, MLB_SHORT[a].split("_")[0],
                              MLB_SHORT[b].split("_")[0])
        q = d["qpos"][body]
        T = q.shape[0]
        half = T // 2
        col = np.array([mlb_rgb(a)] * half + [mlb_rgb(b)] * (T - half))
        frames = render_trail_video(sc.get(style.n_ghosts + 1, W, H), q, col, style,
                                    range(0, T, STEP))
        rel = f"pairs/{MLB_SHORT[a]}__{MLB_SHORT[b]}"
        write_mp4(frames, MEDIA / f"{rel}.mp4", FPS, crf=25)
        write_poster(frames[-1], MEDIA / f"{rel}.jpg")
        out.append({"a": a, "b": b, "a_name": MLB_SHORT[a], "b_name": MLB_SHORT[b],
                    "switch": half / 100.0, "duration": len(frames) / FPS,
                    "src": f"media/{rel}.mp4", "poster": f"media/{rel}.jpg",
                    "readout": readout, "body": body})
        log.info(f"pair {MLB_SHORT[a]} -> {MLB_SHORT[b]} body {body}")
    _write_manifest("pairs", out)


# -------------------------------------------------------------- streams
TEAL = (30 / 255, 128 / 255, 146 / 255)


def render_streams(sc: _Scenes, clips: tuple[int, ...] = (0, 1, 2)) -> None:
    """The same controller under original, GPT and random code streams."""
    style = TrailStyle(n_ghosts=5, ghost_every=30, cam_distance=0.9)
    W, H = 640, 480
    kdir = ROOT / "outputs/fidelity/moseq_inception_distance_mlb"
    methods = [("decoder_original_codes", "Original codes"), ("gpt2", "GPT prior"),
               ("uniform_random", "Uniform random")]
    out = []
    for ci in clips:
        panels = []
        for m, _ in methods:
            q = np.load(kdir / f"rollouts_{m}.npz")[f"raw_{ci}"]
            panels.append(render_trail_video(
                sc.get(style.n_ghosts + 1, W, H), q, np.tile(TEAL, (len(q), 1)), style,
                range(0, len(q), STEP)))
        n = min(len(p) for p in panels)
        frames = _tile([p[:n] for p in panels], 3)
        rel = f"streams/clip{ci}"
        write_mp4(frames, MEDIA / f"{rel}.mp4", FPS, crf=25)
        write_poster(frames[n // 2], MEDIA / f"{rel}.jpg")
        out.append({"clip": ci, "labels": [l for _, l in methods],
                    "src": f"media/{rel}.mp4", "poster": f"media/{rel}.jpg"})
        log.info(f"streams clip {ci}")
    _write_manifest("streams", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", nargs="+",
                    choices=["atlas", "roundtrip", "kick", "pairs", "streams"])
    args = ap.parse_args()
    sc = _Scenes()
    for w in args.what:
        {"atlas": render_atlas, "roundtrip": render_roundtrip, "kick": render_kick,
         "pairs": render_pairs, "streams": render_streams}[w](sc)


if __name__ == "__main__":
    main()
