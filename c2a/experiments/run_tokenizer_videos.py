"""Tokenizer-diagnostics video stage.

Consumes ``tokenizer_diagnostics_summary.json`` (produced by
``run_tokenizer_diagnostics``) — specifically the ``best_per_family``
selection — and renders MP4s into
``outputs/tokenizer_diagnostics/video/``:

1. **Syllable-grid videos** — one per family. Lays out the most-used
   syllables as a grid of mini-renders, all advancing in sync (shorter
   bouts looped to match the longest). Each cell shows one syllable's
   representative bout with a code-color badge.

2. **Segmentation-overlay videos** — five reference clips, each shown
   as: rodent render on top, four time-aligned code bars below (one
   per family). Per-frame "current code" chips next to each bar make
   the alignment between body pose and code easy to read.

Reads:
* ``outputs/tokenizer_diagnostics/tokenizer_diagnostics_summary.json``
* ``data/rodent/rodent_reference_clips.h5`` (raw qpos)
* For each family-winner, the codes file named in
  ``best_per_family.<family>.source_info``.

Run::

    python -m c2a.experiments.run_tokenizer_videos
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Sequence

# Ensure EGL is used for off-screen MuJoCo rendering on headless GPU.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import h5py
import hydra
import imageio
import mujoco
import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from c2a.experiments.shared.kpms_results_loader import load_kpms_syllables
from c2a.rendering_utils import _get_font, get_code_colormap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("tokenizer_videos")


FAMILY_TITLES = {
    "kpms": "KPMS", "vae": "VAE", "vqvae": "VQ-VAE", "ps_vqvae": "PS-VQ-VAE",
}


# =============================================================================
# Code loading
# =============================================================================


def _load_winning_codes(source_info: dict) -> np.ndarray:
    """Load ``(n_clips, n_frames)`` codes for one family's winning config."""
    if "results_h5" in source_info:
        return load_kpms_syllables(source_info["results_h5"])
    if "best_codes_npz" in source_info:
        data = np.load(source_info["best_codes_npz"], allow_pickle=False)
        if "all_codes" in data:
            return np.asarray(data["all_codes"], dtype=np.int32)
        train = np.asarray(data["train_codes"], dtype=np.int32)
        test = np.asarray(data["test_codes"], dtype=np.int32)
        return np.concatenate([train, test], axis=0)
    raise ValueError(
        f"source_info does not contain a recognized codes pointer: "
        f"keys={sorted(source_info.keys())}"
    )


def _load_ref_qpos(reference_h5: str, n_frames_per_clip: int) -> np.ndarray:
    with h5py.File(reference_h5, "r") as f:
        flat = np.asarray(f["qpos"][:], dtype=np.float32)
    n_total = flat.shape[0] // n_frames_per_clip
    return flat[: n_total * n_frames_per_clip].reshape(
        n_total, n_frames_per_clip, -1,
    )


# =============================================================================
# Bouts → representative-bout picker
# =============================================================================


def _bouts_per_syllable(
    codes: np.ndarray, K: int,
) -> dict[int, list[tuple[int, int, int]]]:
    out: dict[int, list[tuple[int, int, int]]] = {c: [] for c in range(K)}
    n_clips, n_frames = codes.shape
    for ci in range(n_clips):
        clip = codes[ci]
        change = np.empty(clip.shape, dtype=bool)
        change[0] = True
        change[1:] = clip[1:] != clip[:-1]
        starts = np.where(change)[0]
        ends = np.append(starts[1:], n_frames)
        for s, e in zip(starts, ends):
            v = int(clip[s])
            if v < 0 or v >= K:
                continue
            out[v].append((ci, int(s), int(e)))
    return out


def _pick_representative_bout(
    bouts: list[tuple[int, int, int]],
    target_duration_frames: int,
    *,
    min_frames: int = 5,
    max_frames: int = 60,
) -> tuple[int, int, int] | None:
    if not bouts:
        return None
    best = min(
        bouts, key=lambda b: abs((b[2] - b[1]) - target_duration_frames),
    )
    ci, s, e = best
    length = e - s
    if length < min_frames:
        return None
    if length > max_frames:
        center = (s + e) // 2
        s = max(0, center - max_frames // 2)
        e = s + max_frames
    return (ci, s, e)


# =============================================================================
# MuJoCo renderer (Nature-styled, light background)
# =============================================================================


class _Renderer:
    def __init__(
        self,
        xml_path: str,
        height: int = 480,
        width: int = 640,
        distance: float = 0.55,
        elevation: float = -22.0,
        azimuth: float = 90.0,
    ) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.height = int(height)
        self.width = int(width)
        self.renderer = mujoco.Renderer(
            self.mj_model, height=self.height, width=self.width,
        )
        self._free_camera = mujoco.MjvCamera()
        self._free_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._free_camera.distance = float(distance)
        self._free_camera.elevation = float(elevation)
        self._free_camera.azimuth = float(azimuth)

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        if qpos.shape[0] != self.mj_model.nq:
            raise ValueError(
                f"qpos has shape {qpos.shape}, expected ({self.mj_model.nq},)"
            )
        self.mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        lookat = np.asarray(self.mj_data.qpos[:3], dtype=np.float64)
        self._free_camera.lookat = lookat
        self.renderer.update_scene(self.mj_data, camera=self._free_camera)
        return self.renderer.render()


# =============================================================================
# Drawing helpers
# =============================================================================


def _ensure_even(n: int) -> int:
    """Round up to nearest even integer (libx264 requires even dims)."""
    return n if n % 2 == 0 else n + 1


def _pad_to_codec(frame: np.ndarray) -> np.ndarray:
    """Pad H/W to multiples of 16 for libx264 compatibility (avoids imageio warnings)."""
    h, w = frame.shape[:2]
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    if new_h == h and new_w == w:
        return frame
    out = np.full((new_h, new_w, *frame.shape[2:]), 255, dtype=frame.dtype)
    out[:h, :w] = frame
    return out


def _draw_banner(
    width: int,
    text: str,
    *,
    height: int = 36,
    font_size: int = 18,
    bg=(245, 245, 250),
    fg=(40, 40, 60),
    weight="bold",
) -> np.ndarray:
    """Light banner with bold dark text — for plot-like title strips."""
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size, bold=(weight == "bold"))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = max(10, (width - tw) // 2)
    ty = max(2, (height - th) // 2 - 1)
    draw.text((tx, ty), text, font=font, fill=fg)
    return np.array(img)


def _draw_code_chip(
    code_idx: int,
    color: tuple[int, int, int],
    *,
    width: int = 120,
    height: int = 36,
    label: str | None = None,
    font_size: int = 16,
) -> np.ndarray:
    """Small colored chip with a code id (and optional row label)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Color swatch on the left.
    swatch_w = 28
    draw.rectangle([(2, 2), (swatch_w, height - 2)], fill=color, outline=(60, 60, 60))
    # Code id text.
    font = _get_font(font_size, bold=True)
    code_text = f"{code_idx}"
    bbox = draw.textbbox((0, 0), code_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((swatch_w + 6, (height - th) // 2 - 1), code_text,
              font=font, fill=(20, 20, 20))
    if label is not None:
        small_font = _get_font(max(font_size - 4, 9), bold=False)
        sb = draw.textbbox((0, 0), label, font=small_font)
        sx = swatch_w + 6 + tw + 8
        if sx + (sb[2] - sb[0]) <= width - 4:
            draw.text((sx, (height - (sb[3] - sb[1])) // 2 - 1),
                      label, font=small_font, fill=(80, 80, 80))
    return np.array(img)


def _draw_bar_strip(
    width: int,
    codes: np.ndarray,
    current_frame_idx: int,
    code_colors: np.ndarray,
    *,
    height: int = 38,
    playhead_width: int = 3,
) -> np.ndarray:
    """A single timeline bar with playhead. White background, no border."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    T = codes.shape[0]
    for j in range(T):
        c = int(codes[j])
        if c < 0:
            continue
        x0 = int(j * width / T)
        x1 = int((j + 1) * width / T)
        if x1 > x0:
            img[2:height - 2, x0:x1] = code_colors[c % len(code_colors)]
    # Playhead.
    ph = int(current_frame_idx * width / T)
    ph = min(ph, width - playhead_width)
    img[:, ph:ph + playhead_width] = (255, 255, 255)
    img[:, ph - 1:ph] = (30, 30, 30)
    img[:, ph + playhead_width:ph + playhead_width + 1] = (30, 30, 30)
    return img


# =============================================================================
# 1) Syllable-grid video (NxM grid layout)
# =============================================================================


def _grid_dims(n: int) -> tuple[int, int]:
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    return rows, cols


def render_syllable_grid_video(
    family: str,
    K: int,
    config_label: str,
    codes: np.ndarray,
    ref_qpos: np.ndarray,
    out_path: Path,
    *,
    xml_path: str,
    fps: int = 50,
    target_dur_frames: int = 20,
    min_bouts: int = 3,
    max_panels: int = 24,
    cell_size: tuple[int, int] = (200, 160),
    cell_pad: int = 4,
) -> None:
    """Lay out all syllables (with enough bouts) as a grid; all cells
    advance simultaneously, with shorter bouts looped to match the
    longest. The result is one video per family showing all syllables
    side-by-side, not concatenated.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bouts = _bouts_per_syllable(codes, K)
    # Order syllables by total frame frequency (most-used first), then
    # take the top `max_panels` syllables that have ≥ `min_bouts`.
    freq = np.bincount(codes[codes >= 0].ravel().astype(np.int64),
                       minlength=K)[:K]
    cand = [
        c for c in np.argsort(-freq).tolist()
        if len(bouts[c]) >= min_bouts
    ][:max_panels]
    if not cand:
        log.warning("[%s] no qualifying syllables; skipping grid video", family)
        return

    # Pick representative bout per chosen syllable.
    picks: list[tuple[int, tuple[int, int, int]]] = []  # (syllable, bout)
    for c in cand:
        pick = _pick_representative_bout(
            bouts[c], target_duration_frames=target_dur_frames,
        )
        if pick is not None:
            picks.append((c, pick))
    if not picks:
        log.warning("[%s] no representative bouts; skipping grid video", family)
        return

    log.info(
        "[%s] %s — K=%d, grid of %d syllables (max %d), rendering → %s",
        family, config_label, K, len(picks), max_panels, out_path,
    )

    # Render each bout's frames at cell size; pad/loop to common length L.
    cell_w, cell_h = cell_size
    code_colors = get_code_colormap(max(K, len(picks)))
    renderer = _Renderer(xml_path=xml_path, height=cell_h, width=cell_w,
                         distance=0.5, elevation=-22.0, azimuth=90.0)
    per_cell_frames: list[list[np.ndarray]] = []
    per_cell_meta: list[tuple[int, int, int]] = []  # (c, n_frames, n_bouts)
    L = 0
    for c, (ci, s, e) in picks:
        frames = []
        for fi in range(s, e):
            rgb = renderer.render_qpos(ref_qpos[ci, fi])
            # Light overlay: code chip in top-left.
            chip = _draw_code_chip(
                code_idx=c, color=tuple(int(x) for x in code_colors[c % len(code_colors)]),
                width=58, height=22, font_size=14,
            )
            out = rgb.copy()
            out[2:2 + chip.shape[0], 2:2 + chip.shape[1]] = chip
            frames.append(out)
        per_cell_frames.append(frames)
        per_cell_meta.append((c, e - s, len(bouts[c])))
        L = max(L, len(frames))

    # Loop shorter cells to match L.
    for k, frames in enumerate(per_cell_frames):
        if len(frames) < L:
            reps = (L + len(frames) - 1) // len(frames)
            per_cell_frames[k] = (frames * reps)[:L]

    # Grid layout.
    rows, cols = _grid_dims(len(picks))
    pad = cell_pad
    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = rows * cell_h + (rows + 1) * pad
    bg = (250, 250, 252)
    banner = _draw_banner(
        grid_w,
        f"{FAMILY_TITLES.get(family, family)} — {config_label}  ·  "
        f"K={K}  ·  showing {len(picks)} syllables (top by usage)",
        height=36,
    )

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8)
    try:
        for fi in range(L):
            canvas = np.full((grid_h, grid_w, 3), bg, dtype=np.uint8)
            for k, frames in enumerate(per_cell_frames):
                r, c = k // cols, k % cols
                y0 = pad + r * (cell_h + pad)
                x0 = pad + c * (cell_w + pad)
                canvas[y0:y0 + cell_h, x0:x0 + cell_w] = frames[fi]
            full = np.concatenate([banner, canvas], axis=0)
            writer.append_data(_pad_to_codec(full))
    finally:
        writer.close()
    log.info("[%s] wrote grid video (%dx%d cells, %d frames) → %s",
             family, rows, cols, L, out_path)


# =============================================================================
# 2) Segmentation-overlay video (rodent + 4 stacked bars + per-row chips)
# =============================================================================


def render_segmentation_overlay_video(
    clip_idx: int,
    ref_qpos_clip: np.ndarray,
    family_codes: dict[str, np.ndarray],
    family_K: dict[str, int],
    family_order: Sequence[str],
    family_labels: Sequence[str],
    out_path: Path,
    *,
    xml_path: str,
    fps: int = 50,
    rodent_size: tuple[int, int] = (720, 540),
    bar_height: int = 38,
    bar_label_width: int = 220,
    chip_width: int = 110,
) -> None:
    """Cleaner segmentation overlay:

    Top:  rodent render (large, light background)
    Mid:  banner with clip + frame
    Bottom: stacked bars — for each family, a row of:
            [family + config label] [current-code chip] [timeline bar with playhead]

    All four rows share an x-axis aligned to the same reference motion.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = ref_qpos_clip.shape[0]
    width = rodent_size[0]
    bar_total_w = bar_label_width + chip_width + width  # full row width

    # Build a renderer sized for the top panel.
    renderer = _Renderer(
        xml_path=xml_path,
        height=rodent_size[1],
        width=rodent_size[0],
        distance=0.55, elevation=-22.0, azimuth=90.0,
    )
    max_K = max(family_K[f] for f in family_order)
    code_colors = get_code_colormap(max_K)

    log.info(
        "Rendering segmentation overlay for clip %d, %d frames → %s",
        clip_idx, n_frames, out_path,
    )

    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8)
    try:
        bg = (250, 250, 252)
        for fi in range(n_frames):
            rgb = renderer.render_qpos(ref_qpos_clip[fi])
            # Pad rodent to match bar_total_w (centered).
            rod_pad_left = (bar_total_w - rgb.shape[1]) // 2
            rod_pad_right = bar_total_w - rgb.shape[1] - rod_pad_left
            rod_padded = np.concatenate([
                np.full((rgb.shape[0], rod_pad_left, 3), bg, dtype=np.uint8),
                rgb,
                np.full((rgb.shape[0], rod_pad_right, 3), bg, dtype=np.uint8),
            ], axis=1)

            banner = _draw_banner(
                bar_total_w,
                f"clip {clip_idx}  ·  frame {fi:>3d} / {n_frames - 1}",
                height=32, font_size=16,
            )

            # Build the four bar rows.
            row_imgs = []
            for f, lbl in zip(family_order, family_labels):
                fam_codes = family_codes[f][clip_idx]
                c_now = int(fam_codes[fi]) if fam_codes[fi] >= 0 else 0
                color = tuple(int(x) for x in code_colors[c_now % len(code_colors)])

                # Left: family label (white panel with right-aligned text).
                lbl_img = Image.new("RGB", (bar_label_width, bar_height), (255, 255, 255))
                ldraw = ImageDraw.Draw(lbl_img)
                lfont = _get_font(13, bold=True)
                bbox = ldraw.textbbox((0, 0), lbl, font=lfont)
                lt_y = (bar_height - (bbox[3] - bbox[1])) // 2 - 1
                ldraw.text((8, lt_y), lbl, font=lfont, fill=(30, 30, 30))
                # Middle: current-code chip.
                chip = _draw_code_chip(
                    code_idx=c_now, color=color,
                    width=chip_width, height=bar_height, font_size=15,
                )
                # Right: timeline bar with playhead.
                bar = _draw_bar_strip(
                    width=width, codes=fam_codes, current_frame_idx=fi,
                    code_colors=code_colors, height=bar_height,
                )
                row = np.concatenate([np.array(lbl_img), chip, bar], axis=1)
                row_imgs.append(row)

            # Thin separators between bar rows.
            sep_h = 2
            sep = np.full((sep_h, bar_total_w, 3), (220, 220, 225), dtype=np.uint8)
            stacked_rows = []
            for i, row in enumerate(row_imgs):
                stacked_rows.append(row)
                if i < len(row_imgs) - 1:
                    stacked_rows.append(sep)
            bars_block = np.concatenate(stacked_rows, axis=0)

            composite = np.concatenate([rod_padded, banner, bars_block], axis=0)
            writer.append_data(_pad_to_codec(composite))
    finally:
        writer.close()


# =============================================================================
# Main
# =============================================================================


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="tokenizer_videos",
)
def main(cfg: DictConfig) -> None:
    out_dir = Path(str(cfg.output.base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    summary_path = Path(str(cfg.input.summary_json))
    with open(summary_path) as f:
        summary = json.load(f)
    best_per_family = summary.get("best_per_family", {})
    if not best_per_family:
        raise RuntimeError(
            f"No best_per_family in {summary_path}; re-run "
            f"run_tokenizer_diagnostics first."
        )
    log.info("Best per family: %s", {
        k: v["config"] for k, v in best_per_family.items()
    })

    n_frames = int(cfg.reference.n_frames_per_clip)
    fps = int(cfg.reference.fps)
    ref_qpos = _load_ref_qpos(str(cfg.reference.h5_path), n_frames)
    log.info("Reference qpos: %s", ref_qpos.shape)

    family_codes: dict[str, np.ndarray] = {}
    family_K: dict[str, int] = {}
    family_label: dict[str, str] = {}
    for family in ("kpms", "vae", "vqvae", "ps_vqvae"):
        entry = best_per_family.get(family)
        if entry is None:
            continue
        codes = _load_winning_codes(entry["source_info"])
        n_use = min(codes.shape[1], n_frames)
        codes = codes[:, :n_use]
        family_codes[family] = codes
        family_K[family] = int(entry["K"])
        family_label[family] = entry["config"]
        log.info("[%s] codes shape=%s K=%d", family, codes.shape, family_K[family])

    xml_path = str(cfg.renderer.xml_path)

    # --- 1) Syllable-grid videos (one per family) ---
    if bool(cfg.outputs.syllable_grid):
        for family, codes in family_codes.items():
            render_syllable_grid_video(
                family=family,
                K=family_K[family],
                config_label=family_label[family],
                codes=codes,
                ref_qpos=ref_qpos,
                out_path=out_dir / f"syllables_{family}.mp4",
                xml_path=xml_path,
                fps=fps,
                target_dur_frames=int(cfg.outputs.target_dur_frames),
                min_bouts=int(cfg.outputs.min_bouts_per_syllable),
                max_panels=int(cfg.outputs.get("grid_max_panels", 24)),
                cell_size=(
                    int(cfg.outputs.get("grid_cell_w", 200)),
                    int(cfg.outputs.get("grid_cell_h", 160)),
                ),
            )

    # --- 2) Segmentation-overlay videos ---
    if bool(cfg.outputs.segmentation_overlay):
        clip_picks = list(cfg.outputs.segmentation_clip_ids)
        # The codes arrays cover only the balanced split (≈484 clips of
        # the 842 total). Clamp to whichever is smaller across the
        # families we'll plot so every selected clip has codes everywhere.
        max_clip = min(
            ref_qpos.shape[0],
            *[family_codes[f].shape[0] for f in family_codes],
        )
        if not clip_picks:
            rng = np.random.RandomState(int(cfg.outputs.segmentation_seed))
            clip_picks = sorted(
                rng.choice(max_clip, size=5, replace=False).tolist()
            )
        else:
            clip_picks = [int(c) for c in clip_picks if 0 <= int(c) < max_clip]
            if not clip_picks:
                log.warning(
                    "No valid clip ids in segmentation_clip_ids; "
                    "falling back to random pick.",
                )
                rng = np.random.RandomState(int(cfg.outputs.segmentation_seed))
                clip_picks = sorted(
                    rng.choice(max_clip, size=5, replace=False).tolist()
                )
        family_order = [f for f in ("kpms", "vae", "vqvae", "ps_vqvae")
                        if f in family_codes]
        family_labels = [
            f"{FAMILY_TITLES.get(f, f)} ({family_label[f]})"
            for f in family_order
        ]
        for clip_idx in clip_picks:
            render_segmentation_overlay_video(
                clip_idx=int(clip_idx),
                ref_qpos_clip=ref_qpos[clip_idx],
                family_codes=family_codes,
                family_K=family_K,
                family_order=family_order,
                family_labels=family_labels,
                out_path=out_dir / f"segmentation_clip{clip_idx:04d}.mp4",
                xml_path=xml_path,
                fps=fps,
                rodent_size=(
                    int(cfg.outputs.get("seg_rodent_w", 720)),
                    int(cfg.outputs.get("seg_rodent_h", 540)),
                ),
                bar_height=int(cfg.outputs.get("bar_height", 38)),
                bar_label_width=int(cfg.outputs.get("bar_label_width", 220)),
                chip_width=int(cfg.outputs.get("chip_width", 110)),
            )

    log.info("=" * 60)
    log.info("Done — videos in %s", out_dir)


if __name__ == "__main__":
    main()
