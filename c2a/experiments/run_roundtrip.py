"""Code round-trip reconstruction experiment.

Does a discrete code sequence survive the round trip
``code -> generated motion -> re-segmentation -> code``?

Two stages (stage 3 — unconditional generation — is a future extension):

Stage 1 — segmenter self-consistency
    Segment a few reference clips several times with each of the four
    segmenters (kpms, vae, vqvae, ps_vqvae). The three autoencoders are
    deterministic functions and should score ~100%. KPMS applies via Gibbs
    sampling, so it is run under two conditions: ``same_seed`` (pinned PRNG
    key — pure determinism) and ``diff_seed`` (the honest Gibbs-variance
    number).

Stage 2 — generate-from-real-code round trip
    For each (decoder checkpoint, segmenter) pair: take real reference
    clips, their precomputed code sequence ``c``; roll out the c2a decoder
    decoder-only (``z_e_scale=0``) to generate motion; re-segment the
    generated motion to ``ĉ``; compare ``ĉ`` vs ``c``.

Metrics (per-frame multiclass): accuracy, precision, recall, F1
(macro / micro / weighted). Stage 2 additionally reports a
**transition-masked** variant — frames within ``transient_window`` of a
code change are dropped, since the GRU needs a few frames to express a new
code — which is the headline steady-state fidelity.

Usage::

    python -m c2a.experiments.run_roundtrip                 # stages 1 + 2
    python -m c2a.experiments.run_roundtrip stages=[1]      # stage 1 only
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
import logging
from pathlib import Path

import h5py
import hydra
import jax

# orbax (used by load_moseq_checkpoint) calls jax.monitoring.record_scalar,
# which was removed in newer JAX. Same stub as run_code_grid_rollouts.py.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from c2a.experiments.shared.clip_selection import (
    load_balanced_splits,
    select_clips_by_behavior,
)
from c2a.experiments.shared.plotting import NATURE_COLORS, set_nature_style
from c2a.experiments.shared.segmenters import load_segmenter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_roundtrip")

_AVGS = ("macro", "micro", "weighted")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-frame multiclass metrics over a flattened (clips x frames) pair.

    Frames where either side is ``-1`` (KPMS did not assign a syllable) are
    dropped. Precision / recall / F1 average over the labels actually
    present (sklearn default) so an unused codebook slot does not drag the
    macro score down.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    valid = (yt >= 0) & (yp >= 0)
    yt, yp = yt[valid], yp[valid]
    if yt.size == 0:
        out = {"accuracy": 0.0, "n_frames": 0}
        for avg in _AVGS:
            out[f"precision_{avg}"] = 0.0
            out[f"recall_{avg}"] = 0.0
            out[f"f1_{avg}"] = 0.0
        return out
    out = {"accuracy": float(accuracy_score(yt, yp)), "n_frames": int(yt.size)}
    for avg in _AVGS:
        p, r, f, _ = precision_recall_fscore_support(
            yt, yp, average=avg, zero_division=0,
        )
        out[f"precision_{avg}"] = float(p)
        out[f"recall_{avg}"] = float(r)
        out[f"f1_{avg}"] = float(f)
    return out


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, num_codes: int) -> np.ndarray:
    """Fixed-size ``(num_codes, num_codes)`` confusion matrix (valid frames)."""
    from sklearn.metrics import confusion_matrix

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    valid = (yt >= 0) & (yp >= 0) & (yt < num_codes) & (yp < num_codes)
    if valid.sum() == 0:
        return np.zeros((num_codes, num_codes), dtype=np.int64)
    return confusion_matrix(
        yt[valid], yp[valid], labels=list(range(num_codes)),
    ).astype(np.int64)


def _mean_std_dicts(dicts: list[dict]) -> dict:
    """Mean (and std for scalars) of a list of flat metric dicts."""
    if not dicts:
        return {}
    keys = [k for k in dicts[0] if k != "n_frames"]
    out: dict = {"n_samples": len(dicts)}
    for k in keys:
        vals = np.asarray([d[k] for d in dicts], dtype=np.float64)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    out["n_frames_total"] = int(sum(d.get("n_frames", 0) for d in dicts))
    return out


def _transition_keep_mask(codes: np.ndarray, transient_window: int) -> np.ndarray:
    """Boolean per-frame mask; ``False`` for frames within ``transient_window``
    frames *after* a code change (the GRU settling transient)."""
    codes = np.asarray(codes)
    keep = np.ones(codes.shape[0], dtype=bool)
    for t in np.where(np.diff(codes) != 0)[0] + 1:
        keep[t : t + transient_window] = False
    return keep


def _dist_since_transition(codes: np.ndarray) -> np.ndarray:
    """Frames elapsed since the last code change (0 at a change frame)."""
    codes = np.asarray(codes)
    d = np.zeros(codes.shape[0], dtype=np.int64)
    for i in range(1, codes.shape[0]):
        d[i] = 0 if codes[i] != codes[i - 1] else d[i - 1] + 1
    return d


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    im = ax.imshow(cm / row, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("reconstructed code ĉ")
    ax.set_ylabel("input code c")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(ĉ | c)")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_bars(labels: list[str], values: list[float], title: str,
               ylabel: str, path: Path) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 1.5, 2.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, 0.6, color=NATURE_COLORS["blue"], edgecolor="none")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_code_examples(c: np.ndarray, c_hat: np.ndarray, num_codes: int,
                        n_examples: int, title: str, path: Path) -> None:
    """Per-clip side-by-side code-sequence strips: ``c`` on the left, every
    rollout repeat of ``ĉ`` on the right. Colors are shared across panels
    (same code id → same color)."""
    from matplotlib.colors import ListedColormap

    from c2a.experiments.shared.plotting import get_code_colormap

    set_nature_style()
    n_repeats = c_hat.shape[1]
    n_examples = int(min(n_examples, c.shape[0]))
    rgb = get_code_colormap(num_codes) / 255.0
    cmap = ListedColormap(rgb)

    fig, axes = plt.subplots(
        n_examples, 1 + n_repeats,
        figsize=(2.4 * (1 + n_repeats), 0.55 * n_examples + 0.9),
        squeeze=False,
    )
    headers = ["input c"] + [f"ĉ rep {r}" for r in range(n_repeats)]
    for ci in range(n_examples):
        strips = [c[ci]] + [c_hat[ci, r] for r in range(n_repeats)]
        for col, arr in enumerate(strips):
            ax = axes[ci, col]
            a = np.asarray(arr).reshape(1, -1).astype(float)
            a[a < 0] = np.nan
            ax.imshow(
                a, aspect="auto", cmap=cmap,
                vmin=-0.5, vmax=num_codes - 0.5, interpolation="nearest",
            )
            ax.set_yticks([])
            ax.set_xticks([])
            if ci == 0:
                ax.set_title(headers[col], fontsize=7)
            if col == 0:
                ax.set_ylabel(
                    f"clip {ci}", fontsize=6, rotation=0,
                    labelpad=14, va="center",
                )
    fig.suptitle(title, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


def _plot_accuracy_vs_transition(dist: np.ndarray, correct: np.ndarray,
                                 title: str, path: Path,
                                 max_dist: int = 30) -> None:
    set_nature_style()
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    xs, ys = [], []
    for d in range(max_dist + 1):
        m = dist == d
        if m.sum() > 0:
            xs.append(d)
            ys.append(float(correct[m].mean()))
    ax.plot(xs, ys, "o-", color=NATURE_COLORS["blue"])
    ax.set_xlabel("frames since last code change")
    ax.set_ylabel("per-frame round-trip accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Clip selection / reference qpos
# ---------------------------------------------------------------------------


def _select_clips(cfg: DictConfig) -> tuple[list[int], list[int]]:
    """Pick the round-trip clip set from the test split.

    Returns:
        ``(local_positions, global_ids)`` — ``local_positions`` index the
        test split (and the decoders' ``test_codes`` / ``test_clips``);
        ``global_ids`` index the full reference H5.
    """
    splits = load_balanced_splits(cfg.paths.balanced_split)
    test_global = list(splits["balanced"]["test_indices"])
    sel = select_clips_by_behavior(
        splits,
        split="test",
        k_per_behavior=int(cfg.clips.k_per_behavior),
        seed=int(cfg.seed),
        behaviors=tuple(cfg.clips.behaviors),
    )
    local = sorted({p for ps in sel.values() for p in ps})
    global_ids = [int(test_global[p]) for p in local]
    return local, global_ids


def _load_ref_qpos(reference_h5: str, global_ids: list[int],
                   clip_length: int) -> np.ndarray:
    """Load ``(n_clips, clip_length, nq)`` reference qpos for ``global_ids``."""
    with h5py.File(reference_h5, "r") as f:
        qpos_flat = f["qpos"][:]
    n_total = qpos_flat.shape[0] // clip_length
    clipped = qpos_flat[: n_total * clip_length].reshape(
        n_total, clip_length, -1,
    )
    return np.stack([clipped[i] for i in global_ids], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 1 — segmenter self-consistency
# ---------------------------------------------------------------------------


def _run_stage1(cfg: DictConfig, output_dir: Path, segmenters: dict,
                ref_qpos: np.ndarray, global_ids: list[int]) -> dict:
    log.info("\n%s\n  STAGE 1 — segmenter self-consistency\n%s",
             "=" * 60, "=" * 60)
    n_repeats = int(cfg.stage1.n_repeats)
    summary: dict = {}

    for name, seg in segmenters.items():
        seg_dir = output_dir / "stage1" / name
        seg_dir.mkdir(parents=True, exist_ok=True)
        is_kpms = str(cfg.segmenters[name].type) == "kpms"
        log.info("\n  --- %s (%s) ---", name, "kpms" if is_kpms else "ae")

        if is_kpms:
            conditions = {
                "same_seed": seg.encode_repeats(ref_qpos, [0] * n_repeats),
                "diff_seed": seg.encode_repeats(
                    ref_qpos, list(range(n_repeats))),
            }
        else:
            conditions = {
                "deterministic": seg.encode_repeats(
                    ref_qpos, list(range(n_repeats))),
            }

        seg_summary: dict = {}
        save: dict = {"clip_ids": np.asarray(global_ids, dtype=np.int64)}
        for cond, repeat_codes in conditions.items():
            save[f"codes_{cond}"] = repeat_codes
            ref = repeat_codes[0]
            vs0 = [_clf_metrics(ref, repeat_codes[r])
                   for r in range(1, repeat_codes.shape[0])]
            pairwise = []
            cm = np.zeros((seg.num_codes, seg.num_codes), dtype=np.int64)
            for a in range(repeat_codes.shape[0]):
                for b in range(a + 1, repeat_codes.shape[0]):
                    pairwise.append(_clf_metrics(repeat_codes[a], repeat_codes[b]))
                    cm += _confusion(repeat_codes[a], repeat_codes[b],
                                     seg.num_codes)
            seg_summary[cond] = {
                "vs_repeat0": _mean_std_dicts(vs0),
                "pairwise": _mean_std_dicts(pairwise),
                "n_repeats": int(repeat_codes.shape[0]),
            }
            _plot_confusion(
                cm, f"{name} stage-1 {cond}",
                seg_dir / f"confusion_{cond}.png",
            )
            acc = seg_summary[cond]["pairwise"].get("accuracy", 0.0)
            log.info("    %-14s pairwise accuracy=%.4f  f1_macro=%.4f",
                     cond, acc,
                     seg_summary[cond]["pairwise"].get("f1_macro", 0.0))

        np.savez_compressed(seg_dir / "codes.npz", **save)
        (seg_dir / "metrics.json").write_text(json.dumps(seg_summary, indent=2))
        summary[name] = seg_summary

    # Headline bar chart: pairwise accuracy per (segmenter, condition).
    # AE segmenters contribute one bar (deterministic); KPMS contributes two
    # (same_seed and diff_seed) so the Gibbs-variance gap is visible.
    labels, values = [], []
    for name, seg_summary in summary.items():
        for cond, metrics in seg_summary.items():
            labels.append(f"{name}\n({cond})")
            values.append(metrics["pairwise"].get("accuracy", 0.0))
    _plot_bars(labels, values, "Stage 1 — segmenter self-consistency",
               "pairwise per-frame accuracy", output_dir / "stage1_accuracy.png")
    return summary


# ---------------------------------------------------------------------------
# Stage 2 — generate-from-real-code round trip
# ---------------------------------------------------------------------------


def _build_decoder_env(ckpt_cfg, reference_h5: str, test_indices: np.ndarray,
                       test_codes: np.ndarray):
    """Build the ``MoSeqImitation`` env for a decoder checkpoint."""
    from track_mjx.config import utils
    from vnl_playground.tasks.rodent.imitation import ReferenceClips

    from c2a.moseq_env_wrapper import MoSeqImitation

    clip_len = int(ckpt_cfg.env_config.clip_length)
    clips = ReferenceClips(
        data_path=reference_h5,
        n_frames_per_clip=clip_len,
        keep_clips_idx=np.asarray(test_indices, dtype=np.int64),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(
        config=env_cfg,
        clips=clips,
        kpms_codes=test_codes,
        code_stack_size=code_stack_size,
    )
    return env, clip_len


def _run_stage2(cfg: DictConfig, output_dir: Path, segmenters: dict,
                local_positions: list[int], global_ids: list[int]) -> dict:
    log.info("\n%s\n  STAGE 2 — generate-from-real-code round trip\n%s",
             "=" * 60, "=" * 60)
    from c2a.experiments.shared.checkpoint_utils import (
        load_moseq_checkpoint,
        make_inference_fn,
        run_rollout,
    )

    splits = load_balanced_splits(cfg.paths.balanced_split)
    split_test_idx = np.asarray(splits["balanced"]["test_indices"], dtype=np.int64)
    transient_window = int(cfg.stage2.transient_window)
    n_rollout_repeats = int(cfg.stage2.n_rollout_repeats)
    z_e_scale = float(cfg.stage2.z_e_scale)
    summary: dict = {}

    for pair in cfg.stage2.pairs:
        dec_name = str(pair["decoder"])
        seg_name = str(pair["segmenter"])
        tag = f"{dec_name}__{seg_name}"
        pair_dir = output_dir / "stage2" / tag
        pair_dir.mkdir(parents=True, exist_ok=True)
        log.info("\n  --- pair %s ---", tag)

        seg = segmenters[seg_name]
        # `c` = the codes the decoder was conditioned on (segmenter best_codes).
        codes_npz = np.load(str(cfg.segmenters[seg_name].codes_npz),
                            allow_pickle=True)
        test_codes = np.asarray(codes_npz["test_codes"], dtype=np.int32)
        codes_test_idx = np.asarray(codes_npz["test_indices"], dtype=np.int64)
        if not np.array_equal(codes_test_idx, split_test_idx):
            raise RuntimeError(
                f"{seg_name} codes_npz test_indices differ from the "
                f"balanced split — the env would inject codes for the "
                f"wrong clips. Refusing to produce meaningless metrics."
            )

        ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(
            str(cfg.decoders[dec_name]),
        )
        if not bool(ckpt_cfg.network_config.get("use_rnn_decoder", False)):
            log.warning("    decoder %s is not RNN — skipping pair.", dec_name)
            continue
        params = (norm_state, policy_params)
        inf_fn = make_inference_fn(
            ppo_networks, use_rnn=True, deterministic=True, z_e_scale=z_e_scale,
        )
        env, clip_len = _build_decoder_env(
            ckpt_cfg, cfg.paths.reference_h5, split_test_idx, test_codes,
        )
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        # `c` for the selected clips.
        c = np.stack([test_codes[p][:clip_len] for p in local_positions], axis=0)
        n_clips = c.shape[0]

        # Roll out: n_clips x n_rollout_repeats decoder-only rollouts.
        gen_list: list[np.ndarray] = []
        key = jax.random.PRNGKey(int(cfg.seed))
        for ci, pos in enumerate(local_positions):
            for rep in range(n_rollout_repeats):
                key, sub = jax.random.split(key)
                result = run_rollout(
                    env, inf_fn, params, ppo_networks, use_rnn=True, key=sub,
                    max_steps=clip_len, reset_clip_idx=int(pos),
                    jit_reset=jit_reset, jit_step=jit_step, ignore_done=True,
                )
                q = np.asarray(result["qpos"], dtype=np.float32)[:clip_len]
                gen_list.append(q)
            log.info("    clip %d/%d (test pos %d) rolled out x%d",
                     ci + 1, n_clips, pos, n_rollout_repeats)
        gen_qpos = np.stack(gen_list, axis=0).reshape(
            n_clips, n_rollout_repeats, clip_len, -1,
        )
        log.info("    generated qpos: shape %s", gen_qpos.shape)

        # Re-segment all (clip, repeat) generated trajectories at once.
        flat = gen_qpos.reshape(n_clips * n_rollout_repeats, clip_len, -1)
        c_hat_flat = seg.encode(flat, seed=int(cfg.seed))
        ch_len = min(c_hat_flat.shape[1], clip_len)
        c_hat = c_hat_flat[:, :ch_len].reshape(
            n_clips, n_rollout_repeats, ch_len,
        )
        c = c[:, :ch_len]

        # Metrics: per (clip, repeat), all-frame + transition-masked. The
        # masked variant drops (a) frames within transient_window of a code
        # change (GRU settling) and (b) the segmenter's window-padding
        # boundary frames (replicated edge codes, not independently observed).
        all_m, masked_m = [], []
        cm = np.zeros((seg.num_codes, seg.num_codes), dtype=np.int64)
        dist_pool, correct_pool = [], []
        bnd_lo, bnd_hi = seg.boundary_pad
        core = np.zeros(c.shape[1], dtype=bool)
        core[bnd_lo : c.shape[1] - bnd_hi] = True
        for ci in range(n_clips):
            keep = _transition_keep_mask(c[ci], transient_window) & core
            dist = _dist_since_transition(c[ci])
            for rep in range(n_rollout_repeats):
                ch = c_hat[ci, rep]
                all_m.append(_clf_metrics(c[ci], ch))
                masked_m.append(_clf_metrics(c[ci][keep], ch[keep]))
                cm += _confusion(c[ci], ch, seg.num_codes)
                valid = (c[ci] >= 0) & (ch >= 0)
                dist_pool.append(dist[valid])
                correct_pool.append((c[ci] == ch)[valid])

        pair_summary = {
            "all_frames": _mean_std_dicts(all_m),
            "transition_masked": _mean_std_dicts(masked_m),
            "n_clips": n_clips,
            "n_rollout_repeats": n_rollout_repeats,
            "transient_window": transient_window,
        }
        (pair_dir / "metrics.json").write_text(json.dumps(pair_summary, indent=2))
        np.savez_compressed(
            pair_dir / "roundtrip.npz",
            c=c, c_hat=c_hat,
            clip_ids=np.asarray(global_ids, dtype=np.int64),
            local_positions=np.asarray(local_positions, dtype=np.int64),
            confusion=cm,
            **({"gen_qpos": gen_qpos} if bool(cfg.stage2.get("save_qpos", False))
               else {}),
        )
        _plot_confusion(cm, f"{tag} round-trip", pair_dir / "confusion.png")
        _plot_code_examples(
            c, c_hat, seg.num_codes,
            n_examples=min(6, n_clips),
            title=f"{tag} — input code c vs reconstructed ĉ",
            path=pair_dir / "code_examples.png",
        )
        _plot_accuracy_vs_transition(
            np.concatenate(dist_pool), np.concatenate(correct_pool),
            f"{tag} round-trip", pair_dir / "accuracy_vs_transition.png",
        )
        summary[tag] = pair_summary
        log.info(
            "    round-trip accuracy: all=%.4f  transition-masked=%.4f",
            pair_summary["all_frames"].get("accuracy", 0.0),
            pair_summary["transition_masked"].get("accuracy", 0.0),
        )

    if summary:
        labels = list(summary.keys())
        values = [summary[t]["transition_masked"].get("accuracy", 0.0)
                  for t in labels]
        _plot_bars(labels, values,
                   "Stage 2 — round-trip fidelity (transition-masked)",
                   "per-frame accuracy", output_dir / "stage2_accuracy.png")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="roundtrip")
def main(cfg: DictConfig) -> None:
    if jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "jax_enable_x64 is True in the driver process — KPMS x64 leaked. "
            "KPMS must run only inside the subprocess worker."
        )

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stages = [int(s) for s in cfg.stages]
    log.info("=== Round-trip experiment — stages %s ===", stages)

    # Clip set (shared by both stages).
    local_positions, global_ids = _select_clips(cfg)
    clip_length = int(cfg.clip_length)
    log.info("Selected %d clips (global ids %s)", len(global_ids), global_ids)

    # Build all segmenters once.
    segmenters: dict = {}
    for name, seg_cfg in cfg.segmenters.items():
        log.info("Loading segmenter %s ...", name)
        segmenters[name] = load_segmenter(
            name, seg_cfg,
            fk_h5=cfg.paths.reference_h5,
            stac_xml=cfg.paths.stac_xml,
            tmp_dir=tmp_dir,
        )

    summary: dict = {"stages": stages, "clip_ids": global_ids}

    if 1 in stages:
        ref_qpos = _load_ref_qpos(
            str(cfg.paths.reference_h5), global_ids, clip_length,
        )
        summary["stage1"] = _run_stage1(
            cfg, output_dir, segmenters, ref_qpos, global_ids,
        )

    if 2 in stages:
        summary["stage2"] = _run_stage2(
            cfg, output_dir, segmenters, local_positions, global_ids,
        )

    (output_dir / "roundtrip_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log.info("\n=== Done. Outputs under %s ===", output_dir)


if __name__ == "__main__":
    main()
