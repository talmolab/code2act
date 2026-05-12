"""shMoSeq hyperparameter grid search (paper-faithful selection).

Sweeps ``trans_kappa × n_states × seed`` over a hierarchical-HMM fit on
top of an existing KPMS code sequence (typically
``c2a/outputs/kpms_sweep/best_codes.npz``).  Selection follows
**Weinreb, Kannan, …, Linderman 2026 (Neuron)** —
*"Spontaneous behavior is a succession of self-directed tasks"*:

1. **Lagged-mutual-information analysis** on the input KPMS sequence
   to derive a target-duration range for the high-level shMoSeq
   states.  We compare the real-data lagged MI against a Markov-1
   surrogate; the lag at which real_MI rises above shuffle_MI but
   approaches markov_MI defines the natural timescale of hierarchical
   structure.
2. **trans_kappa** selection: filter (κ × n_states) cells whose median
   state duration falls inside the MI-derived window; among those,
   pick the cell with the highest log-probability peak (highest
   marginal log-likelihood).
3. **n_states** selection: from the surviving (n_states, trans_kappa)
   combos, pick the one with the highest **average pairwise adjusted-
   Rand-score across seeds** — i.e., the model that most reliably
   reproduces the same state partition under different random
   initializations.

This module is invoked indirectly via ``c2a.codegen.run_sweep`` (the
dispatcher) when the YAML has ``model: shimoseq``.

**CRITICAL**: this module imports ``state_moseq`` which sets
``jax_enable_x64 = True`` process-globally.  Run in a separate process
from RL training.

Usage::

    python -m c2a.codegen.run_sweep --config c2a/configs/shimoseq_sweep.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# state_moseq sets jax_enable_x64 on import; mirror that policy here so
# anything we touch before the lazy import is consistent.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from c2a.paths import get_c2a_root, get_repo_root
from c2a.codegen.shimoseq.config import ShimoseqHyperparams, hyperparams_to_dict
from c2a.codegen.shimoseq.syllable_loader import batch_for_shimoseq, load_kpms_codes
from c2a.codegen.shimoseq.fit_shimoseq import fit_shimoseq, ShimoseqFitResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Lagged-MI: derive the target high-level-state duration from the input
# =============================================================================


def _lagged_mi_target_window(
    seqs: np.ndarray,
    mask: np.ndarray,
    lags: np.ndarray,
    duration_band_frac: tuple[float, float] = (0.5, 1.5),
) -> tuple[float, dict]:
    """Run ``state_moseq.lagged_mutual_information`` and derive a
    target-duration *window* in lag-units.

    The shMoSeq paper picks the high-level-state duration to land on
    the timescale at which real lagged-MI deviates meaningfully from a
    Markov-1 surrogate (i.e., the lag where structure beyond a flat
    HMM begins to appear).  Concretely we look for the lag where
    ``real_MI / max(real_MI - shuff_MI)`` peaks — this is the natural
    timescale of hierarchical structure.

    To allow some slack, we return a window
    ``[peak * band_frac[0], peak * band_frac[1]]``.

    Args:
        seqs, mask: from :func:`c2a.codegen.shimoseq.syllable_loader.batch_for_shimoseq`.
        lags: array of lag values (in input-frame units) at which to evaluate MI.
        duration_band_frac: ±fraction window around the peak lag.

    Returns:
        ``(peak_lag, info)`` where ``info`` is a diagnostic dict with the
        full MI curves and the chosen window edges.
    """
    import state_moseq as sm
    import jax.numpy as jnp

    seqs_j = jnp.asarray(seqs, dtype=jnp.int32)
    mask_j = jnp.asarray(mask, dtype=jnp.bool_)
    # Returns [n_sequences, n_lags]; average across sequences for the curve.
    real_mi, markov_mi, shuff_mi = sm.lagged_mutual_information(seqs_j, mask_j, lags)
    real_mi = np.asarray(real_mi).mean(axis=0)
    markov_mi = np.asarray(markov_mi).mean(axis=0)
    shuff_mi = np.asarray(shuff_mi).mean(axis=0)

    # "Hierarchical excess" = real - markov (positive means structure beyond
    # a flat HMM at that lag).  Skip lag=1 because that captures only
    # first-order autocorrelation — already accounted for by the input
    # syllable's own dwell-time and not where shMoSeq's hierarchy lives.
    # We pick the lag at which excess peaks among lags ≥ 2.
    excess = np.maximum(real_mi - markov_mi, 0.0)
    lags_arr = np.asarray(lags)
    valid = lags_arr >= 2
    excess_for_peak = np.where(valid, excess, -np.inf)
    if not np.any(valid) or np.max(np.where(valid, excess, 0.0)) <= 1e-9:
        log.warning(
            "Lagged-MI excess (real - markov) is nonpositive at lags >= 2 — "
            "the input sequence may not have hierarchical structure beyond "
            "a Markov-1 model.  Falling back to argmax of real_mi at lags >= 2."
        )
        idx_peak = int(np.argmax(np.where(valid, real_mi, -np.inf)))
    else:
        idx_peak = int(np.argmax(excess_for_peak))

    peak_lag = float(lags[idx_peak])
    lo, hi = peak_lag * duration_band_frac[0], peak_lag * duration_band_frac[1]
    info = {
        "lags": np.asarray(lags),
        "real_mi": real_mi,
        "markov_mi": markov_mi,
        "shuff_mi": shuff_mi,
        "peak_lag": peak_lag,
        "window_lo": lo,
        "window_hi": hi,
    }
    log.info(
        f"Lagged-MI: peak lag = {peak_lag:.2f} frames "
        f"(real_MI={real_mi[idx_peak]:.3f}, markov_MI={markov_mi[idx_peak]:.3f}); "
        f"target duration window = [{lo:.2f}, {hi:.2f}] frames"
    )
    return peak_lag, info


def _plot_lagged_mi(
    info: dict,
    output_path: Path,
) -> None:
    """Per-sweep diagnostic: real/markov/shuffle MI vs lag, with the
    derived window highlighted."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(info["lags"], info["real_mi"], "-", color="#222", lw=1.5,
            label="real")
    ax.plot(info["lags"], info["markov_mi"], "-", color="#888", lw=1.0,
            label="Markov-1 surrogate")
    ax.plot(info["lags"], info["shuff_mi"], "--", color="#bbb", lw=1.0,
            label="shuffle null")
    ax.axvspan(info["window_lo"], info["window_hi"], alpha=0.2, color="#3498db",
               label="target window")
    ax.axvline(info["peak_lag"], color="#e74c3c", lw=1, alpha=0.7,
               label=f"peak = {info['peak_lag']:.1f}")
    ax.set_xlabel("Lag (input-frame units)")
    ax.set_ylabel("Lagged mutual information (bits)")
    ax.set_title("shMoSeq target-duration derivation")
    ax.legend(fontsize=7, frameon=False)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Selection (Weinreb 2026 recipe)
# =============================================================================


def _select_best_shimoseq(
    valid_results: list[dict],
    duration_window: tuple[float, float],
    seqs_shape: tuple[int, int],
) -> tuple[dict, list[dict], dict]:
    """Pick the best shMoSeq fit per the Weinreb 2026 recipe.

    Three-stage selection:

      1. ``trans_kappa`` filter: keep only fits whose median state
         duration is inside ``duration_window``.  If empty, fall back
         to closest-to-window-center.
      2. Within survivors: rank by marginal log-likelihood (highest =
         the "log-prob peak" the paper recommends).
      3. ``n_states`` selection: among the top fits, group by
         (n_states, trans_kappa) and compute the average pairwise
         adjusted-Rand-score across seeds.  Pick the (n_states,
         trans_kappa) combo with the highest mean Rand score —
         the most reproducible partition across random inits.

    Args:
        valid_results: list of per-fit metric dicts containing
            ``mean_duration``, ``marginal_loglik``, ``n_states``,
            ``trans_kappa``, ``seed``, and ``per_frame_states_flat``
            (a flattened 1D state sequence used for the Rand-score
            cross-seed comparison).
        duration_window: ``(lo, hi)`` from the lagged-MI step.
        seqs_shape: shape ``(B, L)`` of the batched input — used to
            pre-compute the total number of frames for diagnostics only.

    Returns:
        ``(best_fit_dict, survivors_in_window, info)`` where ``info``
        has the per-cell Rand scores for the diagnostic plot.
    """
    lo, hi = duration_window

    in_window = [r for r in valid_results
                 if lo <= float(r.get("mean_duration", 0)) <= hi]
    if in_window:
        survivors = list(in_window)
    else:
        center = 0.5 * (lo + hi)
        survivors = sorted(
            valid_results,
            key=lambda r: abs(float(r.get("mean_duration", 0)) - center),
        )[:8]
        log.warning(
            f"No fit landed in MI-derived duration window [{lo:.2f}, {hi:.2f}]; "
            f"falling back to 8 fits closest to window center."
        )

    # Group by (n_states, trans_kappa) for the Rand-score consistency step.
    by_cell: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for r in survivors:
        key = (int(r["n_states"]), float(r["trans_kappa"]))
        by_cell[key].append(r)

    # Compute mean pairwise adjusted-Rand-score within each cell (across seeds).
    cell_rand: dict[tuple[int, float], float] = {}
    cell_loglik: dict[tuple[int, float], float] = {}
    for cell, fits in by_cell.items():
        if len(fits) < 2:
            cell_rand[cell] = float("nan")
        else:
            scores = []
            for i in range(len(fits)):
                for j in range(i + 1, len(fits)):
                    a = fits[i]["per_frame_states_flat"]
                    b = fits[j]["per_frame_states_flat"]
                    n = min(len(a), len(b))
                    if n < 2:
                        continue
                    scores.append(adjusted_rand_score(a[:n], b[:n]))
            cell_rand[cell] = float(np.mean(scores)) if scores else float("nan")
        cell_loglik[cell] = float(np.mean(
            [r.get("marginal_loglik", float("-inf")) for r in fits]
        ))

    # Pick best cell: maximize Rand (paper's n_states criterion); ties
    # broken by mean log-likelihood (paper's trans_kappa within-window
    # criterion).
    def _cell_score(cell: tuple[int, float]) -> tuple[float, float]:
        rand = cell_rand[cell]
        if np.isnan(rand):
            rand = -np.inf
        return (rand, cell_loglik[cell])
    best_cell = max(by_cell.keys(), key=_cell_score)

    # Within best cell, pick the seed with the highest marginal log-lik.
    best_fits_in_cell = sorted(
        by_cell[best_cell],
        key=lambda r: -r.get("marginal_loglik", float("-inf")),
    )
    best_fit = best_fits_in_cell[0]

    info = {
        "duration_window": (lo, hi),
        "cell_rand": cell_rand,
        "cell_loglik": cell_loglik,
        "best_cell": best_cell,
        "n_seqs": int(seqs_shape[0]),
        "seg_length": int(seqs_shape[1]),
    }
    return best_fit, survivors, info


# =============================================================================
# Per-fit visualizations
# =============================================================================


def _plot_state_timeline(
    states_dict: dict[str, np.ndarray],
    n_states: int,
    output_path: Path,
    max_clips: int = 30,
) -> None:
    """Stack the discovered state sequences as a heatmap (clips × time)."""
    clip_ids = sorted(states_dict.keys())[:max_clips]
    if not clip_ids:
        return
    arr = np.stack(
        [states_dict[c][: min(len(states_dict[c]), 250)] for c in clip_ids]
    )
    # Right-pad shorter rows with -1 to render uniformly.
    T = max(s.size for s in [arr[i] for i in range(arr.shape[0])])
    fig, ax = plt.subplots(figsize=(7, max(2.0, 0.16 * len(clip_ids))))
    cmap = plt.get_cmap("tab10")
    im = ax.imshow(arr, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(n_states, 10))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Clip")
    ax.set_yticks(range(len(clip_ids)))
    ax.set_yticklabels(clip_ids, fontsize=5)
    ax.set_title(f"shMoSeq state timeline — first {len(clip_ids)} clips")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Discovered state")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_kpms_composition(
    kpms_dict: dict[str, np.ndarray],
    states_dict: dict[str, np.ndarray],
    n_states: int,
    n_kpms_codes: int,
    output_path: Path,
) -> None:
    """For each shMoSeq state, histogram of which KPMS syllables compose it.

    A clean shMoSeq state should be dominated by a small number of KPMS
    syllables (the state aggregates them).  A "junk" state will look
    flat across the KPMS vocabulary.
    """
    H = np.zeros((n_states, n_kpms_codes), dtype=np.int64)
    for cid, st_seq in states_dict.items():
        if cid not in kpms_dict:
            continue
        kp = kpms_dict[cid][: len(st_seq)]
        st = st_seq[: len(kp)]
        for s in range(n_states):
            np.add.at(H[s], kp[st == s].astype(np.int64), 1)

    row_sums = H.sum(axis=1, keepdims=True)
    norm = H / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(max(4.0, 0.12 * n_kpms_codes), 0.4 * n_states + 1.5))
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("KPMS syllable id")
    ax.set_ylabel("shMoSeq state")
    ax.set_title("Per-state KPMS-syllable composition")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("P(KPMS code | state)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_duration_histogram(
    durations: np.ndarray,
    target_window: tuple[float, float],
    output_path: Path,
) -> None:
    """Histogram of state durations with the MI-derived window overlay."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    if durations.size:
        bins = np.logspace(np.log10(max(1, durations.min())),
                           np.log10(max(2, durations.max() + 1)), 40)
        ax.hist(durations, bins=bins, color="#3498db", alpha=0.8,
                edgecolor="white")
    ax.axvspan(target_window[0], target_window[1], alpha=0.15, color="#e74c3c",
               label="MI window")
    ax.set_xscale("log")
    ax.set_xlabel("State duration (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Discovered shMoSeq state durations")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_sweep_summary(
    info: dict,
    output_dir: Path,
) -> None:
    """Sweep-level: Rand-score and log-likelihood as a function of (n_states, trans_kappa)."""
    cell_rand = info["cell_rand"]
    cell_loglik = info["cell_loglik"]
    if not cell_rand:
        return
    n_states_vals = sorted(set(c[0] for c in cell_rand.keys()))
    kappa_vals = sorted(set(c[1] for c in cell_rand.keys()))
    rand_grid = np.full((len(n_states_vals), len(kappa_vals)), np.nan)
    ll_grid = np.full((len(n_states_vals), len(kappa_vals)), np.nan)
    for (n, k), v in cell_rand.items():
        i = n_states_vals.index(n); j = kappa_vals.index(k)
        rand_grid[i, j] = v
    for (n, k), v in cell_loglik.items():
        i = n_states_vals.index(n); j = kappa_vals.index(k)
        ll_grid[i, j] = v

    fig, axes = plt.subplots(1, 2, figsize=(8, 0.5 * len(n_states_vals) + 2.5))
    for ax, grid, title, cbar_label in zip(
        axes,
        (rand_grid, ll_grid),
        ("Avg pairwise Rand score (consistency)", "Mean marginal log-lik"),
        ("Rand", "log-lik"),
    ):
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v < np.nanmean(grid) else "black")
        ax.set_xticks(range(len(kappa_vals)))
        ax.set_xticklabels([f"{k:.0e}" for k in kappa_vals], fontsize=7, rotation=45)
        ax.set_yticks(range(len(n_states_vals)))
        ax.set_yticklabels([str(n) for n in n_states_vals], fontsize=7)
        ax.set_xlabel("trans_kappa")
        ax.set_ylabel("n_states")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)

    # Highlight best cell
    best = info["best_cell"]
    bi = n_states_vals.index(best[0]); bj = kappa_vals.index(best[1])
    for ax in axes:
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                    edgecolor="#e74c3c", lw=2))
    fig.suptitle("shMoSeq sweep — selection summary "
                 f"(best cell: n_states={best[0]}, trans_kappa={best[1]:.1e})")
    fig.tight_layout()
    fig.savefig(output_dir / "sweep_summary.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Main sweep orchestration
# =============================================================================


def run_sweep(cfg: dict) -> dict:
    """Run a shMoSeq hyperparameter grid search.

    The dispatcher (:mod:`c2a.codegen.run_sweep`) hands us a parsed,
    env-resolved config dict with at least:

    * ``data.codes_path`` — path to KPMS ``best_codes.npz``
    * ``sweep.n_states`` — list of candidate state counts (paper used 2–10)
    * ``sweep.trans_kappa`` — log-spaced list of stickiness values
    * ``sweep.seeds_per_setting`` — how many random restarts per cell
    * ``hyperparams.{emission_gd_lr, num_iters, seg_length}``
    * ``selection.duration_band_frac`` — ±fraction around lagged-MI peak
    * ``output.base_dir`` — output directory
    * (optional) ``selection.lags`` — lag grid for the lagged-MI analysis;
      defaults to log-spaced 1–256.

    Returns:
        dict with ``best_model``, ``all_results``, ``mi_info``,
        ``selection_info``.
    """
    sweep_cfg = cfg["sweep"]
    output_dir = Path(cfg["output"]["base_dir"])
    if not output_dir.is_absolute():
        output_dir = get_repo_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + batch input KPMS codes ----
    codes_path = cfg["data"]["codes_path"]
    if not Path(codes_path).is_absolute():
        codes_path = str(get_repo_root() / codes_path)
    syllables_dict, provenance = load_kpms_codes(codes_path)
    n_kpms_codes = int(max(arr.max() for arr in syllables_dict.values())) + 1
    log.info(f"Input KPMS codebook size: {n_kpms_codes}")

    seg_length = int(cfg["hyperparams"]["seg_length"])
    seqs, mask, metadata = batch_for_shimoseq(syllables_dict, seg_length=seg_length)
    log.info(f"Batched: seqs={seqs.shape}, mask={mask.shape}")

    # ---- Lagged-MI: derive target duration window ----
    sel_cfg = cfg.get("selection", {}) or {}
    lags_cfg = sel_cfg.get("lags", None)
    if lags_cfg is None:
        lags = np.unique(np.round(np.logspace(0, np.log10(seg_length // 2), 24)).astype(int))
    else:
        lags = np.asarray(lags_cfg, dtype=int)
    duration_band_frac = tuple(sel_cfg.get("duration_band_frac", [0.5, 1.5]))
    peak_lag, mi_info = _lagged_mi_target_window(
        seqs, mask, lags, duration_band_frac=duration_band_frac,
    )
    _plot_lagged_mi(mi_info, output_dir / "lagged_mi.png")

    # ---- Sweep grid ----
    grid = list(itertools.product(
        sweep_cfg["n_states"],
        sweep_cfg["trans_kappa"],
    ))
    n_seeds = int(sweep_cfg["seeds_per_setting"])
    log.info(
        f"Grid: {len(grid)} (n_states × trans_kappa) × {n_seeds} seeds = "
        f"{len(grid) * n_seeds} fits"
    )

    all_results: list[dict] = []
    fit_objects: dict[tuple[int, float, int], ShimoseqFitResult] = {}

    base_hp = ShimoseqHyperparams(
        emission_gd_lr=float(cfg["hyperparams"].get("emission_gd_lr", 1e-2)),
        emission_gd_iters=int(cfg["hyperparams"].get("emission_gd_iters", 1000)),
        num_iters=int(cfg["hyperparams"].get("num_iters", 100)),
        seg_length=seg_length,
    )

    for gi, (n_states, trans_kappa) in enumerate(grid):
        for seed in range(n_seeds):
            hp = ShimoseqHyperparams(
                n_states=int(n_states),
                trans_kappa=float(trans_kappa),
                emission_gd_lr=base_hp.emission_gd_lr,
                emission_gd_iters=base_hp.emission_gd_iters,
                num_iters=base_hp.num_iters,
                seg_length=base_hp.seg_length,
            )
            log.info(
                f"[{gi + 1}/{len(grid)}] n_states={n_states}, "
                f"trans_kappa={trans_kappa:.1e}, seed={seed}"
            )
            try:
                fit = fit_shimoseq(
                    seqs, mask, metadata, hp, seed=seed, verbose=False,
                )
            except Exception as e:
                log.warning(f"  FAILED: {e}")
                all_results.append({
                    "n_states": int(n_states),
                    "trans_kappa": float(trans_kappa),
                    "seed": seed,
                    "error": str(e),
                })
                continue

            mean_dur = float(fit.durations.mean()) if fit.durations.size else 0.0
            median_dur = float(np.median(fit.durations)) if fit.durations.size else 0.0
            # Flatten per-clip state sequences for the Rand-score step.
            per_frame_states_flat = np.concatenate(
                [fit.states_dict[k] for k in sorted(fit.states_dict.keys())]
            ).astype(np.int32)
            result = {
                "n_states": int(n_states),
                "trans_kappa": float(trans_kappa),
                "seed": seed,
                "marginal_loglik": fit.marginal_loglik,
                "mean_duration": mean_dur,
                "median_duration": median_dur,
                "active_states": fit.active_states,
                "usage_ratio": fit.active_states / int(n_states),
                "per_frame_states_flat": per_frame_states_flat,
            }
            all_results.append(result)
            fit_objects[(int(n_states), float(trans_kappa), seed)] = fit
            log.info(
                f"  marginal_ll={fit.marginal_loglik:.2f}, "
                f"median_dur={median_dur:.2f}, active_states={fit.active_states}/{n_states}"
            )

    # ---- Selection ----
    valid = [r for r in all_results if "error" not in r]
    if not valid:
        raise RuntimeError("No successful shMoSeq fits in sweep.")

    best, survivors, sel_info = _select_best_shimoseq(
        valid,
        duration_window=(mi_info["window_lo"], mi_info["window_hi"]),
        seqs_shape=seqs.shape,
    )
    log.info(
        f"\nBest shMoSeq cell: n_states={best['n_states']}, "
        f"trans_kappa={best['trans_kappa']:.1e}, seed={best['seed']}; "
        f"median_dur={best['median_duration']:.2f}, "
        f"marginal_ll={best['marginal_loglik']:.2f}"
    )

    # ---- Per-fit + sweep-summary plots for the best cell ----
    best_fit_obj = fit_objects.get(
        (best["n_states"], best["trans_kappa"], best["seed"])
    )
    if best_fit_obj is not None:
        best_dir = output_dir / (
            f"best_n{best['n_states']}_k{best['trans_kappa']:.0e}_seed{best['seed']}"
        )
        best_dir.mkdir(parents=True, exist_ok=True)
        _plot_state_timeline(
            best_fit_obj.states_dict, best["n_states"],
            best_dir / "state_timeline.png",
        )
        _plot_kpms_composition(
            syllables_dict, best_fit_obj.states_dict,
            best["n_states"], n_kpms_codes,
            best_dir / "kpms_composition.png",
        )
        _plot_duration_histogram(
            best_fit_obj.durations,
            (mi_info["window_lo"], mi_info["window_hi"]),
            best_dir / "duration_hist.png",
        )

    _plot_sweep_summary(sel_info, output_dir)

    # ---- Save ----
    # Strip the per_frame_states_flat array from the JSON-saved version
    # (it can be large); keep a separate npz.
    json_results = []
    for r in all_results:
        rj = {k: v for k, v in r.items() if k != "per_frame_states_flat"}
        json_results.append(rj)
    summary = {
        "best_model": {k: v for k, v in best.items() if k != "per_frame_states_flat"},
        "all_results": json_results,
        "mi_info": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in mi_info.items()},
        "selection_info": {
            "duration_window": list(sel_info["duration_window"]),
            "cell_rand": {f"n{n}_k{k:.0e}": v
                          for (n, k), v in sel_info["cell_rand"].items()},
            "cell_loglik": {f"n{n}_k{k:.0e}": v
                            for (n, k), v in sel_info["cell_loglik"].items()},
            "best_cell": list(sel_info["best_cell"]),
        },
        "provenance": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in provenance.items()},
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = output_dir / "sweep_results.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved sweep results: {summary_path}")

    # Save per-frame state arrays for the best fit (for downstream codegen).
    if best_fit_obj is not None:
        np.savez_compressed(
            output_dir / "best_states.npz",
            **{k: v for k, v in best_fit_obj.states_dict.items()},
            n_states=np.int32(best["n_states"]),
            trans_kappa=np.float64(best["trans_kappa"]),
            seed=np.int32(best["seed"]),
            mean_duration=np.float64(best["mean_duration"]),
            median_duration=np.float64(best["median_duration"]),
        )
        log.info(f"Saved best-fit states: {output_dir / 'best_states.npz'}")

    # Save per-clip state arrays for the top-K fits (used by
    # c2a.codegen.shimoseq.render_topk_bands to draw the K-row band plots/videos).
    # Keys are namespaced as ``m{rank}__{clip_key}`` (1-indexed rank), with
    # a parallel ``meta_m{rank}`` JSON-encoded string carrying that fit's
    # selection metadata.
    topk_for_export = sorted(
        valid, key=lambda r: -float(r.get("marginal_loglik", float("-inf"))),
    )[:max(5, 1)]
    arrays_to_save: dict[str, np.ndarray] = {}
    for rank_idx, r in enumerate(topk_for_export, start=1):
        key = (int(r["n_states"]), float(r["trans_kappa"]), int(r["seed"]))
        fit = fit_objects.get(key)
        if fit is None:
            continue
        for clip_key, st in fit.states_dict.items():
            arrays_to_save[f"m{rank_idx}__{clip_key}"] = np.asarray(st, dtype=np.int32)
        meta = {
            "rank": rank_idx,
            "n_states": int(r["n_states"]),
            "trans_kappa": float(r["trans_kappa"]),
            "seed": int(r["seed"]),
            "marginal_loglik": float(r["marginal_loglik"]),
            "median_duration": float(r["median_duration"]),
            "mean_duration": float(r["mean_duration"]),
            "active_states": int(r["active_states"]),
        }
        arrays_to_save[f"meta_m{rank_idx}"] = np.array([json.dumps(meta)])
    if arrays_to_save:
        topk_path = output_dir / "topk_states.npz"
        np.savez_compressed(topk_path, **arrays_to_save)
        log.info(
            f"Saved top-{len(topk_for_export)} fit states: {topk_path} "
            f"({sum(1 for k in arrays_to_save if k.startswith('m') and '__' in k)} "
            f"per-clip arrays)"
        )

    return summary


def main():
    """Direct CLI entry point. Prefer the dispatcher
    ``python -m c2a.codegen.run_sweep`` which routes here when the YAML
    has ``model: shimoseq``."""
    parser = argparse.ArgumentParser(description="shMoSeq hyperparameter sweep")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    if args.config is None:
        args.config = str(get_c2a_root() / "configs" / "shimoseq_sweep.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    run_sweep(cfg)


if __name__ == "__main__":
    main()
