"""Lightweight wandb helpers for AR training.

The main decoder training reuses ``track_mjx.agent.wandb_logging``, but
that module is built around the PPO training loop's
``progress_fn(num_steps, metrics)`` callback. AR training has a simpler
shape (epoch loop over supervised cross-entropy), so we use the
underlying ``wandb.log`` directly and just standardize metric grouping.

Group prefixes match the conventions in
``c2a/train_moseq_decoder.py:_grouped_wandb_progress``: ``loss/*``,
``train/*``, ``sample/*``. ``em/*`` is added here for the HMM EM loop.
``histograms/*`` is the bar-chart group.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def log_metrics(
    wandb_run: Any,
    step: int,
    metrics: dict[str, float],
    group: str | None = None,
) -> None:
    """Log a flat dict of metrics under an optional group prefix.

    Args:
        wandb_run: An active wandb Run, or ``None`` (no-op).
        step: Global step (epoch * steps_per_epoch + step_in_epoch, or
            EM iter number for HMM).
        metrics: ``{name: scalar}``.
        group: Optional prefix; if set, keys become ``"{group}/{name}"``.
    """
    if wandb_run is None:
        return
    if group is not None:
        metrics = {f"{group}/{k}": v for k, v in metrics.items()}
    metrics["step"] = step
    try:
        wandb_run.log(metrics, step=step)
    except Exception as e:
        # Never let a wandb hiccup kill training.
        log.warning(f"wandb.log failed (step={step}): {e}")


def log_code_histograms_figure(
    wandb_run: Any,
    step: int,
    hists: dict[str, np.ndarray],
    num_codes: int,
    title: str,
    panel: str = "histograms",
) -> None:
    """Log a grouped bar chart of code histograms.

    Args:
        wandb_run: Active wandb Run (or None — no-op).
        step: Wandb step.
        hists: Mapping ``{series_name: counts_array_of_length_num_codes}``.
            Each counts array is normalized to a frequency (sums to 1)
            internally so absolute counts can differ across series.
        num_codes: Vocabulary size.
        title: Both the figure title and the wandb key under ``panel/``.
        panel: WandB section the chart lands under. Default
            ``"histograms"`` keeps it in its own panel cleanly separated
            from ``loss/`` and ``train/``.
    """
    if wandb_run is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        x = np.arange(num_codes)
        n_series = max(1, len(hists))
        width = 0.8 / n_series
        fig, ax = plt.subplots(figsize=(max(6.0, num_codes * 0.4), 3.5))
        for i, (name, h) in enumerate(hists.items()):
            h = np.asarray(h, dtype=np.float64)
            total = h.sum()
            freq = h / total if total > 0 else h
            offset = (i - (n_series - 1) / 2.0) * width
            ax.bar(x + offset, freq, width=width, label=name)
        ax.set_xticks(x)
        ax.set_xlabel("Code id")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        wandb_run.log({f"{panel}/{title}": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        log.warning(f"log_code_histograms_figure failed (step={step}, title={title}): {e}")


def code_histogram(codes: np.ndarray, num_codes: int) -> np.ndarray:
    """Bincount over a flat array of codes, length ``num_codes``."""
    flat = np.asarray(codes, dtype=np.int64).ravel()
    return np.bincount(flat, minlength=num_codes)[:num_codes].astype(np.int64)


def predicted_histogram_from_batches(
    predict_fn,
    params,
    tokens: np.ndarray,
    batch_size: int,
    num_codes: int,
    bos_id: int = 0,
) -> np.ndarray:
    """Run teacher-forced argmax over ``tokens`` and histogram results.

    Args:
        predict_fn: ``(params, batch) -> jnp.int32[B, T]`` — model's
            JIT'd argmax-prediction function. Predictions are in the
            +1-shifted token space (``[0, vocab_size)``).
        params: Model parameters.
        tokens: ``[N, T]`` int32 in ``[0, vocab_size)`` — the
            +1-shifted code sequences.
        batch_size: Batch size for the forward pass.
        num_codes: Real codebook size (vocab_size = num_codes + 1).
        bos_id: Token index reserved for BOS — predictions equal to
            ``bos_id`` are ignored from the histogram (model should
            never emit BOS, but be defensive).

    Returns:
        ``int64`` array of length ``num_codes`` — counts of *predicted*
        codes after subtracting the +1 shift and dropping BOS.
    """
    import jax.numpy as jnp
    if tokens.size == 0:
        return np.zeros(num_codes, dtype=np.int64)
    preds = []
    n, T = tokens.shape
    # Forward in batches of `batch_size` (or one full pass if smaller).
    starts = list(range(0, n, batch_size)) if n >= batch_size else [0]
    for start in starts:
        end = min(start + batch_size, n) if n >= batch_size else n
        batch = tokens[start:end]
        if batch.shape[0] == 0:
            continue
        p = np.asarray(predict_fn(params, jnp.asarray(batch)))
        preds.append(p)
    pred_tokens = np.concatenate(preds, axis=0) if preds else np.zeros((0, T), dtype=np.int32)
    flat = pred_tokens.ravel()
    flat = flat[flat != bos_id]  # drop any (rare) BOS emissions
    flat = flat - 1  # back to [0, num_codes)
    flat = flat[(flat >= 0) & (flat < num_codes)]
    return np.bincount(flat, minlength=num_codes)[:num_codes].astype(np.int64)


def _log_code_dist(
    wandb_run,
    step: int,
    params,
    config,
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    batch_size: int,
    actual_train_hist: np.ndarray,
    actual_val_hist,
    num_codes: int,
    predict_fn,
) -> None:
    """Compute predicted code histograms and log bar charts for train/val.

    Shared by gpt2/mamba/s5 fit loops — they all have the same
    ``predict_fn(params, batch) -> argmax_tokens`` signature.
    """
    if wandb_run is None:
        return
    pred_train_hist = predicted_histogram_from_batches(
        predict_fn, params, train_tokens, batch_size, num_codes,
    )
    log_code_histograms_figure(
        wandb_run, step,
        {"actual": actual_train_hist, "predicted": pred_train_hist},
        num_codes, title="train_code_dist",
    )
    if actual_val_hist is not None and val_tokens.size > 0:
        pred_val_hist = predicted_histogram_from_batches(
            predict_fn, params, val_tokens, batch_size, num_codes,
        )
        log_code_histograms_figure(
            wandb_run, step,
            {"actual": actual_val_hist, "predicted": pred_val_hist},
            num_codes, title="val_code_dist",
        )
