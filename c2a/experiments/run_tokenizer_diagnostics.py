"""Tokenizer diagnostics — produces six artifacts:

1. KPMS EML curve (one line per ``n_states``, mean ± stderr over seeds).
2. Frequency-distribution heatmaps, one per tokenizer family (rows =
   hyperparameter configs, cols = syllable rank descending, color = per-row
   normalized frame frequency).
3. Median-duration histograms, one subplot per family, with the
   ``duration_target_ms`` reference line marked.
4. ``active_syllables_table.{md,csv}``.
5. ``instances_per_syllable_table.{md,csv}``.
6. ``tokenizer_diagnostics_summary.json``.

Reads:

* KPMS: ``sweep_results.json`` (for EML curve + tables for all 240
  KPMS runs) plus per-(setting, seed=representative_seed) ``results.h5``
  for the frequency heatmap.
* VAE / VQVAE / PS-VQVAE: ``best_codes.npz`` from each ckpt dir (the
  standalone + sweep). The npz already has the schema-compatible
  ``all_codes`` / ``train_codes`` / ``test_codes`` produced by the
  matching tokenizer's tokenize script.

Reuses ``c2a.experiments.shared.tokenizer_metrics`` and
``c2a.experiments.shared.kpms_results_loader`` so the metric math is
isolated and unit-tested.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from c2a.experiments.shared.kpms_results_loader import (
    kpms_project_dir_for_setting,
    load_kpms_syllables,
)
from c2a.experiments.shared.tokenizer_metrics import (
    TokenizerMetrics,
    compute_tokenizer_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("tokenizer_diagnostics")


# =============================================================================
# Per-family config-collection
# =============================================================================


@dataclass(frozen=True)
class FamilyConfig:
    """A single (family, config-label, codes) bundle for diagnostics."""

    family: str
    config_label: str
    codes: np.ndarray
    num_codes: int
    sort_key: float | tuple  # for stable row ordering inside a family
    source_info: dict | None = None  # family-specific provenance (e.g. KPMS setting/seed paths)


# =============================================================================
# Best-per-family selection
# =============================================================================


def score_config(
    metrics,
    *,
    target_duration_ms: float = 400.0,
    duration_sigma_ms: float = 200.0,
    active_threshold_key: str = "active_at_1pct",
) -> dict:
    """Score one config against the three Goldilocks axes.

    - duration: Gaussian centered at ``target_duration_ms`` with σ
      ``duration_sigma_ms`` (so off-by-200ms gets ~0.37, off-by-400ms
      gets ~0.018).
    - uniformity: ``H(freq) / log K`` ∈ [0, 1].
    - active_fraction: ``active@1% / K`` ∈ [0, 1].

    Returns the three sub-scores + their unweighted mean.
    """
    median_ms = float(metrics.median_duration_ms)
    K = max(int(metrics.num_codes), 1)
    freq = metrics.frame_frequency.astype(np.float64)
    total = float(freq.sum())
    if total > 0:
        p = freq / total
        p_pos = p[p > 0]
        entropy = float(-np.sum(p_pos * np.log(p_pos)))
        uniformity = entropy / math.log(max(K, 2))  # avoid /0 at K=1
    else:
        uniformity = 0.0

    active = float(getattr(metrics, active_threshold_key))
    active_fraction = active / K

    duration = math.exp(-((median_ms - target_duration_ms) / duration_sigma_ms) ** 2)

    combined = (duration + uniformity + active_fraction) / 3.0
    return {
        "duration_score": float(duration),
        "uniformity_score": float(uniformity),
        "active_fraction_score": float(active_fraction),
        "combined_score": float(combined),
        "median_duration_ms": median_ms,
        "K": K,
        "active_threshold_key": active_threshold_key,
    }


def select_best_per_family(
    family_to_configs: dict[str, list[FamilyConfig]],
    family_to_metrics: dict[str, list],
    *,
    target_duration_ms: float = 400.0,
    duration_sigma_ms: float = 200.0,
) -> dict[str, dict]:
    """Pick the highest-combined-score config in each family.

    Returns a dict ``{family: {config, K, score_breakdown, source_info}}``.
    """
    out: dict[str, dict] = {}
    for family, configs in family_to_configs.items():
        metrics = family_to_metrics[family]
        if not configs:
            continue
        scored = [
            (score_config(
                m,
                target_duration_ms=target_duration_ms,
                duration_sigma_ms=duration_sigma_ms,
            ), fc)
            for fc, m in zip(configs, metrics)
        ]
        scored.sort(key=lambda x: x[0]["combined_score"], reverse=True)
        best_score, best_fc = scored[0]
        out[family] = {
            "config": best_fc.config_label,
            "K": int(best_fc.num_codes),
            "score": best_score,
            "source_info": best_fc.source_info,
        }
    return out


def collect_kpms_configs(cfg: DictConfig) -> list[FamilyConfig]:
    """Read the KPMS sweep, return one FamilyConfig per (n_states, kappa).

    Only the ``representative_seed`` is loaded; other seeds inform the
    EML curve only.
    """
    sweep_root = Path(str(cfg.kpms.sweep_root))
    seed = int(cfg.kpms.representative_seed)

    n_states_in = [int(x) for x in cfg.kpms.heatmap_filter.get("n_states_in", []) or []]
    kappa_in = [int(x) for x in cfg.kpms.heatmap_filter.get("kappa_in", []) or []]

    with open(cfg.kpms.sweep_results_json) as f:
        sweep = json.load(f)
    all_results = sweep["all_results"]

    # Group by (n_states, kappa); for each group pick the result for the
    # requested representative seed.
    by_key: dict[tuple[int, int], dict] = {}
    for r in all_results:
        if int(r.get("seed", -1)) != seed:
            continue
        key = (int(r["n_states"]), int(r["kappa"]))
        by_key[key] = r

    out: list[FamilyConfig] = []
    for (n_states, kappa), r in sorted(by_key.items()):
        if n_states_in and n_states not in n_states_in:
            continue
        if kappa_in and kappa not in kappa_in:
            continue
        try:
            project_dir = kpms_project_dir_for_setting(
                sweep_root, setting=r["setting"], seed=seed,
            )
            codes = load_kpms_syllables(project_dir / "results.h5")
        except FileNotFoundError as exc:
            log.warning(
                "Skipping KPMS (n_states=%d, kappa=%d): %s",
                n_states, kappa, exc,
            )
            continue
        out.append(
            FamilyConfig(
                family="kpms",
                config_label=f"s={n_states}, κ={kappa}",
                codes=codes,
                num_codes=n_states,
                sort_key=(n_states, kappa),
                source_info={
                    "setting": r["setting"],
                    "seed": seed,
                    "n_states": n_states,
                    "kappa": kappa,
                    "project_dir": str(project_dir),
                    "results_h5": str(project_dir / "results.h5"),
                },
            )
        )
    log.info("Collected %d KPMS configs (seed=%d).", len(out), seed)
    return out


def _load_npz_codes(npz_path: Path) -> tuple[np.ndarray, int]:
    """Return ``(all_codes, num_codes)`` from a tokenizer best_codes npz."""
    data = np.load(str(npz_path), allow_pickle=False)
    if "all_codes" in data:
        codes = np.asarray(data["all_codes"], dtype=np.int32)
    else:
        # Older npz: concat train + test in their stored order.
        train = np.asarray(data["train_codes"], dtype=np.int32)
        test = np.asarray(data["test_codes"], dtype=np.int32)
        codes = np.concatenate([train, test], axis=0)
    if "num_states" in data:
        num_codes = int(data["num_states"].item())
    else:
        num_codes = int(codes.max()) + 1
    return codes, num_codes


def collect_neural_configs(family_cfg: DictConfig) -> list[FamilyConfig]:
    """Collect one FamilyConfig per ckpt dir under sweep_dir + standalone."""
    family = str(family_cfg.family)
    out: list[FamilyConfig] = []

    # Standalone (the default ckpt at the family root).
    standalone = family_cfg.get("standalone", None)
    if standalone is not None:
        ckpt_dir = Path(str(standalone.ckpt_dir))
        npz = ckpt_dir / "best_codes.npz"
        if npz.exists():
            codes, K = _load_npz_codes(npz)
            out.append(
                FamilyConfig(
                    family=family,
                    config_label=str(standalone.get("label", "default")),
                    codes=codes,
                    num_codes=K,
                    sort_key=(K, 0),
                    source_info={
                        "ckpt_dir": str(ckpt_dir),
                        "best_codes_npz": str(npz),
                        "kind": "standalone",
                    },
                )
            )
        else:
            log.warning("Standalone %s ckpt missing: %s", family, npz)

    # Sweep entries.
    sweep_dir = Path(str(family_cfg.sweep_dir))
    if sweep_dir.exists():
        for sub in sorted(sweep_dir.iterdir()):
            if not sub.is_dir():
                continue
            npz = sub / "best_codes.npz"
            if not npz.exists():
                continue
            codes, K = _load_npz_codes(npz)
            out.append(
                FamilyConfig(
                    family=family,
                    config_label=sub.name,
                    codes=codes,
                    num_codes=K,
                    sort_key=(K, hash(sub.name) & 0xFFFF),
                    source_info={
                        "ckpt_dir": str(sub),
                        "best_codes_npz": str(npz),
                        "kind": "sweep",
                    },
                )
            )

    out.sort(key=lambda fc: fc.sort_key)
    log.info("Collected %d %s configs.", len(out), family)
    return out


# =============================================================================
# Plot 1: KPMS EML curve
# =============================================================================


def plot_kpms_eml_curve(cfg: DictConfig, out_dir: Path) -> dict:
    """One curve per n_states, with mean ± stderr over seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(cfg.kpms.sweep_results_json) as f:
        sweep = json.load(f)
    all_results = sweep["all_results"]
    best = sweep.get("best_model", {})

    # Group: (n_states, kappa) -> list[eml]
    groups: dict[tuple[int, int], list[float]] = {}
    for r in all_results:
        if r.get("eml_score") is None:
            continue
        groups.setdefault(
            (int(r["n_states"]), int(r["kappa"])),
            [],
        ).append(float(r["eml_score"]))

    n_states_unique = sorted({k[0] for k in groups})
    summary: dict = {"n_states": {}, "best": best}

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(n_states_unique), 1)))
    for color, n_states in zip(colors, n_states_unique):
        ks = sorted({k[1] for k in groups if k[0] == n_states})
        means, stderrs = [], []
        for k in ks:
            vals = np.asarray(groups[(n_states, k)], dtype=np.float64)
            means.append(float(vals.mean()))
            stderrs.append(
                float(vals.std(ddof=1) / np.sqrt(max(vals.size, 1)))
                if vals.size > 1 else 0.0
            )
        means_arr = np.asarray(means)
        stderrs_arr = np.asarray(stderrs)
        ax.errorbar(
            ks, means_arr, yerr=stderrs_arr,
            marker="o", capsize=3, linewidth=1.2,
            label=f"n_states={n_states}", color=color,
        )
        summary["n_states"][int(n_states)] = {
            "kappa": [int(k) for k in ks],
            "eml_mean": means,
            "eml_stderr": stderrs,
            "n_seeds": int(np.asarray([len(groups[(n_states, k)]) for k in ks]).max()),
        }

    # Mark the chosen best.
    if best:
        ax.axvline(
            float(best["kappa"]),
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=0.9,
            label=(
                f"best: n_states={int(best['n_states'])}, "
                f"κ={int(best['kappa'])}"
            ),
        )

    ax.set_xscale("log")
    ax.set_xlabel("κ (kappa) — log scale")
    ax.set_ylabel("EML score (higher = better)")
    ax.set_title("KPMS kappa sweep — Expected Marginal Likelihood")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    png = out_dir / "kpms_eml_curve.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    log.info("Wrote %s", png)
    return summary


# =============================================================================
# Plot 2: Frequency-distribution heatmap (one per family)
# =============================================================================


def plot_frequency_heatmap(
    family: str,
    configs: list[FamilyConfig],
    metrics_per_config: list[TokenizerMetrics],
    out_dir: Path,
    norm: str = "per_row_fraction",
    best_label: str | None = None,
) -> None:
    """Heatmap: rows = configs (sorted by K), cols = syllable RANK
    (descending frame frequency within each row), color = frequency.

    Cross-row columns are ranks, not code IDs — same column doesn't
    mean same syllable across rows. This deliberate choice makes the
    head/tail PROFILE comparable across configs even though raw code
    labels aren't aligned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not configs:
        log.info("No %s configs; skipping heatmap.", family)
        return

    max_K = max(c.num_codes for c in configs)
    n_rows = len(configs)
    mat = np.full((n_rows, max_K), np.nan, dtype=np.float64)
    row_labels: list[str] = []

    for i, (fc, m) in enumerate(zip(configs, metrics_per_config)):
        freqs = m.frame_frequency.astype(np.float64)
        sorted_desc = np.sort(freqs)[::-1]
        if norm == "per_row_fraction":
            total = sorted_desc.sum()
            if total > 0:
                sorted_desc = sorted_desc / total
        mat[i, : sorted_desc.shape[0]] = sorted_desc
        row_labels.append(fc.config_label)

    # Clamp the figure width so that very-large-K families (e.g. PS-VQ-VAE
    # K=1024) don't blow up the aspect ratio. matplotlib clips at the
    # configured DPI × inches budget anyway.
    fig_w = float(np.clip(0.18 * max_K + 2, 6, 22))
    fig_h = float(max(3, 0.32 * n_rows + 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        mat, aspect="auto", cmap="viridis",
        interpolation="nearest", origin="upper",
    )
    ax.set_xlabel("syllable rank (descending frame frequency)")
    ax.set_ylabel(f"{family} config")
    ax.set_yticks(np.arange(n_rows))
    # Mark the best row with a leading ★ + bold.
    decorated_labels = []
    for lbl in row_labels:
        if best_label is not None and lbl == best_label:
            decorated_labels.append(f"★ {lbl}")
        else:
            decorated_labels.append(lbl)
    ax.set_yticklabels(decorated_labels, fontsize=8)
    # Highlight the best row with a red rectangle around the full row.
    if best_label is not None:
        for i, lbl in enumerate(row_labels):
            if lbl == best_label:
                ax.add_patch(plt.Rectangle(
                    (-0.5, i - 0.5), max_K, 1.0,
                    fill=False, edgecolor="#e74c3c", lw=1.5,
                ))
                break
    ax.set_xticks(np.arange(0, max_K, max(max_K // 10, 1)))
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("frame fraction" if norm == "per_row_fraction" else "frame count")

    family_title = {
        "kpms": "KPMS", "vae": "VAE", "vqvae": "VQ-VAE", "ps_vqvae": "PS-VQ-VAE",
    }.get(family, family)
    ax.set_title(
        f"{family_title} — frame-frequency distribution per config\n"
        f"(columns = ranks within each row; not aligned across rows)"
    )
    fig.tight_layout()

    png = out_dir / f"frequency_heatmap_{family}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Plot 3: Median-duration histograms (one subplot per family)
# =============================================================================


def plot_duration_histograms(
    family_to_configs: dict[str, list[FamilyConfig]],
    family_to_metrics: dict[str, list[TokenizerMetrics]],
    out_dir: Path,
    fps: float = 50.0,
    target_ms: float = 400.0,
) -> None:
    """One subplot per family — pooled run-length histogram per config,
    with each config's median marked and a global target line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = list(family_to_configs.keys())
    n = len(families)
    if n == 0:
        return
    # 2x2 grid (or wider single row if ≤2 families) — easier to read
    # than a tall single column when n=4.
    if n <= 2:
        nrows, ncols = 1, n
    else:
        nrows, ncols = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 3.5 * nrows), squeeze=False,
    )
    axes_flat = axes.ravel()
    ms_per_frame = 1000.0 / fps

    for ax, family in zip(axes_flat, families):
        configs = family_to_configs[family]
        metrics = family_to_metrics[family]
        if not configs:
            ax.set_visible(False)
            continue
        # Pool a subset to avoid the chart becoming unreadable. If many
        # configs in a family, show up to 6 evenly-spaced.
        keep = configs if len(configs) <= 6 else (
            [configs[i] for i in np.linspace(0, len(configs) - 1, 6).astype(int)]
        )
        keep_metrics = metrics if len(metrics) <= 6 else (
            [metrics[i] for i in np.linspace(0, len(metrics) - 1, 6).astype(int)]
        )

        # Use a global x-axis upper bound based on the 95th percentile
        # of all bout lengths in this family — clips the long tail so
        # the bulk of the distribution is readable.
        all_ms = np.concatenate([
            m.bout_lengths.astype(np.float64) * ms_per_frame
            for m in keep_metrics if m.bout_lengths.size > 0
        ]) if any(m.bout_lengths.size > 0 for m in keep_metrics) else np.array([])
        if all_ms.size > 0:
            x_max = float(max(np.quantile(all_ms, 0.95), target_ms * 2.0))
        else:
            x_max = target_ms * 2.5

        cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(keep)))
        for color, fc, m in zip(cmap, keep, keep_metrics):
            if m.bout_lengths.size == 0:
                continue
            ms = m.bout_lengths.astype(np.float64) * ms_per_frame
            ax.hist(
                ms, bins=40, range=(0, x_max),
                histtype="step", color=color, label=fc.config_label,
                density=True, linewidth=1.0,
            )
            ax.axvline(
                m.median_duration_ms, color=color, linestyle=":",
                alpha=0.6, linewidth=0.8,
            )

        ax.axvline(
            target_ms, color="red", linestyle="--",
            linewidth=1.0, label=f"target {target_ms:.0f} ms",
        )
        family_title = {
            "kpms": "KPMS", "vae": "VAE",
            "vqvae": "VQ-VAE", "ps_vqvae": "PS-VQ-VAE",
        }.get(family, family)
        ax.set_xlabel("bout duration (ms)")
        ax.set_ylabel("density")
        ax.set_title(f"{family_title}\n(dotted = per-config median)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.3)

    # Hide unused axes (when n is odd in a 2x2 grid).
    for ax in axes_flat[len(families):]:
        ax.set_visible(False)

    fig.tight_layout()
    png = out_dir / "median_duration_hist.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Tables
# =============================================================================


_FAMILY_COLORS: dict[str, str] = {
    "kpms": "#cfe2f3",      # pale blue
    "vae": "#d9ead3",       # pale green
    "vqvae": "#fff2cc",     # pale yellow
    "ps_vqvae": "#f4cccc",  # pale red
}


def _save_table_png(
    rows: list[dict],
    columns: list[tuple[str, str, str]],
    title: str,
    out_path: Path,
    best_rows: set[tuple[str, str]] | None = None,
) -> None:
    """Render a list-of-dicts as a matplotlib table PNG.

    Args:
        rows: source rows. Each row must have all keys referenced in
            ``columns``.
        columns: ``(key, header, fmt)`` tuples. ``fmt`` is a Python
            format string applied to ``row[key]``; e.g. ``"{:d}"`` for
            int, ``"{:.1f}"`` for float, ``"{}"`` for plain str.
        title: figure title.
        out_path: PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        log.warning("No rows for %s; skipping.", out_path.name)
        return

    headers = [h for _, h, _ in columns]
    cell_text: list[list[str]] = []
    row_colors: list[str] = []
    best_set = best_rows or set()
    for r in rows:
        cells = [fmt.format(r[k]) for k, _, fmt in columns]
        if (r.get("family", ""), r.get("config", "")) in best_set:
            # Prepend ★ to the config cell of the winning row.
            cfg_idx = next(
                (i for i, (k, _, _) in enumerate(columns) if k == "config"), None,
            )
            if cfg_idx is not None:
                cells[cfg_idx] = f"★ {cells[cfg_idx]}"
            row_colors.append("#ffd966")  # gold highlight
        else:
            row_colors.append(_FAMILY_COLORS.get(r.get("family", ""), "#ffffff"))
        cell_text.append(cells)

    n_rows = len(rows)
    n_cols = len(columns)
    # Figure size: per-cell budget. ~0.32" tall per row, ~1.1" wide per col,
    # plus margin for title.
    fig_w = max(7.0, 1.1 * n_cols + 1.5)
    fig_h = max(2.5, 0.32 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=[[c] * n_cols for c in row_colors],
        colColours=["#e0e0e0"] * n_cols,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.3)
    # Bold header row
    for j in range(n_cols):
        tbl[(0, j)].set_text_props(weight="bold")

    ax.set_title(title, fontsize=11, pad=10, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def write_tables(
    rows: list[dict],
    out_dir: Path,
    fps: float = 50.0,
    best_per_family: dict[str, dict] | None = None,
) -> None:
    """Active-syllables + instances-per-syllable tables as PNG + canonical CSV."""
    best_set: set[tuple[str, str]] = set()
    if best_per_family:
        for family, entry in best_per_family.items():
            best_set.add((family, entry["config"]))
    import csv as csvmod

    # --- CSV (one canonical file with all columns; for machine reads) ---
    csv_path = out_dir / "tokenizer_diagnostics.csv"
    if rows:
        with open(csv_path, "w") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log.info("Wrote %s", csv_path)

    # --- Active syllables table (PNG) ---
    _save_table_png(
        rows=rows,
        columns=[
            ("family", "family", "{}"),
            ("config", "config", "{}"),
            ("K", "K", "{:d}"),
            ("active_any", "active@any", "{:d}"),
            ("active_at_0p1pct", "active@0.1%", "{:d}"),
            ("active_at_1pct", "active@1%", "{:d}"),
            ("usage_ratio", "usage ratio", "{:.3f}"),
        ],
        title="Active syllables per tokenizer config (★ = best in family)",
        out_path=out_dir / "active_syllables_table.png",
        best_rows=best_set,
    )

    # --- Instances-per-syllable + duration table (PNG) ---
    _save_table_png(
        rows=rows,
        columns=[
            ("family", "family", "{}"),
            ("config", "config", "{}"),
            ("K", "K", "{:d}"),
            ("n_bouts_total", "n_bouts", "{:d}"),
            ("instances_per_code_min", "inst/code (min)", "{:d}"),
            ("instances_per_code_median", "inst/code (median)", "{:.1f}"),
            ("instances_per_code_max", "inst/code (max)", "{:d}"),
            ("mean_duration_ms", "mean dur (ms)", "{:.1f}"),
            ("median_duration_ms", "median dur (ms)", "{:.1f}"),
        ],
        title="Instances per syllable + duration (★ = best in family)",
        out_path=out_dir / "instances_per_syllable_table.png",
        best_rows=best_set,
    )


# =============================================================================
# Main
# =============================================================================


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="tokenizer_diagnostics",
)
def main(cfg: DictConfig) -> None:
    out_dir = Path(str(cfg.output.base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    # --- 1. KPMS EML curve (also yields the summary dict) ---
    eml_summary = plot_kpms_eml_curve(cfg, out_dir)

    # --- 2. Collect per-family configs ---
    fps = float(cfg.neural.fps)
    family_to_configs: dict[str, list[FamilyConfig]] = {
        "kpms": collect_kpms_configs(cfg),
        "vae": collect_neural_configs(cfg.neural.vae),
        "vqvae": collect_neural_configs(cfg.neural.vqvae),
        "ps_vqvae": collect_neural_configs(cfg.neural.ps_vqvae),
    }

    # --- 3. Compute metrics per config ---
    family_to_metrics: dict[str, list[TokenizerMetrics]] = {}
    rows: list[dict] = []
    summary: dict = {"eml_summary": eml_summary, "configs": []}
    for family, configs in family_to_configs.items():
        metrics_list: list[TokenizerMetrics] = []
        for fc in configs:
            m = compute_tokenizer_metrics(
                fc.codes, num_codes=fc.num_codes, fps=fps,
            )
            metrics_list.append(m)
            row = m.to_table_row(family=family, config=fc.config_label)
            rows.append(row)
            summary["configs"].append(row)
        family_to_metrics[family] = metrics_list

    # --- 4. Best-per-family selection (Goldilocks score) ---
    best_per_family = select_best_per_family(
        family_to_configs, family_to_metrics,
        target_duration_ms=float(cfg.plotting.duration_target_ms),
        duration_sigma_ms=float(cfg.plotting.get("duration_sigma_ms", 200.0)),
    )
    summary["best_per_family"] = best_per_family
    log.info("Best per family (Goldilocks score):")
    for family, entry in best_per_family.items():
        s = entry["score"]
        log.info(
            "  %-9s  %s  (K=%d, dur=%.0fms, score=%.3f = dur %.2f + uni %.2f + act %.2f)",
            family, entry["config"], entry["K"],
            s["median_duration_ms"], s["combined_score"],
            s["duration_score"], s["uniformity_score"], s["active_fraction_score"],
        )

    # --- 5. Frequency heatmaps (one per family) ---
    for family in family_to_configs:
        plot_frequency_heatmap(
            family,
            family_to_configs[family],
            family_to_metrics[family],
            out_dir,
            norm=str(cfg.plotting.heatmap_norm),
            best_label=best_per_family.get(family, {}).get("config"),
        )

    # --- 6. Median-duration histograms ---
    plot_duration_histograms(
        family_to_configs, family_to_metrics, out_dir,
        fps=fps, target_ms=float(cfg.plotting.duration_target_ms),
    )

    # --- 7. Tables ---
    write_tables(rows, out_dir, fps=fps, best_per_family=best_per_family)

    # --- 8. Summary JSON ---
    summary_path = out_dir / "tokenizer_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)

    log.info("=" * 60)
    log.info("Done — %d configs across %d families.",
             len(rows), len(family_to_configs))


if __name__ == "__main__":
    main()
