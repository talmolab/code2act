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
# Model-fit extraction (per-family fit metric)
# =============================================================================


def _parse_neural_final_loss(ckpt_dir: Path) -> float | None:
    """Read the last reported training loss from a neural-tokenizer training log.

    Each neural-family train script (vae/vqvae/ps_vqvae) writes lines of the
    form ``... step <N>  loss=<float>  ...`` into ``train.log`` in the ckpt
    directory. We return the final-step loss; ``None`` if not parseable.
    """
    log_path = Path(ckpt_dir) / "train.log"
    if not log_path.exists():
        return None
    last_loss: float | None = None
    try:
        with log_path.open("r", errors="ignore") as f:
            for line in f:
                # Match ``loss=<num>`` anywhere on the line.
                idx = line.find("loss=")
                if idx < 0:
                    continue
                tail = line[idx + 5:]
                # Stop at first non-numeric / non-sign / non-dot / non-e char.
                end = 0
                while end < len(tail) and (
                    tail[end].isdigit() or tail[end] in ".+-eE"
                ):
                    end += 1
                if end == 0:
                    continue
                try:
                    last_loss = float(tail[:end])
                except ValueError:
                    continue
    except OSError:
        return None
    return last_loss


def extract_model_fit(
    family: str,
    fc: FamilyConfig,
    sweep_results: dict | None,
) -> float | None:
    """Return one scalar fit per config. Larger = better.

    * KPMS: mean EML across seeds for the (n_states, kappa, model_type)
      triple this FamilyConfig points to (already log-likelihood; higher
      is better natively).
    * Neural (vae, vqvae, ps_vqvae): negative final training loss read
      from the matching ``train.log`` — so higher is still better.
    """
    if family == "kpms":
        if sweep_results is None or fc.source_info is None:
            return None
        target = (
            int(fc.source_info["n_states"]),
            int(fc.source_info["kappa"]),
            str(fc.source_info.get("model_type", "arhmm")),
        )
        scores: list[float] = []
        for r in sweep_results.get("all_results", []):
            if (
                int(r.get("n_states", -1)) == target[0]
                and int(r.get("kappa", -1)) == target[1]
                and str(r.get("model_type", "arhmm")) == target[2]
                and r.get("eml_score") is not None
            ):
                scores.append(float(r["eml_score"]))
        return float(np.mean(scores)) if scores else None
    # neural: source_info["ckpt_dir"] is the ckpt root for sweep configs.
    if fc.source_info is None or "ckpt_dir" not in fc.source_info:
        return None
    ckpt_dir = Path(str(fc.source_info["ckpt_dir"]))
    # For VAE sweep, the train.log lives in the kl_<tag>/ root (one level
    # up from the flattened ``kl_<tag>_k<K>`` -> ``kl_<tag>/tokenize_k<K>``
    # symlinks I added for diagnostics). Resolve through symlinks first,
    # then walk up two levels to be safe across families.
    resolved = ckpt_dir.resolve()
    candidates = [
        ckpt_dir, ckpt_dir.parent,
        resolved, resolved.parent, resolved.parent.parent,
    ]
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        loss = _parse_neural_final_loss(c)
        if loss is not None and np.isfinite(loss):
            return -float(loss)  # invert: bigger = better
    return None


# =============================================================================
# Purity metric (MMD: within-syllable vs across-syllable on kinematic space)
# =============================================================================


# Cap on pooled (frame, feature) samples per MMD argument — matches the
# code-distinctness experiment for parity with the c2a metric.
_PURITY_MAX_POOL = 1000


def _pool_bouts(
    bouts: list[np.ndarray], strip_root: bool, rng: np.random.RandomState,
) -> np.ndarray:
    """Pool a list of ``(T_i, nq)`` bouts into ``(<= _PURITY_MAX_POOL, feat)``.

    Mirrors ``run_code_distinctness._pool``: optionally strips the root
    xyz (first 3 dims of qpos) so the MMD compares POSTURE, not absolute
    body position, then subsamples uniformly to cap memory.
    """
    if not bouts:
        return np.zeros((0, 1), dtype=np.float32)
    flat = np.concatenate(
        [b.reshape(b.shape[0], -1) for b in bouts], axis=0,
    ).astype(np.float32)
    if strip_root and flat.shape[1] > 3:
        flat = flat[:, 3:]
    if flat.shape[0] > _PURITY_MAX_POOL:
        flat = flat[rng.choice(flat.shape[0], _PURITY_MAX_POOL, replace=False)]
    return flat


def _estimate_sigma_from_pooled(
    pooled: np.ndarray, rng: np.random.RandomState, n_pairs: int = 1024,
) -> float:
    """Median pairwise L2 distance over ``n_pairs`` random sample pairs.

    Standard RBF-MMD bandwidth heuristic — sets the kernel scale to the
    data's intrinsic length scale.
    """
    if pooled.shape[0] < 2:
        return 1.0
    n = pooled.shape[0]
    i = rng.randint(0, n, size=n_pairs)
    j = rng.randint(0, n, size=n_pairs)
    keep = i != j
    if not keep.any():
        return 1.0
    diffs = pooled[i[keep]] - pooled[j[keep]]
    d = np.linalg.norm(diffs, axis=1)
    s = float(np.median(d))
    return max(s, 1e-6)


def _extract_bouts_per_syllable(
    codes: np.ndarray, ref_qpos: np.ndarray, clip_indices: np.ndarray,
    K: int,
) -> dict[int, list[np.ndarray]]:
    """For each syllable, return a list of ``(T, nq)`` bouts pulled from the
    reference data at the right (clip, start:end) slices.

    ``codes`` is shape ``(n_local_clips, n_frames)`` and ``clip_indices`` is
    the matching global clip indices into ``ref_qpos`` (shape
    ``(n_global_clips, n_frames, nq)``).
    """
    out: dict[int, list[np.ndarray]] = {c: [] for c in range(K)}
    n_local, n_frames = codes.shape
    for local_i in range(n_local):
        global_i = int(clip_indices[local_i])
        if global_i >= ref_qpos.shape[0]:
            continue
        clip_codes = codes[local_i]
        change = np.empty(clip_codes.shape, dtype=bool)
        change[0] = True
        change[1:] = clip_codes[1:] != clip_codes[:-1]
        starts = np.where(change)[0]
        ends = np.append(starts[1:], n_frames)
        for s, e in zip(starts, ends):
            v = int(clip_codes[s])
            if v < 0 or v >= K:
                continue
            out[v].append(ref_qpos[global_i, s:e])
    return out


def compute_tokenizer_purity(
    fc: FamilyConfig,
    ref_qpos: np.ndarray,
    *,
    strip_root: bool = True,
    n_across_pairs: int = 30,
    min_bouts_for_within: int = 4,
    rng: np.random.RandomState | None = None,
) -> dict:
    """MMD-based purity for one tokenizer config.

    Definition:

    * For each active syllable c with ≥ ``min_bouts_for_within`` bouts,
      compute MMD between two random halves of its bouts → one
      *within-syllable* sample.
    * For ``n_across_pairs`` random unordered pairs (c, c'), compute MMD
      between c's pooled bouts and c''s pooled bouts → one
      *across-syllable* sample.
    * ``purity_score = median(across) / max(median(within), eps)`` — higher
      = each syllable is internally tighter than the typical pair of
      different syllables.

    All MMDs use a single RBF bandwidth estimated as the median pairwise
    L2 distance over a random sample drawn from the entire (frame, feat)
    pool — so families with different feature scales remain comparable.
    """
    from c2a.experiments.shared.code_grid import mmd_squared

    rng = rng or np.random.RandomState(0)
    codes = np.asarray(fc.codes, dtype=np.int32)
    K = int(fc.num_codes)

    # Resolve clip indices for the codes' rows.
    src = fc.source_info or {}
    if fc.family == "kpms":
        # KPMS codes are loaded from results.h5 in the order returned by
        # load_kpms_syllables; for purity we assume all 484 train+test
        # clips are present at indices 0..n-1.
        # That ordering matches the kpms_sweep config (clip_set='all',
        # csv_to_h5_subsample=2 → 842 clips * 50% = 421… actually 484).
        # We treat codes' row i as referring to the i-th clip in
        # ref_qpos.
        clip_indices = np.arange(codes.shape[0], dtype=np.int64)
    else:
        # Neural families' best_codes.npz stores explicit train/test
        # indices into the full reference. Concat them in the same order
        # _load_npz_codes used when constructing fc.codes.
        npz_path = src.get("best_codes_npz")
        if npz_path is None:
            clip_indices = np.arange(codes.shape[0], dtype=np.int64)
        else:
            d = np.load(str(npz_path), allow_pickle=False)
            if "all_codes" in d.files:
                # all_codes is concat(train, test); indices follow same order.
                tr = np.asarray(d.get("train_indices", []), dtype=np.int64)
                te = np.asarray(d.get("test_indices", []), dtype=np.int64)
                if tr.size + te.size == codes.shape[0]:
                    clip_indices = np.concatenate([tr, te])
                else:
                    clip_indices = np.arange(codes.shape[0], dtype=np.int64)
            else:
                clip_indices = np.arange(codes.shape[0], dtype=np.int64)

    # 1. Bin bouts by syllable.
    bouts_per_syl = _extract_bouts_per_syllable(
        codes, ref_qpos, clip_indices, K,
    )

    # 2. Restrict to syllables with enough bouts for the within-split.
    active = [c for c, b in bouts_per_syl.items() if len(b) >= min_bouts_for_within]
    if len(active) < 2:
        return {
            "purity_score": float("nan"),
            "within_median": float("nan"),
            "across_median": float("nan"),
            "within_samples": [],
            "across_samples": [],
            "n_active_for_purity": int(len(active)),
            "sigma": float("nan"),
        }

    # 3. Estimate sigma from a global pool sample.
    pool_all = _pool_bouts(
        sum((bouts_per_syl[c] for c in active), []),
        strip_root, rng,
    )
    sigma = _estimate_sigma_from_pooled(pool_all, rng)

    # 4. Within-syllable: MMD between two random halves of each active
    #    syllable's bouts.
    within_samples: list[float] = []
    for c in active:
        bouts = list(bouts_per_syl[c])
        rng.shuffle(bouts)
        half = len(bouts) // 2
        if half < 1 or len(bouts) - half < 1:
            continue
        a = _pool_bouts(bouts[:half], strip_root, rng)
        b = _pool_bouts(bouts[half:], strip_root, rng)
        if a.shape[0] == 0 or b.shape[0] == 0:
            continue
        mmd2 = mmd_squared(a, b, sigma)
        within_samples.append(float(np.sqrt(max(mmd2, 0.0))))

    # 5. Across-syllable: MMD between bouts of two distinct random
    #    syllables, drawn ``n_across_pairs`` times.
    across_samples: list[float] = []
    for _ in range(int(n_across_pairs)):
        pair = rng.choice(len(active), size=2, replace=False)
        c1, c2 = active[int(pair[0])], active[int(pair[1])]
        a = _pool_bouts(bouts_per_syl[c1], strip_root, rng)
        b = _pool_bouts(bouts_per_syl[c2], strip_root, rng)
        if a.shape[0] == 0 or b.shape[0] == 0:
            continue
        mmd2 = mmd_squared(a, b, sigma)
        across_samples.append(float(np.sqrt(max(mmd2, 0.0))))

    if not within_samples or not across_samples:
        return {
            "purity_score": float("nan"),
            "within_median": float("nan"),
            "across_median": float("nan"),
            "within_samples": within_samples,
            "across_samples": across_samples,
            "n_active_for_purity": int(len(active)),
            "sigma": float(sigma),
        }
    within_med = float(np.median(within_samples))
    across_med = float(np.median(across_samples))
    purity_score = float(across_med / max(within_med, 1e-9))
    return {
        "purity_score": purity_score,
        "within_median": within_med,
        "across_median": across_med,
        "within_samples": within_samples,
        "across_samples": across_samples,
        "n_active_for_purity": int(len(active)),
        "sigma": float(sigma),
    }


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

    # Group by (n_states, kappa, model_type); for each group pick the
    # result for the requested representative seed.
    by_key: dict[tuple[int, int, str], dict] = {}
    for r in all_results:
        if int(r.get("seed", -1)) != seed:
            continue
        mt = str(r.get("model_type", "arhmm"))
        key = (int(r["n_states"]), int(r["kappa"]), mt)
        by_key[key] = r

    out: list[FamilyConfig] = []
    for (n_states, kappa, _mt), r in sorted(by_key.items()):
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
        model_type = str(r.get("model_type", "arhmm"))
        out.append(
            FamilyConfig(
                family="kpms",
                config_label=f"s={n_states}, κ={kappa}, {model_type}",
                codes=codes,
                num_codes=n_states,
                sort_key=(model_type, n_states, kappa),
                source_info={
                    "setting": r["setting"],
                    "seed": seed,
                    "n_states": n_states,
                    "kappa": kappa,
                    "model_type": model_type,
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

    # Group: (n_states, kappa, model_type) -> list[eml]
    groups: dict[tuple[int, int, str], list[float]] = {}
    for r in all_results:
        if r.get("eml_score") is None:
            continue
        model_type = str(r.get("model_type", "arhmm"))
        groups.setdefault(
            (int(r["n_states"]), int(r["kappa"]), model_type),
            [],
        ).append(float(r["eml_score"]))

    n_states_unique = sorted({k[0] for k in groups})
    model_types_unique = sorted({k[2] for k in groups})
    summary: dict = {"n_states": {}, "best": best}

    # Side-by-side panels: one per model_type (ARHMM, SLDS).
    n_panels = max(len(model_types_unique), 1)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.5 * n_panels, 4.2), sharey=False,
    )
    if n_panels == 1:
        axes = [axes]
    n_states_colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(n_states_unique), 1)))
    for ax, mt in zip(axes, model_types_unique):
        for color, n_states in zip(n_states_colors, n_states_unique):
            ks = sorted({k[1] for k in groups if k[0] == n_states and k[2] == mt})
            if not ks:
                continue
            means, stderrs = [], []
            for k in ks:
                vals = np.asarray(groups[(n_states, k, mt)], dtype=np.float64)
                means.append(float(vals.mean()))
                stderrs.append(
                    float(vals.std(ddof=1) / np.sqrt(max(vals.size, 1)))
                    if vals.size > 1 else 0.0
                )
            means_arr = np.asarray(means)
            stderrs_arr = np.asarray(stderrs)
            label = f"n_states={n_states}" if len(n_states_unique) > 1 else None
            ax.errorbar(
                ks, means_arr, yerr=stderrs_arr,
                marker="o", linestyle="-", capsize=3, linewidth=1.2,
                label=label, color=color,
            )
            summary["n_states"].setdefault(int(n_states), {})[mt] = {
                "kappa": [int(k) for k in ks],
                "eml_mean": means,
                "eml_stderr": stderrs,
                "n_seeds": int(
                    np.asarray([len(groups[(n_states, k, mt)]) for k in ks]).max()
                ),
            }

        # Mark the chosen best only if it matches this panel's model_type.
        if best and str(best.get("model_type", "")) == mt:
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
        ax.set_title(f"{mt.upper()}")
        if len(n_states_unique) > 1 or (best and str(best.get("model_type", "")) == mt):
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("KPMS kappa sweep — Expected Marginal Likelihood", y=1.02)
    fig.tight_layout()

    png = out_dir / "kpms_eml_curve.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    log.info("Wrote %s", png)
    return summary


# =============================================================================
# Plot 2: Fit vs purity scatter (one panel per family)
# =============================================================================


def plot_fit_vs_purity_scatter(
    rows: list[dict], out_dir: Path, best_per_family: dict[str, dict],
) -> None:
    """Per-family scatter: x = model fit (higher = better), y = purity score
    (higher = better). Each dot is one config; the best config per family is
    starred and labeled. The top-right corner of each panel is the sweet spot.

    Fit scales are family-specific (KPMS EML vs negative neural loss), so
    each panel uses its own x range — no cross-family fit comparison.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({r["family"] for r in rows})
    n = len(families)
    if n == 0:
        return
    fig, axes = plt.subplots(
        1, n, figsize=(5.0 * n, 4.2), squeeze=False,
    )
    family_title = {
        "kpms": "KPMS",
        "vae": "VAE",
        "vqvae": "VQ-VAE",
        "ps_vqvae": "PS-VQ-VAE",
    }
    for ax, family in zip(axes[0], families):
        fam_rows = [
            r for r in rows
            if r["family"] == family
            and r.get("model_fit") is not None
            and r.get("purity_score") is not None
            and not math.isnan(float(r["purity_score"]))
        ]
        if not fam_rows:
            ax.set_visible(False)
            continue
        xs = np.asarray([r["model_fit"] for r in fam_rows], dtype=float)
        ys = np.asarray([r["purity_score"] for r in fam_rows], dtype=float)
        ax.scatter(xs, ys, s=42, alpha=0.75, color="#3b82c4", edgecolor="white",
                   linewidth=0.6)
        for r, x, y in zip(fam_rows, xs, ys):
            ax.annotate(
                str(r["config"]), (x, y),
                fontsize=6, alpha=0.75,
                xytext=(3, 3), textcoords="offset points",
            )
        best = best_per_family.get(family)
        if best is not None:
            bx, by = best.get("model_fit"), best.get("purity_score")
            if bx is not None and by is not None and not math.isnan(by):
                ax.scatter(
                    [bx], [by], s=160, marker="*",
                    color="#e74c3c", edgecolor="black", linewidth=0.8,
                    zorder=5, label=f"★ {best['config']}",
                )
                ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("model fit (higher = better)")
        ax.set_ylabel("purity score = median(across) / median(within)")
        ax.set_title(family_title.get(family, family))
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "Tokenizer diagnostics: fit (model evidence) vs purity (kinematic MMD)",
        y=1.02,
    )
    fig.tight_layout()
    png = out_dir / "fit_vs_purity_scatter.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Plot 3: Per-family purity breakdown (within vs across MMD distributions)
# =============================================================================


def plot_purity_breakdown(
    rows: list[dict], out_dir: Path, best_per_family: dict[str, dict],
) -> None:
    """Per-family panels overlaying the within-syllable MMD distribution
    against the across-syllable MMD distribution for the BEST config (and
    a backdrop of all configs in that family in faint gray). A clean tail
    separation between the two distributions means the family's syllables
    are tighter than they are different.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted({r["family"] for r in rows})
    n = len(families)
    if n == 0:
        return
    if n <= 2:
        nrows, ncols = 1, n
    else:
        nrows, ncols = 2, (n + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows), squeeze=False,
    )
    axes_flat = axes.ravel()
    family_title = {
        "kpms": "KPMS",
        "vae": "VAE",
        "vqvae": "VQ-VAE",
        "ps_vqvae": "PS-VQ-VAE",
    }
    for ax, family in zip(axes_flat, families):
        fam_rows = [r for r in rows if r["family"] == family]
        if not fam_rows:
            ax.set_visible(False)
            continue
        # Backdrop: pooled within/across across all configs in family.
        all_within = np.concatenate([
            np.asarray(r.get("within_samples", []), dtype=float)
            for r in fam_rows
        ]) if any(r.get("within_samples") for r in fam_rows) else np.array([])
        all_across = np.concatenate([
            np.asarray(r.get("across_samples", []), dtype=float)
            for r in fam_rows
        ]) if any(r.get("across_samples") for r in fam_rows) else np.array([])
        if all_within.size == 0 and all_across.size == 0:
            ax.set_visible(False)
            continue
        all_vals = np.concatenate(
            [a for a in (all_within, all_across) if a.size > 0]
        )
        x_max = float(np.quantile(all_vals, 0.99) * 1.05) if all_vals.size else 1.0
        bins = np.linspace(0, x_max, 30)
        if all_within.size > 0:
            ax.hist(
                all_within, bins=bins, alpha=0.35, color="#2c7fb8",
                density=True, label=f"within (all configs, n={all_within.size})",
            )
        if all_across.size > 0:
            ax.hist(
                all_across, bins=bins, alpha=0.35, color="#d95f0e",
                density=True, label=f"across (all configs, n={all_across.size})",
            )
        # Highlight best config: dashed vertical lines at its medians.
        best = best_per_family.get(family)
        if best is not None:
            w_med = best.get("within_median")
            a_med = best.get("across_median")
            if w_med is not None and not math.isnan(w_med):
                ax.axvline(
                    w_med, color="#2c7fb8", ls="--", lw=2.0,
                    label=f"★ {best['config']} within median = {w_med:.3g}",
                )
            if a_med is not None and not math.isnan(a_med):
                ax.axvline(
                    a_med, color="#d95f0e", ls="--", lw=2.0,
                    label=f"★ {best['config']} across median = {a_med:.3g}",
                )
        ax.set_xlabel("MMD (RBF, root-stripped qpos)")
        ax.set_ylabel("density")
        ax.set_title(family_title.get(family, family), fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Per-family purity breakdown: within-syllable vs across-syllable MMD",
        y=1.01,
    )
    fig.tight_layout()
    png = out_dir / "purity_breakdown.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Plot 4: Code-duration heatmap (rows = configs, cols = syllable rank,
# color = median bout duration in ms). Timescale is a free parameter
# under the new selection rule, but visualizing per-config durations is
# still useful context.
# =============================================================================


def _per_syllable_median_durations_ms(
    codes: np.ndarray, K: int, fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(durations_ms, frame_counts)``, each shape ``(K,)``.

    ``durations_ms[c]`` is the median bout duration of syllable ``c`` in
    milliseconds (NaN if the syllable never appears). ``frame_counts[c]``
    is the syllable's total frame count, used as the row-sort key.
    """
    codes = np.asarray(codes, dtype=np.int32)
    n_clips, n_frames = codes.shape
    per_syl: list[list[int]] = [[] for _ in range(K)]
    counts = np.zeros(K, dtype=np.int64)
    ms_per_frame = 1000.0 / float(fps)
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
            length = int(e - s)
            per_syl[v].append(length)
            counts[v] += length
    durations = np.full(K, np.nan, dtype=np.float64)
    for c in range(K):
        if per_syl[c]:
            durations[c] = float(np.median(per_syl[c])) * ms_per_frame
    return durations, counts


def plot_duration_heatmap(
    family: str,
    configs: list[FamilyConfig],
    out_dir: Path,
    fps: float,
    best_label: str | None = None,
) -> None:
    """For one family: heatmap of per-syllable median bout duration (ms).

    Rows = configs, cols = syllable rank (descending frame usage within
    each row, so the leftmost column is the dominant syllable of that
    row). For KPMS we split rows by model_type into ARHMM and SLDS
    side-by-side panels, mirroring the EML curve.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if not configs:
        log.info("No %s configs; skipping duration heatmap.", family)
        return

    # Split KPMS by model_type, else single panel.
    panel_groups: list[tuple[str, list[FamilyConfig]]] = []
    if family == "kpms":
        by_mt: dict[str, list[FamilyConfig]] = {}
        for fc in configs:
            mt = str(((fc.source_info or {}).get("model_type")) or "arhmm")
            by_mt.setdefault(mt, []).append(fc)
        for mt in sorted(by_mt.keys()):
            panel_groups.append((mt, by_mt[mt]))
    else:
        panel_groups = [("", configs)]

    # Compute per-config durations + usage-sort.
    panel_data: list[tuple[str, list[FamilyConfig], list[np.ndarray]]] = []
    max_active = 1
    for panel_tag, panel_configs in panel_groups:
        rows_data: list[np.ndarray] = []
        for fc in panel_configs:
            durations, counts = _per_syllable_median_durations_ms(
                fc.codes, fc.num_codes, fps,
            )
            order = np.argsort(-counts)  # descending usage
            rows_data.append(durations[order])
            # active threshold: at least 1% of total frames
            total = counts.sum()
            if total > 0:
                active = int((counts >= total * 0.01).sum())
            else:
                active = 0
            if active > max_active:
                max_active = active
        panel_data.append((panel_tag, panel_configs, rows_data))

    # Choose column width: family-wide max active syllables, capped to a
    # readable bound (large K=1024 PS-VQ-VAE would otherwise dominate).
    max_K_family = max(c.num_codes for c in configs)
    max_K = max(min(max_active, max_K_family), 1)

    n_panels = len(panel_data)
    n_rows_total = max(len(g[1]) for g in panel_data)
    per_panel_w = float(np.clip(0.18 * max_K + 3.5, 7, 22))
    fig_w = per_panel_w * n_panels
    fig_h = float(max(3, 0.32 * n_rows_total + 1))
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"wspace": 0.6},
    )
    if n_panels == 1:
        axes = [axes]

    family_title = {
        "kpms": "KPMS", "vae": "VAE",
        "vqvae": "VQ-VAE", "ps_vqvae": "PS-VQ-VAE",
    }.get(family, family)

    # Shared color scale across panels for readability.
    all_finite = np.concatenate([
        row[~np.isnan(row)]
        for _, _, rows in panel_data for row in rows
    ])
    vmin = float(max(all_finite.min(), 1.0)) if all_finite.size > 0 else 1.0
    vmax = float(all_finite.max()) if all_finite.size > 0 else 1000.0
    norm = LogNorm(vmin=max(vmin, 1.0), vmax=max(vmax, vmin * 1.01))

    last_im = None
    for ax, (panel_tag, panel_configs, rows_data) in zip(axes, panel_data):
        n_rows = len(panel_configs)
        mat = np.full((n_rows, max_K), np.nan, dtype=np.float64)
        row_labels: list[str] = []
        for i, (fc, row) in enumerate(zip(panel_configs, rows_data)):
            n_cols_row = min(row.shape[0], max_K)
            mat[i, :n_cols_row] = row[:n_cols_row]
            row_labels.append(fc.config_label)

        last_im = ax.imshow(
            mat, aspect="auto", cmap="magma",
            interpolation="nearest", origin="upper", norm=norm,
        )
        ax.set_xlabel("syllable rank (descending frame usage)")
        if ax is axes[0]:
            ax.set_ylabel(f"{family} config")
        ax.set_yticks(np.arange(n_rows))
        decorated_labels = []
        for lbl in row_labels:
            if best_label is not None and lbl == best_label:
                decorated_labels.append(f"★ {lbl}")
            else:
                decorated_labels.append(lbl)
        ax.set_yticklabels(decorated_labels, fontsize=8)
        if best_label is not None:
            for i, lbl in enumerate(row_labels):
                if lbl == best_label:
                    ax.add_patch(plt.Rectangle(
                        (-0.5, i - 0.5), max_K, 1.0,
                        fill=False, edgecolor="#e74c3c", lw=1.5,
                    ))
                    break
        ax.set_xticks(np.arange(0, max_K, max(max_K // 10, 1)))
        if panel_tag:
            ax.set_title(panel_tag.upper())

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label("median bout duration (ms, log scale)")

    fig.suptitle(
        f"{family_title} — per-syllable median bout duration "
        f"(rows: configs; cols: usage rank)",
        y=1.02,
    )
    png = out_dir / f"duration_heatmap_{family}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Hyperparameter parsers (per family) — used by the neural hparam curves.
# =============================================================================


def _parse_vae_label(label: str) -> tuple[float, int] | None:
    """``kl_0p001_k10`` → (0.001, 10). ``k=30 (default)`` → None."""
    import re

    m = re.match(r"^kl_([0-9p]+)_k(\d+)$", label)
    if not m:
        return None
    kl = float(m.group(1).replace("p", "."))
    K = int(m.group(2))
    return (kl, K)


def _parse_vqvae_label(label: str) -> tuple[int, float] | None:
    """``k8_commit0p1`` → (8, 0.1). ``k=30 (default)`` → None."""
    import re

    m = re.match(r"^k(\d+)_commit([0-9p]+)$", label)
    if not m:
        return None
    K = int(m.group(1))
    commit = float(m.group(2).replace("p", "."))
    return (K, commit)


def _parse_ps_vqvae_label(label: str) -> dict | None:
    """Returns ``{"stage": "A"|"B"|"C", ...stage_params}`` or None."""
    import re

    m = re.match(r"^stageA_k(?:_(\d+))+$", label)
    if m:
        nums = [int(s) for s in re.findall(r"\d+", label[len("stageA_k_"):])]
        return {"stage": "A", "group_sizes": tuple(nums),
                "K_total": int(np.prod(nums))}
    m = re.match(r"^stageB_decay([0-9p]+)_rt([0-9p]+)$", label)
    if m:
        decay = float(m.group(1).replace("p", "."))
        rt = float(m.group(2).replace("p", "."))
        return {"stage": "B", "decay": decay, "rt": rt}
    m = re.match(r"^stageC_commit([0-9p]+)$", label)
    if m:
        commit = float(m.group(1).replace("p", "."))
        return {"stage": "C", "commit": commit}
    return None


def plot_neural_hparam_curves(
    rows: list[dict], out_dir: Path, best_per_family: dict[str, dict],
) -> None:
    """Per-family structured curves: primary hyperparameter on x, secondary
    as separate lines. One PNG per neural family with two side-by-side
    panels — left = model fit, right = purity score.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _draw_two_panel(
        family: str, x_label: str, x_is_log: bool,
        groups: dict, out_path: Path,
    ) -> None:
        """``groups[line_label] = list[(x_value, fit, purity, config_label)]``."""
        if not groups:
            return
        fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2))
        color_keys = sorted(groups.keys(), key=lambda k: (
            float(k) if isinstance(k, (int, float)) else str(k)
        ))
        colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(color_keys), 1)))
        best_label = best_per_family.get(family, {}).get("config")
        for color, key in zip(colors, color_keys):
            pts = sorted(groups[key], key=lambda t: t[0])
            xs = [t[0] for t in pts]
            fits = [t[1] for t in pts]
            purs = [t[2] for t in pts]
            cfgs = [t[3] for t in pts]
            label = str(key)
            ax_f.plot(xs, fits, marker="o", color=color, label=label, linewidth=1.4)
            ax_p.plot(xs, purs, marker="o", color=color, label=label, linewidth=1.4)
            # Star the best across both axes.
            for x, fit_v, pur_v, cfg in zip(xs, fits, purs, cfgs):
                if cfg == best_label:
                    if fit_v is not None and not math.isnan(fit_v):
                        ax_f.plot(
                            [x], [fit_v], marker="*", markersize=14,
                            color="#e74c3c", markeredgecolor="black",
                            linewidth=0, zorder=5,
                        )
                    if pur_v is not None and not math.isnan(pur_v):
                        ax_p.plot(
                            [x], [pur_v], marker="*", markersize=14,
                            color="#e74c3c", markeredgecolor="black",
                            linewidth=0, zorder=5,
                        )
        for ax, ylabel in [
            (ax_f, "model fit (higher = better)"),
            (ax_p, "purity = median(across) / median(within)"),
        ]:
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            if x_is_log:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        family_title = {
            "vae": "VAE", "vqvae": "VQ-VAE", "ps_vqvae": "PS-VQ-VAE",
        }.get(family, family)
        fig.suptitle(
            f"{family_title} — hyperparameter sweep "
            f"(left: model fit; right: purity)",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_path)

    # ---- VAE: x = kl, lines = K ----
    vae_groups: dict[int, list] = {}
    for r in rows:
        if r["family"] != "vae":
            continue
        parsed = _parse_vae_label(r["config"])
        if parsed is None:
            continue
        kl, K = parsed
        vae_groups.setdefault(K, []).append(
            (kl, r.get("model_fit"), r.get("purity_score"), r["config"])
        )
    _draw_two_panel(
        "vae", "KL weight", True, vae_groups,
        out_dir / "vae_hparam_curve.png",
    )

    # ---- VQ-VAE: x = commit, lines = K ----
    vqvae_groups: dict[int, list] = {}
    for r in rows:
        if r["family"] != "vqvae":
            continue
        parsed = _parse_vqvae_label(r["config"])
        if parsed is None:
            continue
        K, commit = parsed
        vqvae_groups.setdefault(K, []).append(
            (commit, r.get("model_fit"), r.get("purity_score"), r["config"])
        )
    _draw_two_panel(
        "vqvae", "commit weight", False, vqvae_groups,
        out_dir / "vqvae_hparam_curve.png",
    )

    # ---- PS-VQ-VAE: 3-panel split by stage ----
    # Stage A varies arch (group_codebook_sizes), Stage B varies dynamics
    # (decay × reset_threshold) on the winning Stage A arch, Stage C
    # varies commit weight on the Stage B winner. Three panels per metric.
    by_stage: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    for r in rows:
        if r["family"] != "ps_vqvae":
            continue
        parsed = _parse_ps_vqvae_label(r["config"])
        if parsed is None:
            continue
        by_stage[parsed["stage"]].append({"parsed": parsed, "row": r})
    if any(by_stage.values()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), squeeze=False)
        best_label = best_per_family.get("ps_vqvae", {}).get("config")

        def _draw_ps_panel(ax_f, ax_p, stage_entries: list[dict],
                           x_key: str, x_label: str, title: str,
                           x_is_log: bool = False, x_is_categorical: bool = False):
            if not stage_entries:
                ax_f.set_visible(False)
                ax_p.set_visible(False)
                return
            if stage_entries[0]["parsed"]["stage"] == "B":
                # X = decay; one line per reset_threshold (rt).
                rt_groups: dict[float, list] = {}
                for e in stage_entries:
                    p = e["parsed"]
                    r = e["row"]
                    rt_groups.setdefault(p["rt"], []).append(
                        (p["decay"], r.get("model_fit"), r.get("purity_score"),
                         r["config"])
                    )
                rt_keys = sorted(rt_groups.keys())
                colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(rt_keys), 1)))
                for color, rt in zip(colors, rt_keys):
                    pts = sorted(rt_groups[rt], key=lambda t: t[0])
                    xs = [t[0] for t in pts]
                    fits = [t[1] for t in pts]
                    purs = [t[2] for t in pts]
                    cfgs = [t[3] for t in pts]
                    ax_f.plot(xs, fits, marker="o", color=color,
                              label=f"rt={rt}", linewidth=1.4)
                    ax_p.plot(xs, purs, marker="o", color=color,
                              label=f"rt={rt}", linewidth=1.4)
                    for x, fit_v, pur_v, cfg in zip(xs, fits, purs, cfgs):
                        if cfg == best_label:
                            if fit_v is not None:
                                ax_f.plot([x], [fit_v], marker="*",
                                          markersize=14, color="#e74c3c",
                                          markeredgecolor="black",
                                          linewidth=0, zorder=5)
                            if pur_v is not None:
                                ax_p.plot([x], [pur_v], marker="*",
                                          markersize=14, color="#e74c3c",
                                          markeredgecolor="black",
                                          linewidth=0, zorder=5)
            else:
                # Single line, x is the requested key.
                pts = sorted(
                    [(e["parsed"][x_key]
                      if not x_is_categorical else str(e["parsed"]["group_sizes"]),
                      e["row"].get("model_fit"),
                      e["row"].get("purity_score"),
                      e["row"]["config"])
                     for e in stage_entries],
                    key=lambda t: (t[0] if not x_is_categorical
                                   else str(t[0])),
                )
                if x_is_categorical:
                    xs_str = [t[0] for t in pts]
                    xs = list(range(len(xs_str)))
                    ax_f.set_xticks(xs); ax_f.set_xticklabels(xs_str, rotation=30, ha="right", fontsize=8)
                    ax_p.set_xticks(xs); ax_p.set_xticklabels(xs_str, rotation=30, ha="right", fontsize=8)
                else:
                    xs = [t[0] for t in pts]
                fits = [t[1] for t in pts]
                purs = [t[2] for t in pts]
                cfgs = [t[3] for t in pts]
                ax_f.plot(xs, fits, marker="o", color="#3b82c4", linewidth=1.4)
                ax_p.plot(xs, purs, marker="o", color="#3b82c4", linewidth=1.4)
                for x, fit_v, pur_v, cfg in zip(xs, fits, purs, cfgs):
                    if cfg == best_label:
                        if fit_v is not None:
                            ax_f.plot([x], [fit_v], marker="*",
                                      markersize=14, color="#e74c3c",
                                      markeredgecolor="black",
                                      linewidth=0, zorder=5)
                        if pur_v is not None:
                            ax_p.plot([x], [pur_v], marker="*",
                                      markersize=14, color="#e74c3c",
                                      markeredgecolor="black",
                                      linewidth=0, zorder=5)
            for ax in (ax_f, ax_p):
                ax.set_xlabel(x_label)
                if x_is_log:
                    ax.set_xscale("log")
                ax.grid(True, alpha=0.3)
            ax_f.set_ylabel("model fit (higher = better)")
            ax_p.set_ylabel("purity score")
            ax_f.set_title(f"{title} — fit", fontsize=10)
            ax_p.set_title(f"{title} — purity", fontsize=10)
            if ax_f.has_data() and any(line.get_label() and not line.get_label().startswith("_") for line in ax_f.lines):
                handles, labels = ax_f.get_legend_handles_labels()
                if labels:
                    ax_f.legend(fontsize=8, loc="best")
                    ax_p.legend(fontsize=8, loc="best")

        _draw_ps_panel(
            axes[0, 0], axes[1, 0], by_stage["A"],
            x_key="group_sizes", x_label="group_codebook_sizes",
            title="Stage A — architecture", x_is_categorical=True,
        )
        _draw_ps_panel(
            axes[0, 1], axes[1, 1], by_stage["B"],
            x_key="decay", x_label="EMA decay",
            title="Stage B — dynamics (lines = reset_threshold)",
        )
        _draw_ps_panel(
            axes[0, 2], axes[1, 2], by_stage["C"],
            x_key="commit", x_label="commit weight",
            title="Stage C — commit",
        )
        fig.suptitle(
            "PS-VQ-VAE — staged hyperparameter sweep "
            "(top: fit; bottom: purity)",
            y=1.0,
        )
        fig.tight_layout()
        out_path = out_dir / "ps_vqvae_hparam_curve.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_path)


# =============================================================================
# Plot 4b: Raw training loss bars for the neural families. The scatter
# uses ``model_fit = -final_loss`` so "higher = better" matches KPMS; this
# view shows the same number on its native positive scale, sorted by
# config within each family for direct comparison.
# =============================================================================


def plot_neural_loss_bars(
    rows: list[dict], out_dir: Path, best_per_family: dict[str, dict],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NEURAL = ["vae", "vqvae", "ps_vqvae"]
    family_title = {"vae": "VAE", "vqvae": "VQ-VAE", "ps_vqvae": "PS-VQ-VAE"}
    panels: list[tuple[str, list[dict]]] = []
    for fam in NEURAL:
        fam_rows = [
            r for r in rows
            if r["family"] == fam and r.get("model_fit") is not None
        ]
        if fam_rows:
            panels.append((fam, fam_rows))
    if not panels:
        log.info("No neural rows with model_fit; skipping loss bars.")
        return

    n = len(panels)
    # Width scales with the max number of configs in any one panel.
    n_max_configs = max(len(p[1]) for p in panels)
    fig_w = max(8.0, 0.35 * n_max_configs + 2.0)
    fig, axes = plt.subplots(
        n, 1, figsize=(fig_w, 2.6 * n), sharex=False, squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, (fam, fam_rows) in zip(axes_flat, panels):
        # Final positive loss = -model_fit (we inverted on the way in).
        fam_rows = sorted(fam_rows, key=lambda r: float(-r["model_fit"]))
        names = [r["config"] for r in fam_rows]
        losses = [float(-r["model_fit"]) for r in fam_rows]
        best_label = best_per_family.get(fam, {}).get("config")
        colors = [
            "#e74c3c" if name == best_label else "#3b82c4"
            for name in names
        ]
        bars = ax.bar(np.arange(len(names)), losses, color=colors, alpha=0.85)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(
            [f"★ {n}" if n == best_label else n for n in names],
            rotation=45, ha="right", fontsize=7,
        )
        ax.set_ylabel("final training loss")
        ax.set_title(
            f"{family_title.get(fam, fam)} — final training loss "
            f"(sorted asc; red = ★ best by purity)",
            fontsize=10,
        )
        ax.grid(True, axis="y", alpha=0.3)
        # Light annotation on the bars: value.
        for b, v in zip(bars, losses):
            ax.text(
                b.get_x() + b.get_width() / 2, v, f"{v:.3g}",
                ha="center", va="bottom", fontsize=6,
            )

    fig.tight_layout()
    png = out_dir / "neural_loss_bars.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Plot 5: Purity-and-fit summary table
# =============================================================================


_FAMILY_COLORS_TABLE: dict[str, str] = {
    "kpms": "#cfe2f3",
    "vae": "#d9ead3",
    "vqvae": "#fff2cc",
    "ps_vqvae": "#f4cccc",
}


def write_purity_table(
    rows: list[dict], out_dir: Path, best_per_family: dict[str, dict],
) -> None:
    """One table PNG combining all configs across all families, ranked by
    purity_score within each family. ★ marks the best.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    # Group rows by family, sort each group by purity_score desc.
    by_family: dict[str, list[dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    families = sorted(by_family.keys())
    for fam in families:
        by_family[fam].sort(
            key=lambda r: (
                -(r.get("purity_score") or -math.inf),
                -(r.get("model_fit") or -math.inf),
            )
        )

    best_set: set[tuple[str, str]] = {
        (fam, entry["config"]) for fam, entry in best_per_family.items()
    }

    headers = [
        "family", "config", "K", "active",
        "median dur (ms)", "model fit", "purity (×)", "within MMD",
        "across MMD",
    ]
    data: list[list[str]] = []
    row_colors: list[list[str]] = []
    for fam in families:
        for r in by_family[fam]:
            is_best = (fam, r["config"]) in best_set
            cfg_label = f"★ {r['config']}" if is_best else r["config"]
            fit = r.get("model_fit")
            fit_str = f"{fit:.4g}" if fit is not None else "—"
            purity = r.get("purity_score")
            purity_str = f"{purity:.2f}" if purity is not None and not math.isnan(purity) else "—"
            w = r.get("within_median")
            a = r.get("across_median")
            w_str = f"{w:.3g}" if w is not None and not math.isnan(w) else "—"
            a_str = f"{a:.3g}" if a is not None and not math.isnan(a) else "—"
            data.append([
                fam, cfg_label, str(int(r["K"])),
                str(int(r.get("n_active_at_1pct", 0) or 0)),
                f"{r.get('median_duration_ms', 0):.1f}",
                fit_str, purity_str, w_str, a_str,
            ])
            base = _FAMILY_COLORS_TABLE.get(fam, "#eeeeee")
            row_colors.append([base] * len(headers))

    n_rows = len(data)
    fig_h = max(2.5, 0.32 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=data, colLabels=headers, cellColours=row_colors,
        loc="upper left", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.15)
    ax.set_title(
        "Tokenizer configs ranked by purity (★ = best per family)",
        loc="left", fontsize=11,
    )
    png = out_dir / "purity_table.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png)


# =============================================================================
# Main
# =============================================================================


def _load_reference_qpos(cfg: DictConfig) -> np.ndarray | None:
    """Load qpos as ``(n_clips, n_frames, nq)`` from the reference h5.

    Required for purity computation. Returns None if the file or the
    expected ``qpos`` dataset is missing.
    """
    import h5py

    h5_path = cfg.purity.get("reference_h5", None) if "purity" in cfg else None
    if h5_path is None:
        h5_path = str(cfg.get("reference", {}).get("h5_path", ""))
    if not h5_path:
        return None
    h5_path = str(h5_path)
    if not Path(h5_path).exists():
        log.warning("Reference h5 not found: %s — skipping purity.", h5_path)
        return None
    n_frames = int(cfg.purity.get("n_frames_per_clip", 250)) if "purity" in cfg else 250
    with h5py.File(h5_path, "r") as f:
        if "qpos" not in f:
            log.warning("No 'qpos' dataset in %s — skipping purity.", h5_path)
            return None
        q = np.asarray(f["qpos"], dtype=np.float32)
    if q.ndim == 2:
        nq = q.shape[1]
        n_total = q.shape[0]
        if n_total % n_frames != 0:
            log.warning(
                "qpos has %d rows, not divisible by n_frames=%d.",
                n_total, n_frames,
            )
            return None
        q = q.reshape(n_total // n_frames, n_frames, nq)
    log.info("Loaded reference qpos: shape=%s", q.shape)
    return q


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="tokenizer_diagnostics",
)
def main(cfg: DictConfig) -> None:
    out_dir = Path(str(cfg.output.base_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    # --- 1. KPMS EML curve (fit metric for KPMS) ---
    eml_summary = plot_kpms_eml_curve(cfg, out_dir)
    sweep_results: dict | None = None
    try:
        with open(cfg.kpms.sweep_results_json) as f:
            sweep_results = json.load(f)
    except (OSError, AttributeError):
        sweep_results = None

    # --- 2. Collect per-family configs ---
    fps = float(cfg.neural.fps)
    family_to_configs: dict[str, list[FamilyConfig]] = {
        "kpms": collect_kpms_configs(cfg),
        "vae": collect_neural_configs(cfg.neural.vae),
        "vqvae": collect_neural_configs(cfg.neural.vqvae),
        "ps_vqvae": collect_neural_configs(cfg.neural.ps_vqvae),
    }
    total = sum(len(v) for v in family_to_configs.values())
    log.info(
        "Collected %d configs (kpms=%d, vae=%d, vqvae=%d, ps_vqvae=%d)",
        total,
        len(family_to_configs["kpms"]),
        len(family_to_configs["vae"]),
        len(family_to_configs["vqvae"]),
        len(family_to_configs["ps_vqvae"]),
    )

    # --- 3. Load reference qpos (shared kinematic space for purity) ---
    ref_qpos = _load_reference_qpos(cfg)

    # --- 4. Compute fit + purity per config ---
    strip_root = bool(cfg.purity.get("strip_root", True)) if "purity" in cfg else True
    n_across = int(cfg.purity.get("n_across_pairs", 30)) if "purity" in cfg else 30
    min_bouts = int(cfg.purity.get("min_bouts_for_within", 4)) if "purity" in cfg else 4
    seed = int(cfg.purity.get("seed", 0)) if "purity" in cfg else 0
    rng = np.random.RandomState(seed)

    rows: list[dict] = []
    summary: dict = {"eml_summary": eml_summary, "configs": []}
    for family, configs in family_to_configs.items():
        for fc in configs:
            m = compute_tokenizer_metrics(
                fc.codes, num_codes=fc.num_codes, fps=fps,
            )
            fit = extract_model_fit(family, fc, sweep_results)
            if ref_qpos is None:
                purity = {
                    "purity_score": float("nan"),
                    "within_median": float("nan"),
                    "across_median": float("nan"),
                    "within_samples": [],
                    "across_samples": [],
                    "n_active_for_purity": 0,
                    "sigma": float("nan"),
                }
            else:
                purity = compute_tokenizer_purity(
                    fc, ref_qpos,
                    strip_root=strip_root,
                    n_across_pairs=n_across,
                    min_bouts_for_within=min_bouts,
                    rng=rng,
                )
            row = {
                "family": family,
                "config": fc.config_label,
                "K": int(fc.num_codes),
                "median_duration_ms": float(m.median_duration_ms),
                "mean_duration_ms": float(m.mean_duration_ms),
                "n_active_at_1pct": int(getattr(m, "active_at_1pct", 0)),
                "model_fit": float(fit) if fit is not None else None,
                "purity_score": float(purity["purity_score"])
                if not math.isnan(purity["purity_score"]) else None,
                "within_median": float(purity["within_median"])
                if not math.isnan(purity["within_median"]) else None,
                "across_median": float(purity["across_median"])
                if not math.isnan(purity["across_median"]) else None,
                "within_samples": list(purity["within_samples"]),
                "across_samples": list(purity["across_samples"]),
                "n_active_for_purity": int(purity["n_active_for_purity"]),
                "sigma": float(purity["sigma"])
                if not math.isnan(purity["sigma"]) else None,
                "source_info": fc.source_info,
            }
            rows.append(row)
            summary["configs"].append(row)
            log.info(
                "  %-9s %s  fit=%s purity=%s  (within=%s across=%s n_active=%d)",
                family, fc.config_label,
                f"{row['model_fit']:.4g}" if row["model_fit"] is not None else "—",
                f"{row['purity_score']:.2f}" if row["purity_score"] is not None else "—",
                f"{row['within_median']:.3g}" if row["within_median"] is not None else "—",
                f"{row['across_median']:.3g}" if row["across_median"] is not None else "—",
                row["n_active_for_purity"],
            )

    # --- 5. Best-per-family: highest purity, tie-break by fit ---
    best_per_family: dict[str, dict] = {}
    for family in family_to_configs:
        candidates = [
            r for r in rows
            if r["family"] == family and r.get("purity_score") is not None
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda r: (
                r["purity_score"],
                r["model_fit"] if r["model_fit"] is not None else -math.inf,
            ),
        )
        best_per_family[family] = best
    summary["best_per_family"] = best_per_family
    log.info("Best per family (max purity, tie-break by fit):")
    for fam, b in best_per_family.items():
        log.info(
            "  %-9s  %s  (K=%d, purity=%.2f, fit=%s, dur=%.0fms)",
            fam, b["config"], int(b["K"]),
            float(b["purity_score"]),
            f"{b['model_fit']:.4g}" if b["model_fit"] is not None else "—",
            float(b["median_duration_ms"]),
        )

    # --- 6. Plots ---
    plot_fit_vs_purity_scatter(rows, out_dir, best_per_family)
    plot_purity_breakdown(rows, out_dir, best_per_family)
    plot_neural_loss_bars(rows, out_dir, best_per_family)
    plot_neural_hparam_curves(rows, out_dir, best_per_family)
    for family, configs in family_to_configs.items():
        plot_duration_heatmap(
            family, configs, out_dir, fps=fps,
            best_label=best_per_family.get(family, {}).get("config"),
        )
    write_purity_table(rows, out_dir, best_per_family)

    # --- 7. Summary JSON ---
    summary_path = out_dir / "tokenizer_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s", summary_path)

    log.info("=" * 60)
    log.info(
        "Done — %d configs across %d families.", len(rows), len(family_to_configs),
    )


if __name__ == "__main__":
    main()
