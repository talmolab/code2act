"""Categorical HMM code generator (moved from ``run_code_generation.py``).

Uses dynamax's :class:`CategoricalHMM` with EM. Sweeps over a small grid
of ``num_states`` and picks the best by validation marginal
log-likelihood. The pre-move behavior — fit on a random subset of train
clips, evaluate on the rest — is preserved; the only change is that
training and sampling are now decoupled (you fit once, save a
checkpoint, and the experiment script loads it).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax
import numpy as np

from c2a.ar_codegen.models.base import CodeGenerator, GeneratorMeta
from c2a.ar_codegen.wandb_utils import (
    code_histogram, log_code_histograms_figure, log_metrics,
)

log = logging.getLogger(__name__)


HMM_FILENAME = "hmm.npz"


class HMMGenerator(CodeGenerator):
    """dynamax CategoricalHMM with an EM sweep over ``num_states``.

    Configuration keys read from ``cfg`` (passed to :meth:`fit`):

    * ``num_states_sweep``: list of K to try (e.g. ``[4, 8, 16]``).
    * ``num_em_iters``: EM iterations per (K, clip) fit.
    * ``train_ratio``: fraction of clips used for fitting; remainder is
      held out for the best-K selection. Defaults to 0.8.
    * ``seed``: PRNG seed.
    """

    MODEL_TYPE = "hmm"

    def __init__(self, num_codes: int):
        super().__init__(num_codes=num_codes)
        self._best_params: Any = None
        self._best_K: int | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_codes: np.ndarray,
        val_codes: np.ndarray,
        cfg: Any,
        seed: int = 0,
        wandb_run: Any = None,
    ) -> None:
        # Lazy import: dynamax pulls in tensorflow_probability, which
        # we don't want to import at package load time.
        from dynamax.hidden_markov_model import CategoricalHMM

        num_states_sweep = list(cfg.get("num_states_sweep", [4, 8, 16]))
        num_em_iters = int(cfg.get("num_em_iters", 50))
        train_ratio = float(cfg.get("train_ratio", 0.8))
        seed = int(cfg.get("seed", seed))

        # Optional override: combine train + val for the HMM since the
        # outer split is already balanced and we re-split internally to
        # pick best-K. Keep them separate by default to mirror the
        # original behavior in run_code_generation.py.
        log.info(
            "HMM fit: codes %s, vocab=%d, K_sweep=%s, em_iters=%d, "
            "train_ratio=%.2f",
            train_codes.shape, self.num_codes, num_states_sweep,
            num_em_iters, train_ratio,
        )

        # Internal train/val split inside the *train* set — preserves the
        # original ``run_code_generation.py`` behavior (which only ever
        # saw ``all_codes``; we get a more honest val signal by separately
        # holding out the outer val set too, but the K-selection is on
        # the internal val to keep the contract).
        n_clips = train_codes.shape[0]
        if n_clips < 2:
            raise RuntimeError(
                f"HMM fit needs at least 2 train clips; got {n_clips}."
            )
        n_train = max(1, int(n_clips * train_ratio))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_clips)
        inner_train = train_codes[perm[:n_train]]
        inner_val = train_codes[perm[n_train:]]
        if inner_val.shape[0] == 0:
            # Fall back to outer val if internal split left val empty.
            inner_val = val_codes

        emissions_train = inner_train[:, :, None].astype(np.int32)
        emissions_val = inner_val[:, :, None].astype(np.int32) if inner_val.size > 0 else None

        best_ll = -np.inf
        best_params = None
        best_K = None
        global_step = 0

        for K in num_states_sweep:
            log.info(f"  Fitting K={K}...")
            hmm = CategoricalHMM(
                num_states=int(K), emission_dim=1, num_classes=self.num_codes,
            )
            # Fold K into the init key so each candidate gets its own
            # random initialization. A shared key biases the K-sweep by
            # giving every model the same starting point in posterior
            # space.
            key = jax.random.fold_in(jax.random.PRNGKey(seed), int(K))
            params, props = hmm.initialize(key)

            for ep in range(emissions_train.shape[0]):
                try:
                    params, lls = hmm.fit_em(
                        params, props, emissions_train[ep],
                        num_iters=num_em_iters,
                    )
                except Exception as e:
                    log.warning(f"    EM failed for clip {ep}: {e}")
                    continue
                # Per-clip end-of-EM LL is the last value in lls.
                last_ll = float(lls[-1]) if hasattr(lls, "__len__") else float(lls)
                log_metrics(
                    wandb_run, global_step,
                    {f"K{K}_last_em_ll": last_ll}, group="em",
                )
                global_step += 1

            # Evaluate on val
            if emissions_val is not None:
                val_ll = 0.0
                for ep in range(emissions_val.shape[0]):
                    try:
                        ll = hmm.marginal_log_prob(params, emissions_val[ep])
                        val_ll += float(ll)
                    except Exception:
                        pass
            else:
                val_ll = -np.inf  # no val available; can't select

            log.info(f"    K={K}: val_ll={val_ll:.1f}")
            log_metrics(
                wandb_run, global_step,
                {f"K{K}_val_ll": val_ll}, group="em",
            )
            if val_ll > best_ll:
                best_ll = val_ll
                best_params = params
                best_K = int(K)

        if best_params is None:
            raise RuntimeError(
                "HMM fit produced no valid model across the K sweep. "
                "Check the data shape and num_em_iters."
            )

        self._best_params = best_params
        self._best_K = best_K
        log.info(f"HMM fit done: best K={best_K} (val_ll={best_ll:.1f})")

        # Histogram logging — sample sequences from the fitted HMM and
        # compare their marginal code distribution to the actual splits.
        # Unlike the neural models we don't do teacher-forced argmax
        # because dynamax's CategoricalHMM doesn't expose a clean
        # per-position posterior predictive without running a smoother
        # over each clip; sampling matches how the experiment uses it.
        try:
            actual_train_hist = code_histogram(train_codes, self.num_codes)
            actual_val_hist = (
                code_histogram(val_codes, self.num_codes)
                if val_codes.size > 0 else None
            )
            n_samples_for_hist = max(
                train_codes.shape[0], val_codes.shape[0] if val_codes.size > 0 else 0,
            )
            sampled = self.sample(
                n_seq=max(n_samples_for_hist, 32),
                length=int(train_codes.shape[1]),
                seed=seed + 9999,
            )
            sampled_hist = code_histogram(sampled, self.num_codes)
            log_code_histograms_figure(
                wandb_run, global_step,
                {"actual": actual_train_hist, "sampled": sampled_hist},
                self.num_codes, title="train_code_dist",
            )
            if actual_val_hist is not None:
                log_code_histograms_figure(
                    wandb_run, global_step,
                    {"actual": actual_val_hist, "sampled": sampled_hist},
                    self.num_codes, title="val_code_dist",
                )
        except Exception as e:
            log.warning("Failed to log HMM code histograms: %s", e)

        # Provenance for the meta file.
        self.meta = GeneratorMeta(
            model_type=self.MODEL_TYPE,
            num_codes=self.num_codes,
            code_source=str(cfg.get("_code_source", "unknown")),
            n_frames_per_clip=int(train_codes.shape[1]),
            config_snapshot={
                "num_states_sweep": num_states_sweep,
                "num_em_iters": num_em_iters,
                "train_ratio": train_ratio,
                "seed": seed,
            },
            provenance={
                "best_K": int(best_K),
                "best_val_ll": float(best_ll),
            },
        )

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(
        self,
        n_seq: int,
        length: int,
        seed: int = 0,
        temperature: float = 1.0,
    ) -> np.ndarray:
        from dynamax.hidden_markov_model import CategoricalHMM

        if self._best_params is None or self._best_K is None:
            raise RuntimeError("HMMGenerator.sample called before fit/load.")
        if temperature != 1.0:
            log.warning(
                "HMMGenerator ignores temperature (no logits); sampling at T=1."
            )

        hmm = CategoricalHMM(
            num_states=self._best_K, emission_dim=1, num_classes=self.num_codes,
        )
        seqs = np.zeros((n_seq, length), dtype=np.int32)
        for i in range(n_seq):
            key = jax.random.PRNGKey(int(seed) + 1000 + i)
            _states, emissions = hmm.sample(
                self._best_params, key, num_timesteps=int(length),
            )
            seqs[i] = np.array(emissions[:, 0], dtype=np.int32)
        return seqs

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def _save_model_files(self, path: Path) -> None:
        # Flatten the dynamax ParamsCategoricalHMM into plain arrays.
        # dynamax params are nested NamedTuples; we serialize the leaves
        # via the canonical jax tree util so reload is robust.
        from jax import tree_util

        leaves, treedef = tree_util.tree_flatten(self._best_params)
        leaves_np = [np.asarray(leaf) for leaf in leaves]

        # Save the leaves as an .npz; pickle the treedef shape separately.
        # treedef is not picklable across dynamax versions, so we rebuild
        # it on load by re-initializing a fresh HMM and copying leaves in.
        np.savez_compressed(
            path / HMM_FILENAME,
            num_codes=self.num_codes,
            best_K=self._best_K,
            **{f"leaf_{i}": x for i, x in enumerate(leaves_np)},
            n_leaves=np.int32(len(leaves_np)),
        )

    @classmethod
    def _load_model_files(cls, path: Path, meta: GeneratorMeta) -> "HMMGenerator":
        import jax.numpy as jnp
        from dynamax.hidden_markov_model import CategoricalHMM
        from jax import tree_util

        data = np.load(str(path / HMM_FILENAME))
        num_codes = int(data["num_codes"])
        best_K = int(data["best_K"])
        n_leaves = int(data["n_leaves"])
        # IMPORTANT: load as jnp, not np. dynamax's hmm.sample() runs
        # under jax.lax.scan; numpy leaves cannot be indexed by traced
        # ints in there (TracerArrayConversionError). The fresh init
        # would have produced jnp leaves, so the loaded params must be
        # jnp to match.
        leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(n_leaves)]

        # Rebuild a fresh params pytree of the right structure, then swap
        # the leaves in. The structure is deterministic given (K, vocab).
        hmm = CategoricalHMM(
            num_states=best_K, emission_dim=1, num_classes=num_codes,
        )
        fresh_params, _ = hmm.initialize(jax.random.PRNGKey(0))
        _, treedef = tree_util.tree_flatten(fresh_params)
        # Match leaf counts before reconstructing — a mismatch here would
        # silently produce wrong samples.
        fresh_leaves, _ = tree_util.tree_flatten(fresh_params)
        if len(fresh_leaves) != len(leaves):
            raise RuntimeError(
                f"HMM checkpoint leaf count {len(leaves)} != current "
                f"dynamax structure {len(fresh_leaves)} for K={best_K}, "
                f"vocab={num_codes}. dynamax version may have changed."
            )
        params = tree_util.tree_unflatten(treedef, leaves)

        gen = cls(num_codes=num_codes)
        gen._best_params = params
        gen._best_K = best_K
        return gen
