# Repository Structure

Single source of truth for the code2act repo layout. `README.md` keeps a
short pointer here so the project description stays focused on what the
code does, not where it lives.

## Top level

```
code2act/
├── README.md                  # what + install + quickstart
├── pyproject.toml             # package "c2a"; py 3.11 / 3.12
├── docs/
│   ├── codebase_structure.md  # THIS file
│   ├── run_c2a.md             # single-run + sweep + tokenize + decoder
│   ├── moseq_decoder_config.md# every yaml key explained
│   └── experiments.md         # 18 analysis scripts
├── bash_scripts/              # pipeline orchestration (grouped into subfolders)
│   ├── README.md  _pyrun.py
│   ├── pipeline/              # run_kpms_effect_sweep.sh, run_vae_effect_sweep.sh
│   ├── sweeps/                # KPMS / VAE tokenizer sweeps
│   └── training/              # kpms_baseline, c2c_additive, designed_segmenter
├── c2a/                       # pip-installable package — see below
├── Code2Act-website/          # Astro site (★ gitignored sibling)
├── track-mjx/                 # ★ user-cloned sibling, gitignored
├── .venv/                     # ★ user-created, gitignored
├── data/rodent/               # ★ H5 reference clips, gitignored
├── trained_mimic_ckpts/       # ★ distillation-target ckpts, gitignored
├── trained_kinematics_vae_ckpts/  # ★ kinematics-VAE ckpts, gitignored
└── model_checkpoints/         # ★ decoder RL ckpts, gitignored
```

★ = gitignored. The user populates `track-mjx/` by `git clone`, `.venv/`
by `uv venv`, `data/rodent/` from HuggingFace.

## The `c2a/` package

```
c2a/
├── __init__.py
├── paths.py                   # get_repo_root() / get_c2a_root() / get_asset_path()
├── assets/rodent.xml          # vendored correct rodent MJCF (NOT vnl_playground's)
├── rendering_utils.py         # EGL-backed video helpers (vendored from track-mjx vqvae_jax)
├── render/
│   └── topk_bands_render.py   # variant-agnostic top-K band renderer
│                              #   shared by kpms + kinematics_vae
│
├── codegen/                   # UNIFIED behavioral-code generation
│   ├── __init__.py
│   ├── run_sweep.py           # sweep dispatcher (routes by model: kpms)
│   ├── code_sources.py        # ★ chokepoint loader: CodesPayload + VALID_SOURCES
│   │                          #   {kpms_raw, kpms_merged,
│   │                          #    kinematics_vae, gt_directional_turn}
│   ├── shared/
│   │   ├── label_alignment.py # rodent_labels.csv loader; behavior+direction tracks
│   │   ├── coverage_metrics.py# NMI, Hungarian-purity
│   │   └── npz_schema.py      # ★ write_codes_npz — single writer, validates
│   │                          #   shape/dtype/range AND all_codes sorted-union order
│   ├── kpms/                  # Keypoint-MoSeq tokenizer
│   │   ├── config.py
│   │   ├── fit_kpms.py
│   │   ├── keypoint_loader.py
│   │   ├── sweep_impl.py      # Weinreb-2024 EML-ranked sweep
│   │   ├── merge_states.py    # AR-HMM Bhattacharyya auto-merge
│   │   ├── native_viz.py      # (xy/xz × syllable) trajectory grid
│   │   ├── render_syllable_grids.py
│   │   ├── render_topk_bands.py
│   │   ├── generate_codes.py  # best fit → CodesPayload .npz
│   │   └── generate_all_codes.py # per-(K,κ) .npz files + manifest
│   └── kinematics_vae/        # kinematics-feature VAE family
│       ├── backbone/          # Scott v15 — shared encoder/decoder/predictor
│       │   ├── frame_features.py    # m_t = (p, θ, q_active); 3-way body split
│       │   ├── window_dataset.py    # make_windows
│       │   ├── normalizer.py        # per-feature z-score
│       │   ├── mlp.py               # Dense → ELU → LayerNorm block
│       │   ├── encoder.py           # MotionEncoderConv1DSharedSplit
│       │   ├── decoder.py           # MotionDecoderSplit
│       │   ├── predictor.py         # MotionPredictorSplit (aux dynamics loss)
│       │   ├── losses.py            # KL, masked MSE, σ-cap, pretrain_loss
│       │   ├── train_loop.py        # AdamW + warmup-hold-cosine + ckpt
│       │   └── codegen.py           # encode_balanced_clips + load_checkpoint
│       ├── vae/                     # VAE + post-hoc K-means
│       │   ├── bottleneck.py        # GaussianBottleneck (KL + σ-cap)
│       │   ├── train.py             # Hydra entry
│       │   ├── tokenize.py          # PCA → KMeans → codes npz
│       │   └── configs/pretrain.yaml
│       └── viz/
│           ├── projections.py       # PCA / t-SNE / UMAP wrappers
│           └── plot_latent.py       # 3-cols × N-rows latent grid
│
├── configs/                   # Hydra YAMLs (decoder + KPMS sweeps)
│   ├── moseq_decoder.yaml
│   ├── kpms_sweep.yaml
│   ├── kpms_sweep_smoke.yaml
│   ├── mimic_encoder.yaml
│   ├── inference.yaml
│   └── kpms_effect_sweep.yaml
│
├── ar_codegen/                # AR sequence-model code generators (HMM/GPT2-xs/Mamba)
├── experiments/               # evaluation scripts in 6 theme subfolders:
│                              #   tokenizer/ behavior/ fidelity/ dynamics/
│                              #   code_grid/ ensemble_manifold/ — each with its own
│                              #   configs/. Invoke c2a.experiments.<theme>.<script>
│                              #   experiments/shared/: code_labels.py
│                              #   (per-code majority-label helpers) +
│                              #   behavior_mmd.py (MMD/KID + GT-segment
│                              #   helpers, mirrors TopoVNL feature_analysis/mmd_kid)
├── figures/                   # ~20 publication-figure scripts
├── tests/                     # pytest suite (load-bearing; see CLAUDE.md)
│
├── moseq_env_wrapper.py       # MoSeqImitation(Imitation) — injects kpms_code obs
├── moseq_decoder_network.py   # encoder–GRU–decoder policy
├── moseq_losses.py            # PPO + KL + distillation + code metrics
├── moseq_ppo.py               # PPO wrapper (monkey-patches ff_ppo)
├── moseq_ppo_networks.py
├── train_mimic_encoder.py     # pretrain the distillation target VAE
└── train_moseq_decoder.py     # main entry — Hydra
```

## Key non-obvious invariants

- **`c2a/codegen/code_sources.py::load_codes`** is the only entry point
  every downstream consumer (env wrapper, decoder, RL loop, experiments)
  uses for behavioral codes. Adding a new tokenizer family means adding a
  new `SOURCE_*` constant and a dispatch branch — nothing else changes.
- **`c2a/codegen/shared/npz_schema.py::write_codes_npz`** is the only
  writer every tokenizer's codegen must call. It validates the schema
  before disk write and enforces `all_codes` sorted-union ordering when
  `all_indices` is provided — downstream experiments index `all_codes`
  by sorted-union position, so a wrong-order array would silently
  misalign every cross-tokenizer plot.
- **`c2a/codegen/kinematics_vae/backbone/`** is a Scott-v15 verbatim
  port. The encoder / decoder / predictor / train_loop / codegen are
  shared; the VAE supplies its own Gaussian bottleneck.

## Output locations (all gitignored)

| Output | Path | Producer |
|---|---|---|
| KPMS sweep | `c2a/outputs/kpms_sweep/` | `c2a.codegen.run_sweep` (model: kpms) |
| KPMS codes | `c2a/outputs/kpms_sweep/best_codes.npz` | `c2a.codegen.kpms.generate_codes` |
| VAE ckpts | `trained_kinematics_vae_ckpts/vae/step_*` | `c2a.codegen.kinematics_vae.vae.train` |
| VAE codes | `trained_kinematics_vae_ckpts/vae/best_codes.npz` | `c2a.codegen.kinematics_vae.vae.tokenize` |
| MIMIC encoder | `trained_mimic_ckpts/` | `c2a.train_mimic_encoder` |
| Decoder RL | `model_checkpoints/` | `c2a.train_moseq_decoder` |
| Figures | `outputs/figures/<figure_name>/` | every `c2a.figures.*` script (one subdir per script) |
| Experiments | `outputs/<theme>/<exp_name>/` | every `c2a.experiments.<theme>.*` script |
| WandB | `wandb/` (or `c2a/wandb/`) | every training script |

`outputs/` mirrors the code layout: experiment outputs live under their theme
(`outputs/<theme>/<exp_name>/`, matching `c2a/experiments/<theme>/`) and figure
render outputs are grouped under `outputs/figures/<figure_name>/` (matching
`c2a/figures/`). The final publication figures are copied into `paper/figures/`.

## See also

- `CLAUDE.md` — project conventions, code style, JAX/Flax patterns, pitfalls.
- `docs/run_c2a.md` — how to run each stage of the pipeline.
- `docs/moseq_decoder_config.md` — every Hydra key in `moseq_decoder.yaml`.
- `docs/experiments.md` — the 18 analysis scripts.
