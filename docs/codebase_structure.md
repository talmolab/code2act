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
│   └── experiments.md         # 13 analysis scripts
├── bash_scripts/              # pipeline orchestration
│   ├── README.md
│   ├── run_pipeline.sh                    # KPMS × decoder grid
│   ├── run_pipeline_best_kpms.sh          # fixed-KPMS decoder sweep
│   └── run_pipeline_kinematics_vae.sh     # VAE/VQVAE/PS-VQVAE × decoder
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
│                              #   shared by kpms + shimoseq + (future) kinematics_vae
│
├── codegen/                   # UNIFIED behavioral-code generation
│   ├── __init__.py
│   ├── run_sweep.py           # sweep dispatcher (routes by model: kpms | shimoseq)
│   ├── code_sources.py        # ★ chokepoint loader: CodesPayload + VALID_SOURCES
│   │                          #   {kpms_raw, kpms_merged, shimoseq,
│   │                          #    kinematics_vae, vqvae, ps_vqvae,
│   │                          #    gt_directional_turn}
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
│   ├── shimoseq/              # shMoSeq high-level state tokenizer
│   │   ├── config.py
│   │   ├── syllable_loader.py
│   │   ├── fit_shimoseq.py
│   │   ├── sweep_impl.py      # Weinreb-2026 lagged-MI + Rand-score sweep
│   │   ├── render_topk_bands.py
│   │   └── generate_codes.py  # best fit → CodesPayload .npz (new in PR1)
│   └── kinematics_vae/        # kinematics-feature VAE family
│       ├── backbone/          # Scott v15 — shared across all 3 variants
│       │   ├── frame_features.py    # m_t = (p, θ, q_active); 3-way body split
│       │   ├── window_dataset.py    # make_windows
│       │   ├── normalizer.py        # per-feature z-score
│       │   ├── mlp.py               # Dense → ELU → LayerNorm block
│       │   ├── encoder.py           # MotionEncoderConv1DSharedSplit
│       │   ├── decoder.py           # MotionDecoderSplit
│       │   ├── predictor.py         # MotionPredictorSplit (aux dynamics loss)
│       │   ├── losses.py            # KL, masked MSE, σ-cap, pretrain_loss
│       │   ├── train_loop.py        # AdamW + warmup-hold-cosine + ckpt
│       │   │                        #   handles mutable bottleneck buffers
│       │   └── codegen.py           # encode_balanced_clips + load_checkpoint
│       ├── vae/                     # variant 1: VAE + post-hoc K-means
│       │   ├── bottleneck.py        # GaussianBottleneck (KL + σ-cap)
│       │   ├── train.py             # Hydra entry
│       │   ├── tokenize.py          # PCA → KMeans → codes npz
│       │   └── configs/pretrain.yaml
│       ├── vqvae/                   # variant 2: vanilla VQVAE (gradient codebook)
│       │   ├── quantizer.py         # VectorQuantizer (STE + commit + codebook)
│       │   ├── bottleneck.py        # single VQ over concat-24-D latent
│       │   ├── train.py             # Hydra entry
│       │   ├── tokenize.py          # codebook argmin → codes npz
│       │   └── configs/pretrain.yaml
│       ├── ps_vqvae/                # variant 3: product-space VQVAE (VQ-MAP)
│       │   ├── quantizer.py         # QuantizeEMAReset + MultiGroupQuantizer
│       │   │                        #   per-group EMA-VQ + dead-code reseed
│       │   ├── bottleneck.py        # body-part split as product groups
│       │   ├── train.py             # Hydra entry
│       │   ├── tokenize.py          # per-group codes → product index → npz
│       │   └── configs/pretrain.yaml
│       └── viz/
│           ├── projections.py       # PCA / t-SNE / UMAP wrappers
│           └── plot_latent.py       # 3-cols × N-rows per-variant grid
│                                    # + PS-VQVAE joint K_i×K_j heatmap
│
├── configs/                   # Hydra YAMLs (decoder + KPMS + shimoseq sweeps)
│   ├── moseq_decoder.yaml
│   ├── kpms_sweep.yaml
│   ├── kpms_sweep_smoke.yaml
│   ├── shimoseq_sweep.yaml
│   ├── shimoseq_sweep_smoke.yaml
│   ├── mimic_encoder.yaml
│   ├── inference.yaml
│   ├── pipeline_sweep.yaml
│   └── pipeline_best_kpms.yaml
│
├── ar_codegen/                # AR sequence-model code generators (HMM/GPT2-xs/Mamba)
├── experiments/               # 13 evaluation scripts + experiments/configs/
├── figures/                   # ~20 publication-figure scripts
├── tests/                     # pytest suite (load-bearing; see CLAUDE.md)
│
├── moseq_env_wrapper.py       # MoSeqImitation(Imitation) — injects kpms_code obs
│                              #   (+ motion_window_stack under e2e_mode)
├── moseq_decoder_network.py   # encoder–GRU–decoder policy (+ live VQ-VAE sub-modules
│                              #   `vqvae_encoder`/`vqvae_bottleneck`/`vqvae_decoder` under e2e_mode)
├── moseq_losses.py            # PPO + KL + distillation + code metrics
│                              #   (+ recon/commit/codebook + compute_e2e_split_grads under e2e_mode)
├── moseq_ppo.py               # PPO wrapper (monkey-patches ff_ppo; e2e weights threaded)
├── moseq_ppo_networks.py
├── e2e_vqvae.py               # ckpt loader + motion-buffer builder + JIT extractor
│                              #   (host-side glue for e2e_mode)
├── train_mimic_encoder.py     # pretrain the distillation target VAE
└── train_moseq_decoder.py     # main entry — Hydra
```

## Key non-obvious invariants

- **`c2a/codegen/code_sources.py::load_codes`** is the only entry point
  every downstream consumer (env wrapper, decoder, RL loop, experiments)
  uses for behavioral codes. Adding a new tokenizer family means adding a
  new `SOURCE_*` constant and a dispatch branch — nothing else changes.
- **e2e mode (`c2a/e2e_vqvae.py` + `e2e_config.e2e_mode`)** is the ONLY
  path that folds a live VQ-VAE into the policy. It requires
  `kpms_config.code_source == "vqvae"` and the codes_path must come from
  the same training run as `pretrained_vqvae_ckpt` (codebook size match is
  enforced at startup). The pretrained codes still flow into the env via
  `kpms_code` (used by the value network's separate embedding and by the
  top row of the eval-video code bar); the policy reads `motion_window_stack`
  to re-encode the same windows with the live VQ-VAE and substitutes
  straight-through `z_q` into the GRU. The reconstruction loss target is
  `motion_window_stack` itself, so we don't need to thread a separate
  recon-target tensor.
- **`c2a/codegen/shared/npz_schema.py::write_codes_npz`** is the only
  writer every tokenizer's codegen must call. It validates the schema
  before disk write and enforces `all_codes` sorted-union ordering when
  `all_indices` is provided — downstream experiments index `all_codes`
  by sorted-union position, so a wrong-order array would silently
  misalign every cross-tokenizer plot.
- **`c2a/codegen/kinematics_vae/backbone/`** is a Scott-v15 verbatim
  port. Only the bottleneck differs across variants; encoder / decoder /
  predictor / train_loop / codegen are shared.
- **The body-part split (root=4, axial=8, limb=12) is the natural
  product-space factorization** for PS-VQVAE. The branches Scott uses
  for per-branch KL allocation in the VAE map 1:1 to the paper's
  product-quantization groups.
- **PS-VQVAE keeps EMA buffers in a Flax `"vq_state"` mutable
  collection.** The train_loop reads `bottleneck.mutable_collections`
  and threads `mutable=["vq_state"]` through `apply`; the optimizer
  only sees the trainable subtree.

## Output locations (all gitignored)

| Output | Path | Producer |
|---|---|---|
| KPMS sweep | `c2a/outputs/kpms_sweep/` | `c2a.codegen.run_sweep` (model: kpms) |
| KPMS codes | `c2a/outputs/kpms_sweep/best_codes.npz` | `c2a.codegen.kpms.generate_codes` |
| shMoSeq sweep | `outputs/shimoseq_sweep/` | `c2a.codegen.run_sweep` (model: shimoseq) |
| shMoSeq codes | `outputs/shimoseq_sweep/best_codes.npz` | `c2a.codegen.shimoseq.generate_codes` |
| VAE ckpts | `trained_kinematics_vae_ckpts/vae/step_*` | `c2a.codegen.kinematics_vae.vae.train` |
| VAE codes | `trained_kinematics_vae_ckpts/vae/best_codes.npz` | `c2a.codegen.kinematics_vae.vae.tokenize` |
| VQVAE ckpts | `trained_kinematics_vae_ckpts/vqvae/step_*` | `c2a.codegen.kinematics_vae.vqvae.train` |
| VQVAE codes | `trained_kinematics_vae_ckpts/vqvae/best_codes.npz` | `c2a.codegen.kinematics_vae.vqvae.tokenize` |
| PS-VQVAE ckpts | `trained_kinematics_vae_ckpts/ps_vqvae/step_*` | `c2a.codegen.kinematics_vae.ps_vqvae.train` |
| PS-VQVAE codes | `trained_kinematics_vae_ckpts/ps_vqvae/best_codes.npz` | `c2a.codegen.kinematics_vae.ps_vqvae.tokenize` |
| MIMIC encoder | `trained_mimic_ckpts/` | `c2a.train_mimic_encoder` |
| Decoder RL | `model_checkpoints/` | `c2a.train_moseq_decoder` |
| WandB | `wandb/` (or `c2a/wandb/`) | every training script |

## See also

- `CLAUDE.md` — project conventions, code style, JAX/Flax patterns, pitfalls.
- `docs/run_c2a.md` — how to run each stage of the pipeline.
- `docs/moseq_decoder_config.md` — every Hydra key in `moseq_decoder.yaml`.
- `docs/experiments.md` — the 13 analysis scripts.
