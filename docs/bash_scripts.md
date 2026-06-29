# bash_scripts — pipeline & sweep orchestration

These scripts wrap the three-process Code2Act pipeline (KPMS / tokenizer fit →
code generation → decoder RL). Each stage runs as an independent process because
`keypoint_moseq` requires `jax_enable_x64=True`, which is irreversible
per-process.

All scripts are run **from the `code2act/` repo root** and derive
`CODE2ACT_ROOT` from their own location, exporting `C2A_ROOT` so every Hydra
config (`${oc.env:C2A_ROOT,.}/...`) resolves correctly.

```bash
cd code2act
source .venv/bin/activate
bash bash_scripts/<group>/<script>.sh
```

Install steps are in [`installation.md`](installation.md); the decoder config
reference is [`moseq_decoder_config.md`](moseq_decoder_config.md).

## Layout

```
bash_scripts/
├── _pyrun.py                 # venv module launcher (used by scripts that call
│                             #   python3.12 bash_scripts/_pyrun.py <module>)
├── pipeline/                 # full multi-stage orchestration
├── sweeps/                   # tokenizer / KPMS hyperparameter sweeps
└── training/                 # single decoder / controller trainers
```

> All scripts compute the repo root as two levels up from their own directory
> (`dirname(dirname(SCRIPT_DIR))`), so they must stay one level deep under
> `bash_scripts/<group>/`. `_pyrun.py` stays at the `bash_scripts/` root and is
> referenced repo-root-relative (`bash_scripts/_pyrun.py`) after each script
> `cd`s to the root.

## `pipeline/` — full multi-stage orchestration

| Script | What it does |
|---|---|
| `run_vae_effect_sweep.sh` | **Kinematics-VAE counterpart of `run_kpms_effect_sweep.sh`.** Trains one kinematics-VAE with the best params (kl=0.001, from `best_vae` 260603_180852_140475), tokenizes at 10 vocabulary sizes K (`num_codes` — the VAE analog of KPMS kappa), then emits one **full-budget decoder job per tokenizer with the byte-identical production config** — only the code stream varies. `--launch` runs them sequentially; `--skip-train`/`--skip-tokenize` reuse cached artifacts. |
| `run_kpms_effect_sweep.sh` | **Controlled tokenizer comparison.** Fits 10 KPMS tokenizers (K=100, 10 log-spaced kappa, arhmm, 1 seed) via `c2a/configs/kpms_effect_sweep.yaml`, generates per-setting codes, then emits one **full-budget decoder job per tokenizer with a byte-identical decoder config** — only the code stream varies, isolating the tokenizer's downstream effect. Default prints the 10 job commands to `outputs/tokenizer/kpms_effect_sweep/decoder_jobs.sh` (submit each as a job); `--launch` runs them sequentially; `--skip-fit` reuses cached codes. |

```bash
bash bash_scripts/pipeline/run_vae_effect_sweep.sh                     # train best VAE, 10 vocab sizes, decoder per tokenizer
bash bash_scripts/pipeline/run_kpms_effect_sweep.sh                    # fit 10 KPMS + emit a decoder job per tokenizer
bash bash_scripts/pipeline/run_kpms_effect_sweep.sh --launch           # ...and run them sequentially here
```

## `sweeps/` — tokenizer / KPMS hyperparameter sweeps

| Script | What it does |
|---|---|
| `sweep_kpms_k100.sh` | KPMS K=100 sweep (kappa × model_type × seeds) via `c2a.codegen.run_sweep` with `kpms_sweep_k100.yaml`. Produces the K=100 ARHMM codes behind the **featured** controller. |
| `sweep_vae.sh` | Kinematics-VAE tokenizer sweep (KL × post-hoc K) via `c2a.codegen.kinematics_vae.vae.{train,tokenize}`. |
| `sweep_all_tokenizers.sh` | Master orchestrator: launches the KPMS / VAE sweeps in parallel `screen` sessions, then runs `c2a.experiments.tokenizer.run_tokenizer_diagnostics`. Calls the sibling sweep scripts via `$SCRIPT_DIR`. |

## `training/` — single decoder / controller trainers

| Script | What it does |
|---|---|
| `run_kpms_baseline.sh` | K=100 KPMS decoder, pure motion-tracking baseline (`reward_mode=off`). |
| `run_c2c_additive.sh` | Decoder training with the **code-consistency (code2code)** additive reward. The featured controller uses this term. |
| `run_train_designed_segmenter.sh` | Trains the controller on the **hand-designed multi-scale codebook** (`c2a.codegen.designed_segmenter` → `c2a.train_moseq_decoder`). Drives all `*_multiscale` paper figures. |

## WandB metrics (decoder runs)

| Metric | Meaning |
|---|---|
| `moseq/perplexity` | Code-usage entropy |
| `moseq/codes_used` | Active codes count |
| `moseq/eval_transition_rate` | Fraction of steps with code changes (~0.005 for durable KPMS codes; high = code-corruption bug) |
| `kl_loss` | KL on continuous latent |
| `distill_kl_loss` | Distillation KL (when `use_distillation_head=true`) |
| `hidden_state_norm` | Mean GRU hidden-state L2 norm |
| `z_e_norm` | Mean L2 norm of continuous latent |

Full metric + config reference: [`moseq_decoder_config.md`](moseq_decoder_config.md).
