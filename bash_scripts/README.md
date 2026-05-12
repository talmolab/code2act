# bash_scripts — orchestration for multi-stage sweeps

These scripts wrap the three-process Code2Act pipeline (KPMS sweep → code
generation → decoder RL training). Each process runs independently because
`keypoint_moseq` requires `jax_enable_x64=True`, which is irreversible
per-process.

All scripts must be run from the `code2act/` repo root. They derive
`CODE2ACT_ROOT` from their own location and set `C2A_ROOT` in the
environment so every Hydra config (`${oc.env:C2A_ROOT,.}/...`) resolves
correctly.

## Prerequisites

See `code2act/README.md` for install steps. In short:

```bash
# From code2act/
git clone -b kevin/kpms-decoder https://github.com/talmolab/track-mjx.git
uv venv && source .venv/bin/activate
cd track-mjx && uv pip install -e ".[cuda13]" && cd ..
uv pip install keypoint-moseq[cuda] jax-moseq[cuda]
uv pip install -e .
```

## `run_pipeline.sh` — full KPMS × decoder grid

Reads `c2a/configs/pipeline_sweep.yaml` and chains:

1. **Stage 1** — `c2a.codegen.run_sweep` fits KPMS for every
   `(num_states × kappa)` combination.
2. **Stage 2** — `c2a.codegen.kpms.generate_all_codes` emits one `.npz` per KPMS
   setting + a `manifest.json` index.
3. **Stage 3** — `c2a.train_moseq_decoder` runs a decoder for every
   `(code_set × latent_dim × kl_weight)` combination.

### Usage

```bash
cd code2act
source .venv/bin/activate

bash bash_scripts/run_pipeline.sh                              # full run
bash bash_scripts/run_pipeline.sh --skip-kpms                  # reuse sweep
bash bash_scripts/run_pipeline.sh --skip-kpms --skip-codegen   # only decoders
bash bash_scripts/run_pipeline.sh --config c2a/configs/custom.yaml
bash bash_scripts/run_pipeline.sh --dry-run                    # preview only
```

Grid size for the default `pipeline_sweep.yaml`:
`3 num_states × 5 kappa = 15 KPMS settings` → `15 code sets` →
`15 × 2 latent_dim × 2 kl_weight = 60 decoder runs`.

### Failure handling

Scripts run decoder jobs sequentially and report failures at the end. To
re-run only the decoder stage after a config fix:

```bash
bash bash_scripts/run_pipeline.sh --skip-kpms --skip-codegen
```

Already-completed WandB runs appear as independent entries in the same
group — use `train_setup.run_name` to identify each combination.

## `run_pipeline_best_kpms.sh` — fixed-KPMS decoder sweep

Reads `c2a/configs/pipeline_best_kpms.yaml`:

1. **Stage 1** — KPMS sweep with fixed `num_states=25`, sweeping over
   `kappa`; selects the best by reconstruction MSE.
2. **Stage 2** — generates one code database from the best model.
3. **Stage 3** — decoder RL sweep over
   `(kl_weight × kl_schedule × z_e_dropout_rate)`.

### Usage

```bash
bash bash_scripts/run_pipeline_best_kpms.sh
bash bash_scripts/run_pipeline_best_kpms.sh --skip-kpms
bash bash_scripts/run_pipeline_best_kpms.sh --dry-run
bash bash_scripts/run_pipeline_best_kpms.sh --config c2a/configs/custom.yaml
```

## Single-run training (no bash, for debugging)

If you want to iterate on decoder hyperparameters without the sweep
orchestration:

```bash
python -m c2a.train_moseq_decoder \
    kpms_config.codes_path=/abs/path/to/best_codes.npz

# Smoke test — random codes, no KPMS required
python -m c2a.train_moseq_decoder kpms_config.codes_path=null \
    train_setup.train_config.num_timesteps=5_000_000

# Use a different decoder config
python -m c2a.train_moseq_decoder \
    network_config.rnn_hidden_sizes="[512]" \
    network_config.kl_weight=0.1
```

See `code2act/docs/run_c2a.md` and `code2act/docs/moseq_decoder_config.md`
for the full config key reference.

## Hyperparameter guide

Results from the initial sweep (Mar–Apr 2026, ~60 fits) inform the ranges
below. Fuller details in `docs/moseq_decoder_config.md`.

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `sweep.num_states` | KPMS vocabulary size. 50 collapses to ~10 effective codes. | `10–30` |
| `sweep.kappa` | Self-transition stickiness. Higher = longer syllables. | `1e2–1e4` |
| `sweep.model_type` | `arhmm` recommended (SLDS has 26× worse MSE). | `["arhmm"]` |
| `decoder.continuous_latent_dim` | Encoder bottleneck. | `2–8` |
| `decoder.kl_weight` | KL on continuous latent. | `0.01–0.5` |
| `decoder.rnn_hidden_sizes` | GRU hidden dims. | `[256]` or `[256, 256]` |
| `decoder.code_stack_size` | Number of lookahead codes concatenated. | `1–5` |

## WandB output

Each run logs:

| Metric | Meaning |
|--------|---------|
| `moseq/perplexity` | Code usage entropy |
| `moseq/codes_used` | Active codes count |
| `moseq/eval_transition_rate` | Fraction of steps with code changes (match ~0.005 for durable KPMS codes; high values indicate the code corruption bug) |
| `kl_loss` | KL on continuous latent |
| `distill_kl_loss` | Distillation KL (when `use_distillation_head=true`) |
| `hidden_state_norm` | Mean GRU hidden state L2 norm |
| `z_e_norm` | Mean L2 norm of continuous latent |

Full metric table in `docs/moseq_decoder_config.md`.

## Pipeline outputs

```
c2a/outputs/pipeline_sweep/
├── kpms/                           # Stage 1
│   ├── sweep_results.json
│   └── s{NS}_k{KAPPA}_l{LD}_arhmm/
│       └── seed{0,1}/
├── codes/                          # Stage 2
│   ├── manifest.json
│   └── s{NS}_k{KAPPA}_arhmm.npz
model_checkpoints/                  # Stage 3 (at code2act root, gitignored)
└── {TIMESTAMP}_{RUN_NAME}/
```
