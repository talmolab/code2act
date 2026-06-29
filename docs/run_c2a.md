# Running code2act

This guide covers single-run training, multi-stage sweeps, and the three
decoder modes (vanilla, distillation, pretrained-decoder).

## Pipeline overview

```
KPMS Sweep → Code Generation → Decoder RL Training
(jax x64)    (jax x64)         (jax float32)
```

Each stage runs in a **separate process** because `keypoint_moseq`
requires `jax_enable_x64=True`, which is process-global and irreversible.

## Prerequisites

See [`../README.md`](../README.md). In short: clone track-mjx as a subfolder,
create venv at repo root, `uv pip install -e ".[cuda13]"` into track-mjx,
then `uv pip install -e .` on c2a.

## Single run

```bash
cd code2act
source .venv/bin/activate

python -m c2a.train_moseq_decoder
```

Reads `c2a/configs/moseq_decoder.yaml` (Hydra). Override any key:

```bash
# Change decoder hidden sizes
python -m c2a.train_moseq_decoder network_config.rnn_hidden_sizes="[512]"

# Fall back to feedforward MLP decoder
python -m c2a.train_moseq_decoder network_config.use_rnn_decoder=false

# Shorter run for smoke test
python -m c2a.train_moseq_decoder \
    train_setup.train_config.num_timesteps=5_000_000

# Random codes (no KPMS required)
python -m c2a.train_moseq_decoder kpms_config.codes_path=null
```

Full config reference: [`moseq_decoder_config.md`](moseq_decoder_config.md).

## Multi-stage sweeps

```bash
bash bash_scripts/pipeline/run_kpms_effect_sweep.sh           # fit 10 KPMS tokenizers, decoder per tokenizer
bash bash_scripts/pipeline/run_vae_effect_sweep.sh           # kinematics-VAE effect sweep
```

See [`../bash_scripts/README.md`](../bash_scripts/README.md) for options and
orchestration details.

## Manual pipeline stages

### Stage 1 — KPMS sweep

```bash
python -m c2a.codegen.run_sweep --config c2a/configs/kpms_sweep.yaml
```

Fits KPMS over `(num_states × kappa)` grid. Results → `c2a/outputs/kpms_sweep/sweep_results.json`.

### Stage 2 — code generation

Single best model:
```bash
python -m c2a.codegen.kpms.generate_codes
```

Per-setting codes (for sweep):
```bash
python -m c2a.codegen.kpms.generate_all_codes \
    --sweep-results outputs/tokenizer/kpms_effect_sweep/sweep_results.json \
    --balanced-split data/rodent/rodent_balanced_splits.json \
    --output-dir outputs/tokenizer/kpms_effect_sweep/codes
```

Produces one `.npz` per `(num_states, kappa)` + `manifest.json`.

### Stage 3 — decoder RL

```bash
python -m c2a.train_moseq_decoder \
    kpms_config.codes_path=c2a/outputs/kpms_sweep/best_codes.npz
```

## Kinematics-VAE tokenizer

An offline tokenizer trains an autoencoder over per-frame kinematic
features (10-frame proprio windows: `(p, θ, q_active)`) and emits a
per-frame discrete code in the same `CodesPayload` schema KPMS uses. It
uses the **Scott v15 conv1d encoder backbone** (3-way root/axial/limb
split, latent_dim=24) with a Gaussian bottleneck:

| Variant | Bottleneck | Discretization | Source ID |
|---|---|---|---|
| `vae` | Gaussian reparam | post-hoc K-Means (+ optional PCA) | `kinematics_vae` |

Stage 1 — **pretrain**:
```bash
python -m c2a.codegen.kinematics_vae.vae.train
```

Writes msgpack ckpts to `trained_kinematics_vae_ckpts/vae/step_*`.
Defaults to v15's recipe — 30k steps, batch 8192, AdamW with
warmup-hold-cosine LR. Override on the CLI:

```bash
python -m c2a.codegen.kinematics_vae.vae.train \
    train.num_steps=5000 \
    train.batch_size=2048 \
    train.ckpt_dir=trained_kinematics_vae_ckpts/vae_smoke
```

Stage 2 — **tokenize** (PCA → KMeans on the latent):
```bash
python -m c2a.codegen.kinematics_vae.vae.tokenize \
    --ckpt-dir trained_kinematics_vae_ckpts/vae/step_0030000 \
    --output trained_kinematics_vae_ckpts/vae/best_codes.npz \
    --num-codes 50 --pca-dim 8
```

Stage 3 — **decoder RL** (same trainer; `code_source` selects the family):
```bash
python -m c2a.train_moseq_decoder \
    kpms_config.code_source=kinematics_vae \
    kpms_config.codes_path=trained_kinematics_vae_ckpts/vae/best_codes.npz
```

Note: `kpms_config.codes_path=null` is **rejected** for the
kinematics-VAE source (would silently produce random codes whose
codebook size doesn't match the tokenizer's). For smoke runs without a
trained tokenizer use `code_source: kpms_raw` with `codes_path: null`.

End-to-end orchestration:
```bash
bash bash_scripts/pipeline/run_vae_effect_sweep.sh            # train best VAE, 10 vocab sizes, decoder per tokenizer
bash bash_scripts/pipeline/run_vae_effect_sweep.sh --launch   # ...and run the decoders sequentially
```

## Pre-training the distillation encoder

`use_distillation_head=true` requires a pretrained MimicEncoder checkpoint.
Two are shipped (`trained_mimic_ckpts/260413_040706_764936/`,
`260405_235031_234849/`). To train your own:

```bash
python -m c2a.train_mimic_encoder
```

Reads `c2a/configs/mimic_encoder.yaml`. The encoder/decoder layer sizes
MUST match `moseq_decoder.yaml` for the subsequent decoder run to load
them. See [`moseq_decoder_config.md`](moseq_decoder_config.md) for the
constraint.

## Decoder architecture modes

Selected via `network_config` keys. All three share the code embedding
`kpms_code → Embed(num_codes, embed_dim)`, proprio, and a continuous
encoder that maps `imitation_target` → `(mean, logvar)`.

### Vanilla RNN (baseline)

```yaml
use_rnn_decoder: true
use_distillation_head: false
use_pretrained_decoder: false
```

Flow:
```
[code_emb, z_e, proprio] → GRU → Dense → action_params
```

z_e gets KL-to-prior regularization via `kl_weight` (optionally scheduled
by `kl_schedule`).

### Distillation head

```yaml
use_distillation_head: true
distillation_encoder_checkpoint: /path/to/mimic_encoder
```

Flow:
```
[code_emb, proprio] → GRU → h_t
h_t → distill_head MLP → (mu_d, logvar_d)
encoder([imitation_target, proprio]) → (mu_e, logvar_e)   [frozen]
loss += distill_kl_weight * KL(sg(mu_e, logvar_e) || (mu_d, logvar_d))
```

z_e is **entirely removed** from the action path. The RNN must learn to
predict the frozen encoder's latent from just `(code_emb, proprio,
hidden_state)`. `kl_weight` is auto-forced to 0 so the continuous KL does
not double-count.

### Pretrained-decoder readout (unified VAE head)

```yaml
use_distillation_head: true
use_pretrained_decoder: true
decoder_layer_sizes_vae: [512, 256, 256, 256]   # must match mimic_encoder.yaml
```

Flow:
```
[code_emb, proprio] → GRU → h_t
h_t → VAE head MLP → (mu, logvar)
z = reparameterize(key, mu, logvar)
[z, proprio] → IntentionDecoder (frozen from checkpoint) → action_params
```

In this mode `(mu, logvar)` from the VAE head ARE the distill prediction
— no separate `_distill_head` MLP is built. The frozen `IntentionDecoder`
maps `z` back to action_params. `freeze_decoder=true` keeps the decoder
subtree frozen during RL.

## Debugging

### "Training silently runs on random codes"
Symptom: `moseq/eval_transition_rate` ≈ 0.49 instead of ~0.005. As of
code2act, this fires `FileNotFoundError` at startup. If you see
`kpms_config.codes_path=null`, training explicitly uses random codes for
smoke-testing.

### "Balanced split is wrong"
`balanced_split_path` raises if set-but-missing (no silent fallback). Set
to `null` to use a seeded random 70/30 split.

### "RNN loss crashes with `policy_rng missing`"
This assertion fires when the rollout isn't threading `policy_rng` into
transition extras. Check `c2a.moseq_ppo_networks.make_moseq_rnn_rollout_policy_fn`.

### "Import c2a fails"
`uv pip install -e .` from the code2act root installs c2a. Also verify
the track-mjx dep is installed in the same venv:
`python -c "import track_mjx; print(track_mjx.__file__)"`.

## WandB output

| Metric | Meaning |
|--------|---------|
| `moseq/perplexity` | Code usage entropy |
| `moseq/codes_used` | Active code count |
| `moseq/eval_transition_rate` | Fraction of steps with a code transition |
| `kl_loss` | Continuous KL (zero-weighted when distill head is on) |
| `distill_kl_loss` | Distill KL (only when `use_distillation_head=true`) |
| `hidden_state_norm` | GRU hidden L2 |
| `z_e_norm` | z_e L2 |

Full table in [`moseq_decoder_config.md`](moseq_decoder_config.md).
