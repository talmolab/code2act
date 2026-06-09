# `c2a.ar_codegen` — autoregressive code-sequence generators

Trains and serves four unconditional generators over per-frame KPMS codes
(or any compatible source from `c2a.codegen.code_sources`). All four obey one
`CodeGenerator.fit / sample / save / load` contract so
`c2a/experiments/run_code_generation.py` can load any of them by path.

| Model            | File                               | Where it comes from |
|------------------|------------------------------------|---------------------|
| GPT-2 XS         | `models/gpt2.py`                   | adapted from `flax/examples/lm1b` (Apache-2.0) |
| Mamba            | `models/mamba.py`                  | ported from `ekellbuch/deep_ssm/mixers/mamba_simple.py` |
| S5               | `models/s5.py` + `models/s5_ssm.py` + `models/s5_ssm_init.py` | vendored from `lindermanlab/S5` (MIT, see `models/S5_LICENSE`) |
| Categorical HMM  | `models/hmm.py`                    | dynamax — moved here from the old inline HMM in `run_code_generation.py` |

Defaults are sized for the c2a data scale (~17 k tokens, ~32-code vocab):
all neural models are ~600-700 k params. Edit `configs/model/*.yaml` to
scale up.

## Code-source genericity

Every generator consumes the standard `c2a.codegen.code_sources.CodesPayload`
(`train_codes`, `test_codes` of shape `[N, T]` int32 in `[0, num_codes)`).
The training entry point selects the source via `data=<name>`:

* `data=kpms_raw`            — raw KPMS sweep output (default)
* `data=kpms_merged`         — KPMS codes after `c2a.codegen.kpms.merge_states`
* `data=gt_directional_turn` — 5-class manual labels from
  `data/rodent/rodent_labels.csv`

## Train

```bash
# GPT-2 XS on raw KPMS codes (default for both flags).
python -m c2a.ar_codegen.train

# Mamba.
python -m c2a.ar_codegen.train model=mamba

# S5 on merged KPMS codes.
python -m c2a.ar_codegen.train model=s5 data=kpms_merged

# HMM on the 5-class GT labels.
python -m c2a.ar_codegen.train model=hmm data=gt_directional_turn

# Override anything inline:
python -m c2a.ar_codegen.train model=gpt2 \
    training.batch_size=32 training.num_epochs=400 \
    model.emb_dim=192 train_setup.run_name=gpt2_big
```

Each run writes a checkpoint directory to:

```
outputs/ar_codegen/{model_name}/{YYMMDD_HHMMSS_user}/
├── meta.json        # model_type, num_codes, code_source, config snapshot
├── params.msgpack   # (or hmm.npz for the HMM)
└── hydra/           # Hydra's run snapshot
```

WandB is on by default (`logging_config.use_wandb=true`,
`project=code2act_ar_codegen`). Disable with
`logging_config.wandb_mode=disabled` or `logging_config.use_wandb=false`.

## Sample

```bash
python -m c2a.ar_codegen.sample \
    checkpoint=outputs/ar_codegen/gpt2/260512_140000_user \
    n=100 length=250 seed=42 \
    out=outputs/ar_codegen/samples/gpt2_kpms_raw.npz
```

Output `.npz`:

```python
{
    "sequences":      np.ndarray[int32]    # [n, length]
    "model_type":     str
    "num_codes":      int
    "code_source":    str                  # provenance
    "checkpoint_path": str
    "temperature":    float
    "seed":           int
    "length":         int
    "n":              int
}
```

## Wire into the code-generation experiment

After training, plug the checkpoints into
`c2a.experiments.tokenizer.run_code_generation`:

```bash
python -m c2a.experiments.tokenizer.run_code_generation \
    ar_checkpoints.gpt2=outputs/ar_codegen/gpt2/<dir> \
    ar_checkpoints.mamba=outputs/ar_codegen/mamba/<dir> \
    ar_checkpoints.s5=outputs/ar_codegen/s5/<dir> \
    ar_checkpoints.hmm=outputs/ar_codegen/hmm/<dir> \
    generation.methods=[gpt2,mamba,s5,hmm_pretrained,arhmm_level1,arhmm_level2,transition_matrix]
```

The experiment loads each checkpoint via `CodeGenerator.from_checkpoint`,
sanity-checks that `num_codes` matches the decoder, and rolls the sampled
sequences through the trained MoSeq decoder. Unchanged inline methods
(empirical TM, ARHMM-level-1 / -2, controls) keep working alongside.

## Test

```bash
pytest c2a/ar_codegen/tests/ -v
```

Five test files:

* `test_gpt2_forward.py` — shape, causal mask, determinism.
* `test_mamba_forward.py` — shape, **scan parity vs. slow-loop reference** (1e-4 tol), causality.
* `test_s5_forward.py` — shape, causality.
* `test_hmm_fit_sample.py` — fit-then-sample on synthetic data, range check.
* `test_unified_interface.py` — every model round-trips `save → load → resample` deterministically.

## Notes

* All neural sample loops are O(L²) (full-sequence forward pass on a growing
  prefix). At L=250 this is fine — sampling 100 sequences runs in well under
  a minute on a single GPU. A cached-kv inference path can be added if longer
  contexts are needed.
* Mamba's pure-JAX selective scan uses `jax.lax.associative_scan` —
  the same operator S5 uses. There is no CUDA kernel dependency, and there
  is no Triton/`mamba-ssm` install requirement on the JAX side.
* S5 here is the same `S5SSM` block from `lindermanlab/S5`, vendored
  with only an import-path patch and a deprecated-typing fix. See
  `models/S5_LICENSE` for the MIT license text.
