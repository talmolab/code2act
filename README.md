# code2act

Code2Act trains a KPMS-driven RNN decoder policy for motion imitation:
**Keypoint-MoSeq** produces a discrete syllable code every frame, and a
GRU decoder maps each code (plus proprioception) to actions.

Website: <https://kbian.org/Code2Act-website/#>

Code2Act is a sibling of [**MIMIC-MJX**](https://mimic-mjx.talmolab.org/).
It reuses MIMIC-MJX's [track-mjx](https://github.com/talmolab/track-mjx)
infrastructure for the RL training loop (PPO, checkpointing, domain
randomization), the environment (`vnl_playground`), and the neural network
primitives (`IntentionEncoder`, `IntentionDecoder`, `reparameterize`). The
c2a package on top contributes:

- Unified behavioral-code generation under `c2a.codegen` — sweep
  dispatcher (`c2a.codegen.run_sweep`), per-tokenizer fitters
  (`c2a.codegen.kpms`, `c2a.codegen.shimoseq`), shared label-alignment
  and codes-npz schema (`c2a.codegen.shared`), and pluggable code-source
  loader (`c2a.codegen.code_sources`).
- `MoSeqImitation` — env subclass that injects KPMS codes into the
  observation dict.
- `MoSeqRecurrentDecoderNetwork` — encoder–GRU–decoder policy with optional
  distillation head and pretrained VAE decoder readout.
- 13 analysis experiments + figure scripts (`c2a.experiments`, `c2a.figures`).

## Installation

Code2Act depends on `track-mjx`, which is NOT a PyPI dep — you clone it as a
subfolder and pip-install it into the same venv. This mirrors the track-mjx
install flow verbatim.

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA 12.x or 13.x for GPU (optional)

### Install

```bash
git clone <code2act-repo-url> && cd code2act

# 1. Pull the track-mjx branch that c2a targets
git clone -b kevin/kpms-decoder https://github.com/talmolab/track-mjx.git

# 2. Create the venv at the code2act root (not inside track-mjx/)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate

# 3. Install track-mjx into this venv (brings in jax, brax, mjx, flax,
#    orbax, hydra, wandb, vnl-playground, mujoco_playground, …)
cd track-mjx
uv pip install -e ".[cuda12]"
cd ..

# 4. Install keypoint_moseq for the KPMS fitting stages
uv pip install keypoint-moseq[cuda] jax-moseq[cuda]

# 5. Install c2a
uv pip install -e .
```

Verify:

```bash
python -c "import jax; print(jax.devices())"
python -c "import c2a, c2a.moseq_ppo, c2a.train_moseq_decoder"
```

### Analysis dependencies (DSA / rSLDS — optional)

The hidden-state analysis scripts under `c2a/figures/plot_hidden_*.py`
need two extra GitHub-only packages: Mitchell Ostrow's **DSA** (NeurIPS
2024) and Linderman lab's **ssm** (rSLDS). They are declared in
`pyproject.toml` under the `[analysis]` extra, but the install needs
sequencing — a plain `uv pip install -e ".[analysis]"` will fail on
`ssm` because that package imports `numpy` + `cython` at build time
without declaring them as build deps. Install in this order:

```bash
# DSA pulls `torch` as a transitive dep with no upper pin. PyTorch
# 2.10+ shifted its Linux wheels to depend on the CUDA-13 NVIDIA
# wheels (`nvidia-cudnn-cu13`, `nvidia-cublas` (unsuffixed),
# `cuda-toolkit==13.0.2`, …). Those clobber the CUDA-12 cuDNN that
# `jax-cuda12-plugin` requires, breaking `import jax` with
# `CUDNN_STATUS_NOT_INITIALIZED`. Pin torch to the last CUDA-12 line
# (≤2.9.x) BEFORE DSA so uv's resolver doesn't pick 2.11.
uv pip install "torch<2.10"

# DSA — uppercase package on GitHub. (PyPI's `dsa` is a different,
# unrelated package; don't `uv pip install DSA` from PyPI.)
uv pip install git+https://github.com/mitchellostrow/DSA.git

# ssm build prerequisites + ssm itself with build-isolation disabled
# so the build sees the venv's numpy and cython.
uv pip install autograd cython
uv pip install --no-build-isolation git+https://github.com/lindermanlab/ssm.git
```

If you already hit the cu13 collision (symptom: `RuntimeError: cuDNN
not found` from `jax_plugins.xla_cuda12`, or `CUDNN_STATUS_NOT_INITIALIZED`
with both `nvidia-cudnn-cu12-*` and `nvidia-cudnn-cu13-*` present in
`uv pip list`), recover with:

```bash
uv pip install --reinstall "torch<2.10"
uv pip uninstall \
    nvidia-cudnn-cu13 nvidia-nccl-cu13 nvidia-cusparselt-cu13 \
    nvidia-nvshmem-cu13 nvidia-cublas nvidia-cuda-runtime \
    nvidia-cuda-nvrtc nvidia-cuda-cupti nvidia-cufft nvidia-nvjitlink \
    cuda-toolkit cuda-bindings triton
# Then re-extract cuDNN in case NFS silly-renamed its .so files during
# the torch swap (you'll see `.nfsXXXX` ghosts under nvidia/cudnn/lib/):
uv pip install --reinstall nvidia-cudnn-cu12==9.10.2.21
```

**One-line patch for DSA on Python 3.12.** DSA uses
`ClassVar[Final]` (un-parameterised `Final`), which Python 3.12's
typing rejects with `TypeError: Plain typing.Final is not valid as
type argument`. Until upstream fixes it, patch in place:

```bash
sed -i 's/compare: ClassVar\[Final\] = "state"/compare: ClassVar[str] = "state"/' \
    .venv/lib/python3.12/site-packages/DSA/dsa.py
```

Verify everything:

```bash
python -c "from DSA import DSA, InputDSA; print('DSA OK')"
python -c "from ssm import SLDS; print('ssm OK')"
```

You only need this for the `plot_hidden_dsa.py` / `plot_hidden_rslds.py`
figure scripts (under `c2a/figures/`). The training pipeline, sweep,
and merge scripts don't need these packages.

### Reference data

The rodent reference clips (`rodent_reference_clips.h5`,
`rodent_balanced_splits.json`) ship in `data/rodent/`. If you cloned without
these (they're gitignored), pull them from the HuggingFace dataset
`talmolab/MIMIC-MJX`:

#### Behavior labels

`data/rodent/rodent_labels.csv` contains manual frame-level annotations of
each clip with two parallel tracks:

- **behavior** ∈ {Immobile, Rear, Turn, Walk}
- **direction** ∈ {Straight, Left, Right}

These labels are sourced from
[KevinBian107/TopoMimic-annotation](https://github.com/KevinBian107/TopoMimic-annotation)
and are used by:

- `c2a.codegen.shared.label_alignment` (loader / coverage metrics)
- `c2a.codegen.run_sweep` (NMI + Hungarian-purity reporting at three
  granularities: behavior / direction / combined)
- `c2a.experiments.run_labeled_hidden_dynamics` (label-anchored hidden
  state experiment)

The CSV is annotated against the original 30 fps videos (500 frames per
clip). The H5 mocap data is at half the rate (250 frames per clip), so
``c2a.codegen.shared.label_alignment`` divides CSV frame numbers by 2 by default
(``csv_to_h5_subsample=2``).

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='talmolab/MIMIC-MJX', repo_type='dataset',
    filename='data/rodent/rodent_reference_clips.h5', local_dir='.',
)
"
```

The rodent XML (`c2a/assets/rodent.xml`) is vendored in-repo and is the
correct one — `vnl_playground` ships a different XML with `nq=67` that
silently produces garbage keypoints; c2a refuses to fall back to it.

## Quick start

### Single decoder run

```bash
source .venv/bin/activate

# If you already have KPMS codes at c2a/outputs/kpms_sweep/best_codes.npz:
python -m c2a.train_moseq_decoder

# Or pass an explicit path:
python -m c2a.train_moseq_decoder \
    kpms_config.codes_path=/abs/path/to/best_codes.npz

# Smoke test — random codes, no KPMS required:
python -m c2a.train_moseq_decoder kpms_config.codes_path=null \
    train_setup.train_config.num_timesteps=5_000_000
```

Config is Hydra. Override any key on the command line
(`network_config.kl_weight=0.1`, `train_setup.run_name=my_run`, …). See
[`docs/moseq_decoder_config.md`](docs/moseq_decoder_config.md) for the full
config key reference.

### Full sweep

```bash
bash bash_scripts/run_pipeline.sh               # KPMS × decoder grid
bash bash_scripts/run_pipeline_best_kpms.sh     # fixed-KPMS decoder sweep
```

See [`bash_scripts/README.md`](bash_scripts/README.md) for sweep details.

### Auto-merge KPMS syllables

After the sweep, collapse near-duplicate syllables in each top-K fit
using the AR-HMM Bhattacharyya criterion, and re-render the merged
trajectory grids:

```bash
python -m c2a.codegen.kpms.merge_states --sweep-dir outputs/kpms_sweep
```

By default this ranks the top-K fits by the **Weinreb 2024**
criterion (`--rank eml`): filter fits whose median state duration sits
in the 400 ms ±30 % window, then sort by EML descending. Same criterion
the sweep uses to pick `best_model`, so the M1..M5 reported here line
up with what the paper recommends. Pass `--rank mse` for the legacy
reconstruction-MSE ranking, or `--rank nmi` for label-aligned NMI.

Outputs at `outputs/kpms_sweep/auto_merged/M{1..5}/`:
`checkpoint_merged.h5`, `auto_merge_report.json`,
`bhattacharyya_matrix.npy`, `trajectory_plots/planes_grid.{png,gif}`,
plus `best_codes_merged.npz` for the M{k} matching `best_codes.npz`.
Wire that file into `kpms_config.code_source: kpms_merged` to train on
merged labels. Tune via `--threshold` (default 0.3) and `--linkage`
(`complete` / `single`).

### Analysis experiments

13 experiments evaluate different aspects of the trained decoder (inference,
code-sequence ordering, generalization, KID/FID, gait dynamics, hidden-state
dynamics). See [`docs/experiments.md`](docs/experiments.md).

```bash
python -m c2a.experiments.run_inference
python -m c2a.experiments.run_generalization_kid
```

## Repository layout

```
code2act/
├── README.md                       # this file
├── pyproject.toml                  # package "c2a"
├── .gitignore                      # ignores track-mjx/, .venv/, data/*.h5, checkpoints
├── bash_scripts/                   # pipeline orchestration (see README)
│   ├── README.md
│   ├── run_pipeline.sh
│   └── run_pipeline_best_kpms.sh
├── docs/
│   ├── run_c2a.md                  # single-run + sweep
│   ├── moseq_decoder_config.md     # every yaml key explained
│   └── experiments.md              # 13 analysis scripts
├── c2a/                            # pip-installable package
│   ├── assets/rodent.xml
│   ├── configs/                    # Hydra YAMLs
│   ├── codegen/                    # generate_codes, generate_all_codes
│   ├── kpms/                       # keypoint-moseq fitting helpers
│   ├── sweep/                      # grid runner
│   ├── experiments/                # 13 evaluation scripts + shared/
│   ├── figures/                    # publication figures + plots
│   ├── tests/
│   ├── paths.py                    # get_repo_root() / get_c2a_root()
│   ├── rendering_utils.py          # vendored video-overlay helpers
│   ├── moseq_env_wrapper.py        # MoSeqImitation(Imitation)
│   ├── moseq_decoder_network.py    # MoSeqRecurrentDecoderNetwork
│   ├── moseq_losses.py
│   ├── moseq_ppo.py                # PPO wrapper (monkey-patches ff_ppo)
│   ├── moseq_ppo_networks.py
│   ├── train_mimic_encoder.py      # pretrain the distill target
│   └── train_moseq_decoder.py      # main entry
├── Code2Act-website/               # Astro site for kbian.org/Code2Act-website
├── track-mjx/                      # ★ user-cloned sibling, gitignored
├── .venv/                          # ★ user-created, gitignored
├── data/rodent/                    # H5 reference clips (gitignored)
├── trained_mimic_ckpts/            # distillation-target checkpoints (gitignored)
└── model_checkpoints/              # decoder RL checkpoints (gitignored)
```

## License

BSD 3-Clause (see [`LICENSE`](LICENSE)).

## References

- **Website:** <https://kbian.org/Code2Act-website/#>
- **MIMIC-MJX:** <https://mimic-mjx.talmolab.org/>
- **Keypoint-MoSeq:** <https://keypoint-moseq.readthedocs.io/en/latest/>
- **TopoMimic-annotation** (behavior + direction labels in
  `data/rodent/rodent_labels.csv`):
  <https://github.com/KevinBian107/TopoMimic-annotation>
