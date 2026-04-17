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

- KPMS hyperparameter sweep and code generation (`c2a.sweep`, `c2a.codegen`,
  `c2a.kpms`).
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
uv venv
source .venv/bin/activate

# 3. Install track-mjx into this venv (brings in jax, brax, mjx, flax,
#    orbax, hydra, wandb, vnl-playground, mujoco_playground, …)
cd track-mjx
uv pip install -e ".[cuda13]"     # or .[cuda12] / . for CPU-only
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

### Reference data

The rodent reference clips (`rodent_reference_clips.h5`,
`rodent_balanced_splits.json`) ship in `data/rodent/`. If you cloned without
these (they're gitignored), pull them from the HuggingFace dataset
`talmolab/MIMIC-MJX`:

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

## Citation

If you use code2act in your research, please cite the MIMIC-MJX paper — the
backbone infrastructure — and the Code2Act write-up:

```bibtex
@misc{zhang2025mimicmjxneuromechanicalemulationanimal,
      title={MIMIC-MJX: Neuromechanical Emulation of Animal Behavior},
      author={Charles Y. Zhang and Yuanjia Yang and Aidan Sirbu and
              Elliott T. T. Abe and Emil Wärnberg and Eric J. Leonardis and
              Diego E. Aldarondo and Adam Lee and Aaditya Prasad and
              Jason Foat and Kaiwen Bian and Joshua Park and Rusham Bhatt and
              Hutton Saunders and Akira Nagamori and
              Ayesha R. Thanawalla and Kee Wui Huang and Fabian Plum and
              Hendrik K. Beck and Steven W. Flavell and David Labonte and
              Blake A. Richards and Bingni W. Brunton and Eiman Azim and
              Bence P. Ölveczky and Talmo D. Pereira},
      year={2025}, eprint={2511.20532}, archivePrefix={arXiv},
      primaryClass={q-bio.NC}, url={https://arxiv.org/abs/2511.20532},
}
```

## License

BSD 3-Clause (see [`LICENSE`](LICENSE)).

## References

- **Website:** <https://kbian.org/Code2Act-website/#>
- **MIMIC-MJX:** <https://mimic-mjx.talmolab.org/>
- **Keypoint-MoSeq:** <https://keypoint-moseq.readthedocs.io/en/latest/>
