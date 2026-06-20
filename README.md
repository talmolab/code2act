# [Code2Act: Motor Programs from Discrete Behavioral Codes for Embodied Control](https://kbian.org/Code2Act-website/)

Computational models of naturalistic behavior fall into two families that do not
meet. Probabilistic state-space models such as Keypoint-MoSeq segment continuous
behavior into discrete syllables from keypoints; they describe what a behavior
looks like but not how it is physically produced. Reinforcement-learning
imitation policies produce biomechanically realistic motion but expose no
behavioral vocabulary one can index to steer them. A behavioral label can be
observed and correlated, but it cannot be run.

**Code2Act** makes a discrete behavioral code *executable*. Conditioning a
distilled physics controller on the code stream compiles each symbolic label
into a runnable motor program for a 38-actuator biomechanical rodent: a code
that can be executed, started from an arbitrary initial state, composed into
sequences, and probed through the controller's internal dynamics. Each held code
behaves as a behaviorally typed attractor that the controller actively defends,
and the same structure holds for a hand-specified codebook the controller never
co-adapted with. This turns the study of behavior from observational and
statistical into generative, interventional, and dynamical.

## Installation

Full setup — track-mjx clone flow, KPMS install, the DSA/rSLDS analysis
extras, the torch CPU/GPU caveats, and reference-data download — lives in
[`docs/installation.md`](docs/installation.md). The short version:

```bash
git clone -b kevin/kpms-decoder https://github.com/talmolab/track-mjx.git
uv venv && source .venv/bin/activate
cd track-mjx && uv pip install -e ".[cuda12]" && cd ..
uv pip install keypoint-moseq[cuda] jax-moseq[cuda]
uv pip install -e .
```

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

### Tokenizer pipelines

```bash
# KPMS-effect sweep: fit 10 KPMS tokenizers (K=100, varying kappa) and train one
# decoder per tokenizer with the identical production config.
bash bash_scripts/pipeline/run_kpms_effect_sweep.sh

# Kinematics-VAE effect sweep: train the best VAE, tokenize at 10 vocabulary
# sizes K, and train one decoder per tokenizer.
bash bash_scripts/pipeline/run_vae_effect_sweep.sh
```

All orchestration scripts: [`docs/bash_scripts.md`](docs/bash_scripts.md).

## Repository layout

See [`docs/codebase_structure.md`](docs/codebase_structure.md) for the
authoritative tree and the location of every output / checkpoint. Related docs:

- [`docs/installation.md`](docs/installation.md) — environment setup.
- [`docs/experiments.md`](docs/experiments.md) — analysis experiment scripts.
- [`docs/bash_scripts.md`](docs/bash_scripts.md) — pipeline / sweep orchestration.
- [`docs/moseq_decoder_config.md`](docs/moseq_decoder_config.md) — config keys.

## License

BSD 3-Clause (see [`LICENSE`](LICENSE)).

## References
- **MIMIC-MJX:** <https://mimic-mjx.talmolab.org/>
- **Keypoint-MoSeq:** <https://keypoint-moseq.readthedocs.io/en/latest/>
- **TopoMimic-annotation** (behavior + direction labels in
  `data/rodent/rodent_labels.csv`):
  <https://github.com/KevinBian107/TopoMimic-annotation>
