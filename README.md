# [Code2Act: Steering Embodied Behavior with Discrete Codes](https://kbian.org/Code2Act-website/)

Computational models of naturalistic behavior fall into two families that do not
meet. Unsupervised segmenters such as Keypoint-MoSeq split continuous behavior
into discrete syllables; they describe what a behavior looks like but not how it
is physically produced. Physics-based reinforcement-learning imitators produce
biomechanically realistic motion but expose no nameable vocabulary: steering one
toward a target behavior means hand-designing a goal-specific reward, a separate
joystick per behavior. A behavioral label can be observed and correlated, but it
cannot be run.

**Code2Act** makes any discrete behavioral code *executable*. Conditioning a
distilled physics controller on a per-frame code stream compiles each symbolic
label into a runnable motor program for a 38-actuator biomechanical rodent,
turning a behavioral vocabulary into a panel of buttons one steers by index, with
no per-behavior reward engineering. The construction is agnostic to where codes
come from: we drive the *same* controller from three interchangeable **steering
strategies** — an unsupervised segmenter (Keypoint-MoSeq), a learned
kinematics-VAE codebook, and a hand-specified set of manually labeled behaviors.
Because the distilled imitation latent encodes a state-dependent *change* signal,
hard motion tracking with distillation forces each code into a *one-to-many* map
that reaches the same goal motion from any initial condition. Across all three
strategies, code-driven behavior is distributionally close to expert motion, each
held code acts as a behaviorally typed attractor the controller actively defends
(reassembling from diverse initial states and restoring after an external
perturbation), and the codes are behaviorally distinct, including for the
hand-specified codebook the controller never co-adapted with. This turns the
study of behavior from observational and statistical into generative,
interventional, and dynamical.

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
