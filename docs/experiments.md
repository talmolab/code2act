# Experiments — analysis scripts

Thirteen experiment scripts in `c2a/experiments/` evaluate trained Code2Act
decoders. All commands below run from the `code2act/` repo root with the
venv active.

| # | Script | Config | Purpose |
|---|--------|--------|---------|
| 1 | `run_inference` | `inference.yaml` | Trajectory matching, reward decomposition, K-body videos, transition matrices |
| 2 | `run_code_sequence` | `code_sequence_exp.yaml` | Temporal order of codes, killer demo (instructional/discriminative codes) |
| 3 | `run_code_generation` | `code_generation_exp.yaml` | Generative models (empirical TM, dynamax HMM, ARHMM L2) + free-loop rollouts |
| 4 | `run_generalization` | `generalization.yaml` | Generalization to unseen continuous data via KPMS re-inference |
| 5 | `run_roundtrip` | `roundtrip.yaml` | Round-trip code consistency (qpos → FK → KPMS → codes) |
| 6 | `run_single_code` | `single_code_exp.yaml` | Single-code sustain grid — each code held for K frames |
| 7 | `run_behavior_parade` | `behavior_parade_exp.yaml` | Behavior transition parade — walk→groom→rear |
| 8 | `run_syllable_viz` | `syllable_viz_exp.yaml` | KPMS syllable 3D viz, frequency/duration stats |
| 9 | `run_inception_distance` | `inception_distance_exp.yaml` | FID/KID — generative distribution quality vs real mocap |
| 10 | `run_gait_dynamics` | `gait_dynamics_exp.yaml` | Gait PSD — Code2Act vs MIMIC-MJX walking frequency |
| 11 | `run_hidden_dynamics` | `hidden_dynamics_exp.yaml` | RNN hidden state dynamics — PCA of GRU hidden states per sustained code (kinematic-heuristic-derived 3-class labels: walk / immobility / rear) |
| 12 | `run_hidden_trajectory` | `hidden_trajectory_exp.yaml` | Natural hidden trajectories colored by code |
| 13 | `run_generalization_kid` | `generalization_kid_exp.yaml` | Generalization KID — MIMIC vs C2A distribution quality on unseen data |
| 14 | `run_labeled_hidden_dynamics` | `labeled_hidden_dynamics_exp.yaml` | RNN hidden state dynamics anchored on **manual** behavior + direction labels from `data/rodent/rodent_labels.csv` (TopoMimic-annotation). Replays actual KPMS code sequences from labeled segments — no kinematic heuristic, no sustained-constant-code regime. Three label granularities saved: behavior (4) / direction (3) / combined (up to 12). |

## Prerequisites

- A trained decoder checkpoint. The repo ships two:
  - `model_checkpoints/260407_031233_484020/` — main decoder (distillation + pretrained-decoder)
  - `model_checkpoints/260406_155304_389732/` — backup decoder
- A MIMIC-MJX oracle checkpoint for baselines:
  - `trained_mimic_ckpts/260405_235031_234849/`
  - `trained_mimic_ckpts/260413_040706_764936/` (distill target for the main decoder)
- KPMS codes at `c2a/outputs/kpms_sweep/best_codes.npz`
- Balanced splits at `data/rodent/rodent_balanced_splits.json`

See the full config reference in [`moseq_decoder_config.md`](moseq_decoder_config.md).

## Running experiments

```bash
cd code2act
source .venv/bin/activate

# Example: run inference
python -m c2a.experiments.run_inference

# Override checkpoint
python -m c2a.experiments.run_inference checkpoint.path=/path/to/checkpoint

# Quick test
python -m c2a.experiments.run_inference rendering.K=3
```

All scripts read config from `c2a/experiments/configs/`. Outputs land under
`c2a/outputs/<experiment>/`.

## Notable experiments

### 1. `run_inference`
Runs rollouts on held-out clips, logs reward decomposition (coarse +
fine), generates K-body ghost videos and solo videos, writes transition
matrices.

### 3. `run_code_generation`
Uses four generative models to produce code sequences (empirical transition
matrix, dynamax HMM, ARHMM with L2 regularization, plus a free-loop
model) and rolls them out through the decoder. Outputs comparison metrics.

### 4, 13. `run_generalization*`
Take unseen continuous tracking data (default:
`data/external/transform_art_2020_12_22_1.h5`), re-run KPMS to get
codes for that data, then roll out the decoder on the new codes. Measures
KID / FID distance between Code2Act rollouts and ground-truth new data.

The default generalization H5 path points to an external file
`/home/jovyan/vast/kaiwen/track-mjx/data/transform_art_2020_12_22_1.h5`;
set `C2A_GENERALIZATION_H5` env var if the file lives elsewhere.

### 10. `run_gait_dynamics`
Computes root-trajectory PSD for Code2Act walking rollouts and compares
with MIMIC-MJX baselines — the main gait-fidelity metric.

### 11, 12. `run_hidden_dynamics`, `run_hidden_trajectory`
Record GRU hidden states during rollouts, run PCA, and visualize how
different codes activate different manifolds of hidden space. Used in the
paper's internal-dynamics analysis.

`run_hidden_dynamics` picks **one representative KPMS code per behavior**
by kinematic heuristic (z_rise / xy_disp / xyz_disp) and *sustains it for
250 frames*. This is fast and gives a clean PCA picture but conflates
"what the GRU does in syllable k" with "what the GRU does when the input
is held constant at the embedding of code k". The companion experiment
below (`run_labeled_hidden_dynamics`) uses manual labels and natural
code sequences to address this.

### 14. `run_labeled_hidden_dynamics`
Counterpart to `run_hidden_dynamics` that uses the manual behavior +
direction labels in `data/rodent/rodent_labels.csv` (sourced from
[KevinBian107/TopoMimic-annotation](https://github.com/KevinBian107/TopoMimic-annotation))
instead of kinematic heuristics. For each labeled segment in each
populated label cell:

* Reset the env to that segment's source clip
* **Replay the actual KPMS code sequence** that occurred during the
  segment via `code_override` (rather than holding a single code
  constant)
* Capture hidden / qpos / proprio / codes per timestep

Three label granularities are saved into the same npz, keyed
`hidden_<level>__<label>`, where `level ∈ {behavior, direction,
combined}`:

* `behavior`: 4 classes — Immobile / Rear / Turn / Walk
* `direction`: 3 classes — Left / Right / Straight
* `combined`: up to 12 (but only ~7 cells are populated in the data —
  Immobile is always Straight, Turn is never Straight, Walk is
  essentially always Straight)

The downstream plotting scripts (`plot_hidden_dsa.py`,
`plot_hidden_rslds.py`) auto-detect the labeled-experiment npz and
accept `--level` and `--labels` flags to filter.

Outputs:
* `outputs/labeled_hidden_dynamics/data/hidden_dynamics_labeled.npz` —
  per-label hidden / qpos / proprio / codes / rewards arrays.
* `outputs/labeled_hidden_dynamics/data/video_<level>/labeled_<level>_<label>.mp4`
  — per-label ghost-rendered verification clip.
* Mirror of the npz at `c2a/figures/data/hidden_dynamics_labeled.npz`
  for the plotting scripts.

```bash
# Run the experiment (requires GPU).
python -m c2a.experiments.run_labeled_hidden_dynamics

# Then re-run InputDSA + rSLDS at any granularity:
python -m c2a.figures.plot_hidden_dsa  --level behavior  --output_suffix beh
python -m c2a.figures.plot_hidden_dsa  --level direction --output_suffix dir
python -m c2a.figures.plot_hidden_dsa  --level combined  --output_suffix com
python -m c2a.figures.plot_hidden_rslds --K 4 --level behavior  --output_suffix beh
python -m c2a.figures.plot_hidden_rslds --K 7 --level combined  --output_suffix com
```

## Figures

All publication figures regenerated by scripts in `c2a/figures/`. Each one
imports from `c2a.experiments.shared` for checkpoint loading, metric
computation, and ghost rendering. Defaults point at
`model_checkpoints/260407_031233_484020/` and `c2a/outputs/...`; override
via `--args` where supported.

```bash
python -m c2a.figures.render_behavior_triptych
python -m c2a.figures.plot_hidden_trajectory
python -m c2a.figures.render_single_code_categories
# ...
```

## Known issues

- **`eval/episode_reward` is broken for the RNN decoder**. Use
  `moseq/episode_reward_mean` instead (same bug as in train).
- **Ghost rendering requires a working EGL/OSMesa**. All `render_*` scripts
  set `MUJOCO_GL=egl` at the top.
