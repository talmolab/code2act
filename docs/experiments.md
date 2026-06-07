# Experiments — analysis scripts

Eighteen experiment scripts in `c2a/experiments/` evaluate trained Code2Act
decoders. All commands below run from the `code2act/` repo root with the
venv active.

| # | Script | Config | Purpose |
|---|--------|--------|---------|
| 1 | `run_inference` | `inference.yaml` | Trajectory matching, reward decomposition, K-body videos, transition matrices |
| 2 | `run_code_sequence` | `code_sequence_exp.yaml` | Temporal order of codes, killer demo (instructional/discriminative codes) |
| 3 | `run_code_generation` | `code_generation_exp.yaml` | Generative models (empirical TM, dynamax HMM, ARHMM L2) + free-loop rollouts |
| 4 | `run_generalization` | `generalization.yaml` | Generalization to unseen continuous data via KPMS re-inference |
| 5 | `run_roundtrip` | `roundtrip.yaml` | Code round-trip reconstruction across all 4 segmenters (kpms, vae, vqvae, ps_vqvae). **Stage 1** — segmenter self-consistency: segment reference clips repeatedly, measure code agreement (accuracy / precision / recall / F1). The 3 autoencoders are deterministic (~100%); KPMS is Gibbs-sampled, run under `same_seed` / `diff_seed` conditions. **Stage 2** — generate-from-real-code round trip: roll out each decoder decoder-only (`z_e_scale=0`), re-segment the generated motion, compare `ĉ` vs `c` (all-frame + transition-masked, since the GRU needs a few frames to settle after a code change). KPMS FK + apply runs in a separate x64 subprocess. |
| 6 | `run_single_code` | `single_code_exp.yaml` | Single-code sustain grid — each code held for K frames |
| 7 | `run_behavior_parade` | `behavior_parade_exp.yaml` | Behavior transition parade — walk→groom→rear |
| 8 | `run_syllable_viz` | `syllable_viz_exp.yaml` | KPMS syllable 3D viz, frequency/duration stats |
| 9 | `run_inception_distance` | `inception_distance_exp.yaml` | FID/KID — generative distribution quality vs real mocap |
| 10 | `run_gait_dynamics` | `gait_dynamics_exp.yaml` | Gait PSD — Code2Act vs MIMIC-MJX walking frequency |
| 11 | `run_hidden_dynamics` | `hidden_dynamics_exp.yaml` | RNN hidden state dynamics — PCA of GRU hidden states per sustained code (kinematic-heuristic-derived 3-class labels: walk / immobility / rear) |
| 12 | `run_hidden_trajectory` | `hidden_trajectory_exp.yaml` | Natural hidden trajectories colored by code |
| 13 | `run_generalization_kid` | `generalization_kid_exp.yaml` | Generalization KID — MIMIC vs C2A distribution quality on unseen data |
| 14 | `run_labeled_hidden_dynamics` | `labeled_hidden_dynamics_exp.yaml` | RNN hidden state dynamics anchored on **manual** behavior + direction labels from `data/rodent/rodent_labels.csv` (TopoMimic-annotation). Replays actual KPMS code sequences from labeled segments — no kinematic heuristic, no sustained-constant-code regime. Three label granularities saved: behavior (4) / direction (3) / combined (up to 12). |
| 15 | `run_labeled_hidden_dynamics_controlled` | `labeled_hidden_dynamics_controlled_exp.yaml` | Controlled 2-pose × 3-sequence diagnostic. Holds each of the 5 directional-turn codes for a fixed duration in a fixed order, from two controlled starting poses (walking / rearing), in **decoder-only mode** (`z_e_scale=0` bypasses the continuous encoder). Produces 6 side-by-side videos with rotating-PCA panel + phase banner. |
| 16 | `run_code_grid_rollouts` | `code_grid_rollouts_exp.yaml` | Per-code 5x5 grid videos from controlled poses + MMD merge + label merge + per-code majority-label tagging. See "Code-grid pipeline" below. |
| 17 | `run_code_grid_dynamics` | `code_grid_rollouts_exp.yaml` | DSA + rSLDS over the merged-cluster hidden trajectories from experiment 16. Runs both MMD-merge and label-merge strategies in one pass; cluster tick labels and regime-trace bands carry the majority-label string. |
| 18 | `run_code_grid_behavior_mmd` | `code_grid_behavior_mmd.yaml` | Per-cluster MMD² (RBF) + KID (poly) between each merged cluster's pooled rollout qpos and each canonical labeled GT behavior class (the 12-cell behavior×direction grid). Runs for **both** MMD-merged and label-merged clusters — the labels now come from the TopoMimic auto-labeler over the rollouts (a separate signal from qpos-MMD), so MMD-merge clusters carry a meaningful label hypothesis and the comparison is no longer tautological. Mirrors the TopoVNL `feature_analysis/mmd_kid` flow. |
| 19 | `run_population_intention_dynamics` | `population_intention_dynamics.yaml` | **Population dynamics in (hidden, intention) space** per pure-behavior-label clip combination. For each of 12 combinations whose collapsed-behavior sequence is exactly a target (e.g. ``Walk``, ``Walk→Rear``, ``Imm→Walk→Imm``), pick a qualifying training clip and extract its per-frame code sequence from the segmenter. Run **n=100 walkers** (random frame-0 inits, same pool reused across all 12 combinations) all fed the same code sequence in **decoder-only mode** (``z_e_scale=0``). Collect ``hidden`` (GRU state) and ``intention`` (encoder posterior mean ``cont_mean``, diagnostic-only when ``z_e_scale=0``). Joint PCA-3D across all 12 cells for shared axes. Saves per-cell npz of shape ``(100, T, ...)``. The video pass (separate helper) renders a 3×4 behavior grid (3 walkers per cell) and per-cell 3-column composites (motion ghost / hidden-PCA / intention-PCA) with 3 highlighted trajectories on top of 97 grey background points. |
| 20 | `run_population_label_patterns` | `population_label_patterns.yaml` | **Format-A (held mode code) + Format-B (per-label code pattern) population rollouts** for the v6/v7/v8 multi-init protocol. 12 cells × 2 formats × N walkers, decoder-only (``z_e_scale=0``), deterministic action mode. Init mode is ``same_pos`` (all walkers reset to source-clip frame 0) or ``walking_pose`` (walkers sampled from train clips with a Walk pose at a shared frame). Saves per-(cell, format) npz of qpos / hidden / intention (vae_mean) / fed_code_seq. Companion renderer `render_population_v5_videos` produces a 4-panel composite (motion ghost + hidden PCA-3D + intention PCA-3D + per-frame code timeline). |
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
Generates code sequences from a number of unconditional sequence priors
and rolls each through the trained Code2Act decoder for survival /
distributional eval. Two classes of generator participate:

**Trainable generators — live in `c2a/ar_codegen/`** (one shared
`CodeGenerator` interface — train once, load by checkpoint path):

| Method key (`generation.methods`) | Model              | Where it lives                            |
|-----------------------------------|--------------------|-------------------------------------------|
| `gpt2`                            | GPT-2 XS (small Flax decoder-only transformer, adapted from `flax/examples/lm1b`) | `c2a/ar_codegen/models/gpt2.py` |
| `mamba`                           | Mamba selective SSM (pure-JAX port of `deep_ssm/mixers/mamba_simple.py`) | `c2a/ar_codegen/models/mamba.py` |
| `s5`                              | S5 SSM (vendored from `lindermanlab/S5`) | `c2a/ar_codegen/models/s5.py` |
| `hmm_pretrained`                  | Categorical HMM via dynamax EM | `c2a/ar_codegen/models/hmm.py` |

Train them via:

```bash
python -m c2a.ar_codegen.train model=gpt2  data.code_source=kpms_raw \
    data.codes_path=c2a/outputs/kpms_sweep/best_codes.npz
python -m c2a.ar_codegen.train model=mamba data.code_source=kpms_raw ...
python -m c2a.ar_codegen.train model=s5    data.code_source=kpms_raw ...
python -m c2a.ar_codegen.train model=hmm   data.code_source=kpms_raw ...
```

Each writes a checkpoint at `outputs/ar_codegen/{model}/{run_id}/` with
`meta.json` + params. Pass the checkpoint paths to the experiment via
`ar_checkpoints.{gpt2,mamba,s5,hmm}=...`.

**Quick generators — inline in `c2a/experiments/run_code_generation.py`**
(no fitting, no checkpoints):

| Method key                          | Model |
|-------------------------------------|-------|
| `transition_matrix` (with `tm_T*`)  | Empirical TM at various sampling temperatures |
| `arhmm_level1`                      | ARHMM transition-only sampling (loaded from the KPMS H5 checkpoint) |
| `arhmm_level2`                      | ARHMM with AR latent dynamics |
| `uniform_random`, `single_code`, `reversed` | Control sequences (baselines) |

**Code-source compatibility.** All seven trainable + quick generators
consume the same per-frame integer code arrays produced by
`c2a.codegen.code_sources.load_codes` (validated as `int32`, values in
`[0, num_codes)`). They are therefore **generic over code source**:

| `data.code_source`     | What it loads                                                  |
|------------------------|----------------------------------------------------------------|
| `kpms_raw`             | Raw KPMS codes from the sweep (`c2a/outputs/kpms_sweep/best_codes.npz`) |
| `kpms_merged`          | KPMS codes after `c2a.codegen.kpms.merge_states` (requires `is_merged=True` in the .npz) |
| `gt_directional_turn`  | 5-class manual labels from `data/rodent/rodent_labels.csv` (Immobile / Rear / Walk / Turn_Left / Turn_Right) |
| (`null` for AR models) | Random codes — smoke test only                                 |

The same `GPT2Generator` / `MambaGenerator` / `S5Generator` /
`HMMGenerator` class trains and samples on any of these — the only
thing that changes between fits is the codebook size (`num_codes` =
2..number-of-syllables-in-the-source). See `c2a/ar_codegen/README.md`
for the full training → sampling → experiment workflow.

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
instead of kinematic heuristics. For each populated directional-turn
label, the script picks the **agreement KPMS code** (mode of the code
histogram across the label's frames) and sustains that code for
`target_steps` control steps × `K_per_label` starting body positions,
capturing hidden / qpos / proprio / codes per timestep.

Label scheme (5 classes, mirrors `c2a.codegen.code_sources.gt_directional_turn`):

* `Immobile`   (any direction collapsed)
* `Rear`       (any direction collapsed)
* `Walk`       (any direction collapsed)
* `Turn_Left`  (Turn × Left)
* `Turn_Right` (Turn × Right)

`Turn_Straight` is absent in the current CSV and stays UNLABELED.

The downstream plotting scripts (`plot_hidden_dsa.py`,
`plot_hidden_rslds.py`, `plot_hidden_dynamics.py`) read the
experiment's npz directly. They auto-detect "legacy mode" (single
underscores, no level prefix) so the 5-class output works without any
flags beyond `--npz` and `--output_suffix`. The `--level` flag still
exists on `plot_hidden_dsa.py` and `plot_hidden_rslds.py` for the
deprecated 12-class `hidden_<level>__<label>` schema; it is **not**
needed for the current 5-class output and should be omitted.

Outputs:

* `outputs/labeled_hidden_dynamics/data/hidden_dynamics_labeled.npz` —
  per-label hidden / qpos / proprio / codes / rewards arrays, keyed
  `hidden_<L>` / `qpos_<L>` / ... for `L` in the 5-class set above.
* `outputs/labeled_hidden_dynamics/data/labeled_<L>.mp4` — one
  ghost-rendered verification clip per populated label.
* Mirror of the npz at `c2a/figures/data/hidden_dynamics_labeled.npz`
  so the plotting scripts find it without `--npz`.

```bash
# Run the experiment (requires GPU).
python -m c2a.experiments.run_labeled_hidden_dynamics

# Run DSA / InputDSA + rSLDS on the labeled output. The --npz flag is
# required because c2a/figures/data/ may also contain a legacy
# hidden_dynamics.npz from run_hidden_dynamics; without --npz the
# plotters would silently prefer that older file. --output_suffix
# isolates the new outputs so they do not overwrite earlier 3-class
# figures.
python -m c2a.figures.plot_hidden_dsa \
    --npz c2a/figures/data/hidden_dynamics_labeled.npz \
    --output_suffix 5class

python -m c2a.figures.plot_hidden_rslds \
    --npz c2a/figures/data/hidden_dynamics_labeled.npz \
    --output_suffix 5class --K 5

# PCA-3D scatter / trajectory plots (same npz, same suffix conventions):
python -m c2a.figures.plot_hidden_dynamics \
    --npz c2a/figures/data/hidden_dynamics_labeled.npz \
    --output_suffix 5class
```

### 15. `run_labeled_hidden_dynamics_controlled`
Controlled diagnostic. For each pair of (starting pose, code sequence),
hold each code for a fixed duration (default 5 s × 5 codes = 25 s per
rollout) with K=10 bodies starting from the same pose-type, in
**decoder-only mode** (`z_e_scale = 0.0`) so the continuous encoder is
bypassed and the GRU receives only `[code_embedding, 0, proprio]`. The
proprio loop drives the autoregressive dynamics; the only varying input
is the per-frame code.

Matrix: 2 starting poses (walking / rearing) × 3 code sequences = 6
rollouts:

- `walk → Turn_Right → Turn_Left → Walk → Rear`
- `Rear → Walk → Turn_Right → Turn_Left → Walk`
- `Immobile → Walk → Turn_Right → Turn_Left → Walk`

Starting pose is selected via the labels CSV — K=10 test clips whose
directional-turn label at the configured probe frame (default frame 20)
matches the requested pose (Walk for walking-pose, Rear for
rearing-pose). The env is built with `start_frame_range = [probe_frame,
probe_frame]` so the rodent resets at exactly the labeled frame.

Outputs:
* `outputs/labeled_hidden_dynamics_controlled/data/hidden_<pose>_<seq>.npz`
  — per (pose, sequence): `hidden`, `qpos`, `proprio`, `codes`,
  `rewards` arrays of shape `(K, T, ...)` plus the K reset clip indices.
* `outputs/labeled_hidden_dynamics_controlled/data/pca_basis.npz` —
  joint PCA components + variance + axes limits (shared across all 6
  videos so panels are visually comparable).
* `outputs/labeled_hidden_dynamics_controlled/videos/<pose>_<seq>.mp4`
  — side-by-side rotating-PCA + ghost-render composite with a
  full-width phase banner at the top.

```bash
python -m c2a.experiments.run_labeled_hidden_dynamics_controlled

# Quick smoke test (1 s holds, K=2 bodies, no video):
python -m c2a.experiments.run_labeled_hidden_dynamics_controlled \
    hold_seconds_per_code=1 K_per_rollout=2 render_video=false

# Encoder-on ablation for comparison (z_e contribution restored):
python -m c2a.experiments.run_labeled_hidden_dynamics_controlled \
    z_e_scale=1.0 \
    output.base_dir=./outputs/labeled_hidden_dynamics_controlled_z1
```

### 16, 17, 18. Code-grid pipeline (`run_code_grid_*`)

Tokenizer-agnostic battery for any RNN-decoder checkpoint with a
discrete code source (VAE + k-means, VQ-VAE, PS-VQ-VAE, etc).

**Three stages, run in order:**

1. **`run_code_grid_rollouts`** — for each of the top-N codes (by usage),
   roll out K bodies in decoder-only mode (`z_e_scale=0`) holding that
   code for `hold_frames` steps, from two controlled starting poses
   (walking / rearing). Per pose:
   - Save raw 5x5 grid videos (codes 0..24 and 25..49) under
     `videos_raw/`.
   - MMD-cluster the codes by behavioral similarity of the resulting
     qpos trajectories (UPGMA + elbow cutoff) → `clusters_{pose}.json`.
   - Label each code by running the vendored TopoMimic auto-labeler
     (`c2a/experiments/shared/topomimic_labeler/`, mirror of
     [`KevinBian107/TopoMimic-annotation`](https://github.com/KevinBian107/TopoMimic-annotation))
     on the rolled-out qpos. Per body: resample 100 → 30 fps, run the
     5-class behavior heuristic + the 3-class direction heuristic with
     Turn-reconciliation, pick the **longest behavior segment** (Groom →
     Immobile to match c2a's 4-class corpus), assign **majority
     direction within that span**. Per code: one-body-one-vote across all
     K bodies of both walking and rearing rollouts. Output:
     `labels/code_labels.json`. This labels what the decoder *produces*
     for each code, not where the tokenizer places each code in the
     corpus — the two diverge in interesting ways (a code whose
     corpus-frame label is Immobile may actually rollout as Turn_L).
   - Build label-based clusters: codes sharing the same majority label
     are grouped → `clusters_by_label_{pose}.json`. The partition is
     pose-agnostic but is written under both poses for downstream
     symmetry with the MMD clusters.
   - Render merged-grid videos for both strategies — file prefixes
     `merged_clusters` (MMD) and `merged_by_label` (label). Per-panel
     banners carry the cluster's majority-label string.

2. **`run_code_grid_dynamics`** — reads the cached rollouts and clusters,
   PCA-reduces hidden trajectories, and runs DSA + rSLDS per pose, for
   **both** strategies:
   - `dyn_base/{walk,rear}/`         — MMD merge clusters.
   - `dyn_base/{walk_label,rear_label}/` — label merge clusters.

   Each subdir has `dsa_distance.npz`, `dsa_heatmap.{png,pdf}` (tick
   labels include the cluster's majority label, e.g.
   `c3 [Walk_L] (n=7)`), `rslds_fit.pkl`, `rslds_z_per_trial.npz`,
   `rslds_elbos.npy`, `regime_traces.png` (each cluster band annotated
   with its label on the right margin), and `composition.json` (cluster
   id → sub-codes + usage + majority label).

   rSLDS is reloaded from pkl when `reuse_dynamics=true` (default) and
   the cached cluster labeling still matches the current clusters JSON;
   otherwise it refits. The pkl reload requires the `ssm` package to be
   importable in the venv.

3. **`run_code_grid_behavior_mmd`** — per-cluster behavior fidelity vs
   real labeled data. Tests both strategies × both poses: MMD-merged
   clusters now also carry a meaningful label hypothesis (the rollout-
   derived TopoMimic label of the constituent codes), so MMD-vs-GT-
   class is non-tautological. For each (strategy, pose) cell:
   - Build a "cluster window stack" by pooling all (code, body) rollout
     qpos in the cluster, root-stripped (drop first 3 dims) and uniformly
     resampled to K=100 frames.
   - For each canonical GT class (Immobile, Rear, Walk_S/L/R,
     Turn_S/L/R), build the real-data "GT window stack" from the
     `(behavior, direction)` segments in `rodent_labels.csv` ×
     `rodent_reference_clips.h5`, same K=100 frames after root strip.
   - Compute KID (polynomial kernel) + RBF-MMD² (σ = median heuristic on
     the pooled X ∪ Y), unbiased estimator. Mirrors the TopoVNL
     `topo_mimic.analysis.feature_analysis.mmd_kid` flow.
   - Save per (strategy, pose):
     - `results.csv` — long form: row per (cluster, GT class) with KID,
       RBF-MMD², σ, n_cluster, n_gt, cluster_label, cluster_codes.
     - `best_match.csv` — argmin RBF-MMD² GT class per cluster, plus a
       boolean `majority_label_matches_best` for quick scoring.
     - `heatmap_mmd_rbf.{png,pdf}`, `heatmap_kid.{png,pdf}` — clusters
       (rows) × GT classes (cols).
     - `matrices.npz` — raw matrices for downstream tooling.
     - `gt_class_sizes.json` — N actually used per GT class.

**Outputs:**

```
outputs/code_grid_rollouts/<run_name>/
├── data/rollout_{walk,rear}_code{000..049}.npz
├── data/usage_rank.npz
├── videos_raw/{walk,rear}_raw_codes_*.mp4
├── merged/clusters_{walk,rear}.json
├── merged/clusters_by_label_{walk,rear}.json
├── merged/distance_{walk,rear}.npz
├── merged/{walk,rear}_merged_clusters_g00.mp4
├── merged/{walk,rear}_merged_by_label_g00.mp4
└── labels/code_labels.json
outputs/code_grid_dynamics/<run_name>/
├── {walk,rear}/                       # MMD-merge strategy
└── {walk_label,rear_label}/           # label-merge strategy
outputs/code_grid_behavior_mmd/<run_name>/
├── mmd_merge/{walk,rear}/   # MMD-merged clusters vs GT classes
└── label_merge/{walk,rear}/ # label-merged clusters vs GT classes
```

**Default `run_name`** is `default`. For multi-checkpoint sweeps,
override per checkpoint, e.g.
`output.run_name=ckpt_260513_062522_599062`.

**Reuse flags** (rollouts script): `reuse_rollouts=true` skips per-code
NPZ generation if cached, `reuse_merge=true` skips MMD distance compute,
`reuse_labels=true` skips the label cross-tab. The behavior MMD script
has no reuse flag — it always recomputes from the cached rollouts +
clusters (fast since no environment build is needed).

**Tokenizer compatibility.** The code source (KPMS, VAE+kmeans, VQ-VAE,
PS-VQVAE) is read from the checkpoint's saved config. The number of
codes is detected automatically. PS-VQVAE flattens its product code
indices through `c2a.codegen.code_sources.load_codes` so the experiment
sees a single integer code per frame regardless of the source.

```bash
# VAE checkpoint
python -m c2a.experiments.run_code_grid_rollouts \
    output.run_name=ckpt_260513_062522_599062 \
    checkpoint.path=model_checkpoints/260513_062522_599062/
python -m c2a.experiments.run_code_grid_dynamics \
    output.run_name=ckpt_260513_062522_599062 \
    checkpoint.path=model_checkpoints/260513_062522_599062/
python -m c2a.experiments.run_code_grid_behavior_mmd \
    rollouts.run_name=ckpt_260513_062522_599062 \
    output.run_name=ckpt_260513_062522_599062
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

### Hidden-state visualization + analysis scripts

These four read the npz produced by experiment 11
(`run_hidden_dynamics`) or experiment 14 (`run_labeled_hidden_dynamics`).
All accept `--npz` and `--output_suffix`; for the labeled-experiment
output, pass `--npz c2a/figures/data/hidden_dynamics_labeled.npz`
explicitly so the loader does not silently prefer a legacy
`hidden_dynamics.npz` if both exist.

* `plot_hidden_dynamics.py` — PCA-3D scatter + trajectory plots of the
  hidden state per label. Outputs `hidden_scatter_pca3d{,_combined}*`
  and `hidden_trajectories_pca3d{,_combined}*`. The cheapest
  visualization; useful as a sanity check before the heavier DSA / rSLDS
  fits.
* `plot_hidden_trajectory.py` — single-trial trajectory walkthrough in
  PCA-3D with code colour-coding. Outputs `hidden_trajectory_pca3d*`.
* `plot_hidden_dsa.py` — DSA + per-trial-centered DSA + InputDSA
  (qpos as control). Outputs `dsa_vanilla_heatmap*`,
  `dsa_centered_heatmap*`, `dsa_inputdsa_heatmap*`,
  `dsa_within_between*`, `dsa_rank_sweep*`. **Requires the optional
  `[analysis]` deps** (see README "Analysis dependencies").
* `plot_hidden_rslds.py` — recurrent SLDS fit with K discrete regimes
  on the pooled hidden trajectories. Outputs `rslds_confusion*`,
  `rslds_regime_traces*`, `rslds_summary*`. Default K=5 matches the
  current directional-turn label scheme. Also needs the `[analysis]`
  deps.

The `--level` flag on `plot_hidden_dsa.py` and `plot_hidden_rslds.py`
is for the deprecated 12-class `hidden_<level>__<label>` npz schema
and should **not** be passed for the current 5-class output.

## Known issues

- **`eval/episode_reward` is broken for the RNN decoder**. Use
  `moseq/episode_reward_mean` instead (same bug as in train).
- **Ghost rendering requires a working EGL/OSMesa**. All `render_*` scripts
  set `MUJOCO_GL=egl` at the top.
