# Experiments — analysis scripts

The experiment scripts in `c2a/experiments/` evaluate trained Code2Act
decoders. They are organized into **six theme subfolders**, each with its
own co-located `configs/` directory:

| Folder | Scripts | Theme |
|--------|---------|-------|
| `tokenizer/` | `run_tokenizer_diagnostics`, `run_tokenizer_videos`, `run_roundtrip`, `run_code_distinctness`, `run_code_generation` | Segmenter / code quality (no RL decoder) |
| `behavior/` | `run_inference`, `run_code_sequence`, `run_single_code`, `run_behavior_parade`, `run_generalization` | Decoder rollouts & motion-imitation behavior |
| `fidelity/` | `run_inception_distance`, `run_generalization_kid`, `run_gait_dynamics` | Distribution-quality metrics (FID / KID / PSD) |
| `dynamics/` | `run_hidden_dynamics`, `run_labeled_hidden_dynamics_controlled` | GRU hidden-state helpers (now imported as libraries by the `code_grid/` pipeline; no longer standalone paper analyses) |
| `code_grid/` | `run_code_grid_rollouts`, `run_code_grid_dynamics`, `run_code_grid_behavior_mmd` | Code-grid pipeline (rollouts → merge → dynamics → MMD) — the per-code hidden-dynamics hub |
| `ensemble_manifold/` | `run_population_intention_dynamics`, `run_population_held_codes`, `run_population_label_patterns`, `render_population_*` | Multi-walker ensemble rollouts in (hidden, intention) space + renderers (not in the current paper) |
| `ic_contraction/` | `run_ic_contraction` | **Standalone** — initial-condition contraction / the one-to-many test |
| `perturbation/` | `run_perturbation_recovery` | **Standalone** — perturbation→recovery / codes as *defended* attractors (the interventional test) |

`ic_contraction/` is a standalone analysis (its own folder + `outputs/ic_contraction/`,
not under any theme): pure post-processing (no GPU) of the `run_code_grid_rollouts`
held-code rollouts (100 bodies from diverse poses, code held constant,
decoder-only). Per code, in pose space (root-stripped, yaw-removed qpos) **and**
GRU hidden space, it measures whether bodies *converge*: `σ_inst(t)`
(phase-sensitive cross-body dispersion), `d_manifold(t)` (phase-**invariant**
distance to the shared late-time motif manifold, leave-half-out PCA+kNN), and
contraction ratios `ρ=median(late)/median(early)`, with a **cross-code null**
control. Conditions = the three featured tokenizers on the diverse-IC protocol
(KPMS, multiscale-designed, kinematics-VAE). Result: every condition contracts
(phase-invariant ρ≈0.22–0.37≪1) in both spaces while the null diverges (ρ>1);
Immobile/Rear → fixed points, Walk/Turn → limit cycles (manifold collapses,
instantaneous plateaus on phase). Run:
`python -m c2a.experiments.ic_contraction.run_ic_contraction`. Outputs
`ratio_matrix_*`, `overview_*`, `curves_*`, `summary_*` + `contraction_summary.json`
+ `FINDINGS.md` under `outputs/ic_contraction/default/`.

`perturbation/` is a standalone GPU experiment (its own folder +
`outputs/perturbation/`). It is the *interventional* counterpart to
`ic_contraction`: instead of only varying the initial condition, it lets bodies
converge under a held code and then **injects an external velocity impulse**
(a "shove") mid-rollout, measuring whether the controller drives the body back
onto the code-specific attractor. Two properties make it the experiment the
paper's "executable code" thesis needs. (1) It is a genuine *intervention* —
a descriptive token cannot be perturbed; an executable one can. (2) It is
**non-circular**: the goal pose is defined from the UNPERTURBED (amp=0)
rollouts, independently of the perturbed trajectory, so the body is pushed
*away* from a pre-defined target and must return (IC-contraction, by contrast,
defines the goal from the same rollouts it scores). Decoder-only
(`z_e_scale=0`, deterministic), so recovery is attributable to the code's
learned dynamics. Per (code, amplitude) it rolls out `K` bodies from diverse
poses, applies the impulse at `t_perturb` (after convergence), and measures the
phase-invariant distance `d(t)` to the amp=0 attractor cloud (kNN-to-cloud in
PCA pose space and GRU-hidden space). Controls: an **amplitude sweep** (recovery
vs. shove magnitude maps the basin) and a **wrong-code control** (distance of
the perturbed bodies to a *different* code's attractor stays high — recovery is
code-specific, not a generic righting reflex). Outputs `recovery_curves_*`,
`basin_*` figures + `perturbation_summary.json` (per-code floor / peak / final /
recovery-fraction / recovery-time) under
`outputs/perturbation/perturbation_recovery/<run>/`. Run:
`python -m c2a.experiments.perturbation.run_perturbation_recovery`. The featured
controller is the K=100 Keypoint-MoSeq checkpoint `260611_073323_750085` (the
same controller behind the code-grid / IC-contraction results). `codes=null`
auto-picks the highest-usage code per rollout-derived behavior from the
`labels_run` code-grid run.

Invoke a script with its theme in the module path:
`python -m c2a.experiments.<theme>.<script>` (e.g.
`c2a.experiments.behavior.run_inference`). The shared `c2a/experiments/shared/`
helpers and per-experiment outputs (`outputs/<theme>/<experiment>/`) mirror the
same six-theme structure. All commands below run from the `code2act/` repo root
with the venv active.

| # | Script | Config | Purpose |
|---|--------|--------|---------|
| 1 | `run_inference` | `inference.yaml` | Trajectory matching, reward decomposition, K-body videos, transition matrices |
| 2 | `run_code_sequence` | `code_sequence_exp.yaml` | Temporal order of codes, killer demo (instructional/discriminative codes) |
| 3 | `run_code_generation` | `code_generation_exp.yaml` | Generative models (empirical TM, dynamax HMM, ARHMM L2) + free-loop rollouts |
| 4 | `run_generalization` | `generalization.yaml` | Generalization to unseen continuous data via KPMS re-inference |
| 5 | `run_roundtrip` | `roundtrip.yaml` | Code round-trip reconstruction across all 4 segmenters (kpms, vae, vqvae, ps_vqvae). **Stage 1** — segmenter self-consistency: segment reference clips repeatedly, measure code agreement (accuracy / precision / recall / F1). The 3 autoencoders are deterministic (~100%); KPMS is Gibbs-sampled, run under `same_seed` / `diff_seed` conditions. **Stage 2** — generate-from-real-code round trip: roll out each decoder decoder-only (`z_e_scale=0`), re-segment the generated motion, compare `ĉ` vs `c` (all-frame + transition-masked, since the GRU needs a few frames to settle after a code change). KPMS FK + apply runs in a separate x64 subprocess. |
| 6 | `run_single_code` | `single_code_exp.yaml` | Single-code sustain grid — each code held for K frames |
| 7 | `run_behavior_parade` | `behavior_parade_exp.yaml` | Behavior transition parade — walk→groom→rear |
| 9 | `run_inception_distance` | `inception_distance_exp.yaml` | FID/KID — generative distribution quality vs real mocap |
| 10 | `run_gait_dynamics` | `gait_dynamics_exp.yaml` | Gait PSD — Code2Act vs MIMIC-MJX walking frequency |
| 11 | `run_hidden_dynamics` | `hidden_dynamics_exp.yaml` | **Helper library** (no longer a standalone analysis): exposes `run_rollout_with_hidden`, imported by the `code_grid/` pipeline. Originally PCA of GRU hidden states per sustained code. |
| 13 | `run_generalization_kid` | `generalization_kid_exp.yaml` | Generalization KID — MIMIC vs C2A distribution quality on unseen data (not in the current paper) |
| 15 | `run_labeled_hidden_dynamics_controlled` | `labeled_hidden_dynamics_controlled_exp.yaml` | **Helper library** + controlled 2-pose × 3-sequence diagnostic. Exposes `_pick_start_clips` / `_build_pose_env`, imported by the `code_grid/` pipeline and `run_code_distinctness`. Holds each of the 5 directional-turn codes for a fixed duration, from two controlled starting poses, in **decoder-only mode** (`z_e_scale=0`). |
| 16 | `run_code_grid_rollouts` | `code_grid_rollouts_exp.yaml` | Per-code 5x5 grid videos from controlled poses + MMD merge + label merge + per-code majority-label tagging. See "Code-grid pipeline" below. |
| 17 | `run_code_grid_dynamics` | `code_grid_rollouts_exp.yaml` | DSA + rSLDS over the merged-cluster hidden trajectories from experiment 16. Runs both MMD-merge and label-merge strategies in one pass; cluster tick labels and regime-trace bands carry the majority-label string. |
| 18 | `run_code_grid_behavior_mmd` | `code_grid_behavior_mmd.yaml` | Per-cluster MMD² (RBF) + KID (poly) between each merged cluster's pooled rollout qpos and each canonical labeled GT behavior class (the 12-cell behavior×direction grid). Runs for **both** MMD-merged and label-merged clusters — the labels now come from the TopoMimic auto-labeler over the rollouts (a separate signal from qpos-MMD), so MMD-merge clusters carry a meaningful label hypothesis and the comparison is no longer tautological. Mirrors the TopoVNL `feature_analysis/mmd_kid` flow. |
| 19 | `run_population_intention_dynamics` | `population_intention_dynamics.yaml` | **Population dynamics in (hidden, intention) space** per pure-behavior-label clip combination. For each of 12 combinations whose collapsed-behavior sequence is exactly a target (e.g. ``Walk``, ``Walk→Rear``, ``Imm→Walk→Imm``), pick a qualifying training clip and extract its per-frame code sequence from the segmenter. Run **n=100 walkers** (random frame-0 inits, same pool reused across all 12 combinations) all fed the same code sequence in **decoder-only mode** (``z_e_scale=0``, **deterministic action mode** — previously used sampled actions, fixed 2026-06-07). Collect ``hidden`` (GRU state) and ``intention`` (encoder posterior mean ``cont_mean``, diagnostic-only when ``z_e_scale=0``). Joint PCA-3D across all 12 cells for shared axes. Saves per-cell npz of shape ``(100, T, ...)``. The video pass (separate helper) renders a 3×4 behavior grid (3 walkers per cell) and per-cell 3-column composites (motion ghost / hidden-PCA / intention-PCA) with 3 highlighted trajectories on top of 97 grey background points. |
| 20 | `run_population_label_patterns` | `population_label_patterns.yaml` | **Format-A (held mode code) + Format-B (per-label code pattern) population rollouts** for the v6/v7/v8 multi-init protocol. 12 cells × 2 formats × N walkers, decoder-only (``z_e_scale=0``), deterministic action mode. Init mode is ``same_pos`` (all walkers reset to source-clip frame 0) or ``walking_pose`` (walkers sampled from train clips with a Walk pose at a shared frame). Saves per-(cell, format) npz of qpos / hidden / intention (vae_mean) / fed_code_seq. Companion renderer `render_population_v5_videos` produces a 4-panel composite (motion ghost + hidden PCA-3D + intention PCA-3D + per-frame code timeline). |
## Decoder-only invariants (enforced)

Experiments that drive the policy purely from the discrete code stream
(`run_labeled_hidden_dynamics_controlled`, `run_code_grid_rollouts`,
`run_population_label_patterns`, `run_population_held_codes`,
`run_population_intention_dynamics`, `run_roundtrip` stage 2) MUST satisfy two
invariants, now centralized in **`c2a/experiments/shared/decoder_only.py`**:

1. **`deterministic=True`** — action = distribution **mode**, not a sample.
   Sampled actions inject per-step noise that contaminates dynamics analysis.
   *(`run_population_held_codes` and `run_population_intention_dynamics` used
   `deterministic=False` + `sample_no_postprocessing`; fixed to mode 2026-06-07.)*
2. **Encoder severed** — the frozen MIMIC encoder must contribute nothing to the
   action or GRU. The mechanism is checkpoint-dependent
   (`use_continuous_encoder=False`, `use_distillation_head=True`, or
   `z_e_scale=0`), so a flag check is unreliable.

Helpers (call right after building the inference fn):
- `assert_decoder_only_settings(z_e_scale, deterministic)` — static guard.
- `describe_decoder_only_mode(network_config)` — logs *why* the encoder is severed.
- `verify_encoder_severed_from_env(inf_fn, params, ppo_networks, env, key)` —
  **behavioural** check: perturbs the encoder's only private input
  (`obs["task_obs"]`) and asserts the action + hidden are bit-identical.

Contract test: `c2a/tests/test_decoder_only_check.py`.

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
python -m c2a.experiments.behavior.run_inference

# Override checkpoint
python -m c2a.experiments.behavior.run_inference checkpoint.path=/path/to/checkpoint

# Quick test
python -m c2a.experiments.behavior.run_inference rendering.K=3
```

Each script reads config from its theme's co-located `configs/` directory
(`c2a/experiments/<theme>/configs/`). Outputs land under
`outputs/<theme>/<experiment>/`.

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

### 11. `run_hidden_dynamics` (helper library)
No longer a standalone analysis. It exposes `run_rollout_with_hidden`
(sustain a code, record GRU hidden states), imported by
`code_grid/run_code_grid_rollouts`. The per-code hidden-dynamics analysis
that fed the paper now lives in the **`code_grid/` pipeline** (exp 16–18),
which supersedes the old `run_hidden_trajectory` /
`run_labeled_hidden_dynamics` standalone scripts (both removed) and their
`plot_hidden_*` / `render_hidden_dynamics_*` plotters (removed; only
`plot_hidden_rslds.fit_rslds` survives, imported by `run_code_grid_dynamics`).

### 15. `run_labeled_hidden_dynamics_controlled` (helper library + diagnostic)
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
* `outputs/dynamics/labeled_hidden_dynamics_controlled/data/hidden_<pose>_<seq>.npz`
  — per (pose, sequence): `hidden`, `qpos`, `proprio`, `codes`,
  `rewards` arrays of shape `(K, T, ...)` plus the K reset clip indices.
* `outputs/dynamics/labeled_hidden_dynamics_controlled/data/pca_basis.npz` —
  joint PCA components + variance + axes limits (shared across all 6
  videos so panels are visually comparable).
* `outputs/dynamics/labeled_hidden_dynamics_controlled/videos/<pose>_<seq>.mp4`
  — side-by-side rotating-PCA + ghost-render composite with a
  full-width phase banner at the top.

```bash
python -m c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled

# Quick smoke test (1 s holds, K=2 bodies, no video):
python -m c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled \
    hold_seconds_per_code=1 K_per_rollout=2 render_video=false

# Encoder-on ablation for comparison (z_e contribution restored):
python -m c2a.experiments.dynamics.run_labeled_hidden_dynamics_controlled \
    z_e_scale=1.0 \
    output.base_dir=./outputs/dynamics/labeled_hidden_dynamics_controlled_z1
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
outputs/code_grid/code_grid_rollouts/<run_name>/
├── data/rollout_{walk,rear}_code{000..049}.npz
├── data/usage_rank.npz
├── videos_raw/{walk,rear}_raw_codes_*.mp4
├── merged/clusters_{walk,rear}.json
├── merged/clusters_by_label_{walk,rear}.json
├── merged/distance_{walk,rear}.npz
├── merged/{walk,rear}_merged_clusters_g00.mp4
├── merged/{walk,rear}_merged_by_label_g00.mp4
└── labels/code_labels.json
outputs/code_grid/code_grid_dynamics/<run_name>/
├── {walk,rear}/                       # MMD-merge strategy
└── {walk_label,rear_label}/           # label-merge strategy
outputs/code_grid/code_grid_behavior_mmd/<run_name>/
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
python -m c2a.experiments.code_grid.run_code_grid_rollouts \
    output.run_name=ckpt_260513_062522_599062 \
    checkpoint.path=model_checkpoints/260513_062522_599062/
python -m c2a.experiments.code_grid.run_code_grid_dynamics \
    output.run_name=ckpt_260513_062522_599062 \
    checkpoint.path=model_checkpoints/260513_062522_599062/
python -m c2a.experiments.code_grid.run_code_grid_behavior_mmd \
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
python -m c2a.figures.render_behavior_parade_trail
python -m c2a.figures.render_code_atlas_grouped --run kpms
python -m c2a.figures.visualize_moseq_pipeline
# ...
```

The surviving figure scripts (paper producers) are
`render_paper_figures` + `render_ic_contraction_figure` (under
`c2a/figures/paper/`), `render_behavior_parade_trail`,
`render_codebook_atlas`, `render_code_atlas_grouped`,
`render_perturbation_recovery`, `plot_psd_overlay`,
`plot_kinematic_signatures`, `visualize_moseq_pipeline`, plus
`plot_hidden_rslds` (imported by `run_code_grid_dynamics` for its
`fit_rslds`). The earlier exploratory plotters
(`plot_hidden_{dsa,dynamics,trajectory}`, `render_hidden_dynamics_*`,
`render_behavior_triptych*`, `render_single_code_categories`,
`plot_kid_*`, `plot_roundtrip_accuracy`, `render_generalization_comparison`,
…) were removed once the `code_grid/` pipeline replaced them.

### Codebook atlases (medoid body selection)

`render_codebook_atlas` (hand-designed codebook) and `render_code_atlas_grouped`
(data-discovered tokenizers) both render each code held constant as a
frozen-ghost motion trail, reusing the `run_code_grid_rollouts` held-code
rollouts (no new rollouts). Both pick the **medoid** body — the walker whose
late-window poses sit most densely inside the pooled attractor cloud
(root-stripped, yaw-removed pose space, the IC-contraction features) — i.e. the
body that *actually resembles* what the code produces, rather than the
single best-reward body (which can be an idiosyncratic outlier). Use
`--select best_reward` on `render_codebook_atlas` to restore the old behavior.
`render_code_atlas_grouped` lays out the top-N most-used codes **grouped by modal
behavior** (Fig 6b ordering), making the redundancy visible — one behavior is
realized by several codes (e.g. ~10 Immobile codes). Outputs to
`outputs/figures/codebook_atlas/`.

```bash
python -m c2a.figures.render_codebook_atlas --select medoid          # designed (Fig 10)
python -m c2a.figures.render_code_atlas_grouped --run kpms --top 15 \
    --checkpoint model_checkpoints/260611_073323_750085
python -m c2a.figures.render_code_atlas_grouped --run best_vae --top 15 \
    --checkpoint model_checkpoints/260603_180852_140475
```

### Hidden-state dynamics analysis

The per-code GRU hidden-state dynamics (DSA, jPCA, rSLDS) are produced by
the **`code_grid/` pipeline** (exp 16–18), not by standalone plotters. The
DSA / jPCA figures in the paper come from `run_code_grid_dynamics` →
`c2a/figures/paper/render_paper_figures.py`. The rSLDS fit reuses
`plot_hidden_rslds.fit_rslds` as a library (the only surviving
`plot_hidden_*` module); it needs the optional `[analysis]` deps (DSA +
`ssm`, see [`installation.md`](installation.md)). The earlier standalone
`plot_hidden_{dsa,dynamics,trajectory}` / `render_hidden_dynamics_*`
scripts were removed.

## Known issues

- **`eval/episode_reward` is broken for the RNN decoder**. Use
  `moseq/episode_reward_mean` instead (same bug as in train).
- **Ghost rendering requires a working EGL/OSMesa**. All `render_*` scripts
  set `MUJOCO_GL=egl` at the top.
