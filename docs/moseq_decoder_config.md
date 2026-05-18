# `moseq_decoder.yaml` — config reference

This document describes every key in
[`c2a/configs/moseq_decoder.yaml`](../c2a/configs/moseq_decoder.yaml). The
decoder training pipeline has grown a lot of knobs; this is the single
source of truth. The default config ships `use_distillation_head=true` +
`use_pretrained_decoder=true` — the mode that most recent experiments run.

Hydra overrides: pass `key.subkey=value` on the command line, e.g.
`python -m c2a.train_moseq_decoder network_config.kl_weight=0.1`.

---

## `env_config` — environment

| Key | Type | Default | Notes |
|---|---|---|---|
| `mujoco_impl` | str | `"jax"` | MuJoCo backend. |
| `solver` / `iterations` / `ls_iterations` | solver opts | `"newton"` / `5` / `5` | See mujoco_playground docs. |
| `naconmax`, `njmax` | int | `131072`, `128` | Contact/constraint buffers. |
| `sim_dt` / `ctrl_dt` | float | `0.002` / `0.01` | 5 sim steps per policy step. |
| `mocap_hz` | int | `50` | Reference data frame rate. |
| `clip_length` | int | `250` | Frames per reference clip. |
| `reference_length` / `reference_stride` | int | `5` / `1` | Context window in `imitation_target`. |
| `start_frame_range` | `[a,b]` | `[0, 20]` | Random episode start range. |
| `qvel_init` | str | `"zeros"` | Initial velocity mode. |
| `keep_clips_idx` | null / list[int] | `null` | If set, restrict training to a clip subset. |
| `balanced_split_path` | str / null | `${oc.env:C2A_ROOT,.}/data/rodent/rodent_balanced_splits.json` | Path to train/test split JSON. **If set but missing, training raises** — silent fallback to random splits was removed during the code2act migration to prevent train/test contamination across models. |
| `domain_randomization.*` | — | see yaml | Floor friction, mass scales, joint jitter, etc. |
| `reward_terms.*` | — | see yaml | Per-term reward weights + exp_scale. |
| `termination_criteria.*` | — | see yaml | `root_too_far`, `root_too_rotated`, `pose_error`, `nan_termination`. |

---

## `render_config` — eval rollout rendering

| Key | Type | Default | Notes |
|---|---|---|---|
| `render_camera_name` | str | `close_profile` | MuJoCo camera for eval videos. |
| `render_fps` | int | `50` | MP4 fps. |
| `render_interval` | int | `5` | Render every Nth frame. |
| `render_per_code_videos` | bool | `false` | One video per code. Expensive. |
| `eval_rollouts_for_transition` | int | `16` | Number of rollouts used to compute `moseq/eval_transition_rate`. |
| `num_eval_rollout_videos` | int | `3` | How many eval rollouts become videos. |

---

## `kpms_config` — behavioral codes (any tokenizer family)

The `kpms_config` block historically only carried KPMS settings; it now
dispatches to any of seven sources via `code_source`. The yaml key
remains `kpms_config` for backward compatibility.

| Key | Type | Default | Notes |
|---|---|---|---|
| `code_source` | str | `kpms_raw` | One of `kpms_raw`, `kpms_merged`, `shimoseq`, `kinematics_vae`, `vqvae`, `ps_vqvae`, `gt_directional_turn`. Controls which loader and which provenance label appears in WandB. |
| `codes_path` | str / null | `${oc.env:C2A_ROOT,.}/c2a/outputs/kpms_sweep/best_codes.npz` | `.npz` with `train_codes`/`test_codes` `[n_clips, n_frames]`. **If set but missing, training raises** — the March-2026 silent random-codes fallback was removed. Set to `null` to explicitly request random codes for smoke-testing **only for `kpms_raw` / `kpms_merged` / `shimoseq`** (the kinematics-VAE family rejects null with a clear error to avoid silent codebook-size mismatch). |
| `labels_csv_path` | str / null | `${oc.env:C2A_ROOT,.}/data/rodent/rodent_labels.csv` | Required when `code_source=gt_directional_turn`. |
| `csv_to_h5_subsample` | int | `2` | CSV is at 30fps; H5 at 15fps. |

### `code_source` values

| Value | Writer | Codebook semantics |
|---|---|---|
| `kpms_raw` | `c2a.codegen.kpms.generate_codes` | AR-HMM syllables; `num_states` = sweep K |
| `kpms_merged` | `c2a.codegen.kpms.merge_states.reencode_codes_npz` (requires `is_merged=True` in the npz) | Post-merge AR-HMM; `num_states` = merged K |
| `shimoseq` | `c2a.codegen.shimoseq.generate_codes` | Hierarchical-HMM high-level states |
| `kinematics_vae` | `c2a.codegen.kinematics_vae.vae.tokenize` | K-Means on a trained Gaussian-VAE latent (optional PCA pre-reduce) |
| `vqvae` | `c2a.codegen.kinematics_vae.vqvae.tokenize` | Vanilla VQ codebook argmin |
| `ps_vqvae` | `c2a.codegen.kinematics_vae.ps_vqvae.tokenize` | Per-group EMA-VQ codes folded into a single product index; `num_states = prod(K_g)` |
| `gt_directional_turn` | (loaded directly from rodent_labels.csv) | 5-class GT: Immobile / Rear / Walk / Turn_Left / Turn_Right |

---

## `network_config` — decoder architecture

### Core

| Key | Type | Default | Notes |
|---|---|---|---|
| `arch_name` | str | `moseq_decoder` | For bookkeeping only. |
| `num_codes` | int | `50` | KPMS vocabulary size. Overridden from `codes_path`'s `max()+1` if codes load. |
| `code_embed_dim` | int | `16` | Embedding table width. Total code-embedding input to GRU is `code_stack_size * code_embed_dim`. |
| `code_stack_size` | int | `5` | Number of *consecutive* codes stacked into the obs. `5` = current + 4 lookahead. |
| `decoder_layer_sizes` | list[int] | `[512, 512, 256, 256]` | **FF decoder only**. Ignored when `use_pretrained_decoder=true`. |
| `critic_layer_sizes` | list[int] | `[512, 512, 256, 256]` | Value network MLP. |

### Continuous encoder

| Key | Type | Default | Notes |
|---|---|---|---|
| `use_continuous_encoder` | bool | `true` | Encode `imitation_target` (+ proprio) into `(mean, logvar)` for KL-regularized reparam noise. |
| `encoder_layer_sizes` | list[int] | `[512, 512, 256]` | Encoder MLP. |
| `continuous_latent_dim` | int | `16` | Latent size. |
| `encoder_uses_proprio` | bool | `true` | `true` → encoder sees `[imitation_target, proprio]`. `false` → `imitation_target` only. |
| `kl_weight` | float | `0.05` | Base KL coefficient (used when `kl_schedule="none"`). |
| `z_e_dropout_rate` | float | `0.01` | Per-sample probability of zeroing z_e during training (regularizer that nudges the decoder away from over-reliance on z_e). |

### RNN decoder

| Key | Type | Default | Notes |
|---|---|---|---|
| `use_rnn_decoder` | bool | `true` | `false` falls back to feedforward MLP decoder. |
| `rnn_cell_type` | str | `gru` | Only `"gru"` supported today. |
| `rnn_hidden_sizes` | list[int] | `[512, 512, 256, 256]` | One GRUCell per element. |
| `z_e_at_action_head` | bool | `false` | `true` → z_e skips GRU and concat with GRU output before action head. **Auto-forced to `false` when `use_distillation_head=true`.** |
| `reinit_hidden_on_code` | bool | `false` | `true` → reset GRU hidden on every code transition. |
| `learned_hidden_init` | bool | `false` | `true` + `reinit_hidden_on_code=true` → per-code learned init (param `hidden_init_{i}`, shape `[num_codes, H]`). **KNOWN ASYMMETRY:** rollout (`moseq_ppo.generate_unroll_rnn_moseq`) always zero-resets on episode boundaries, ignoring `learned_hidden_init`; only loss's `apply_sequence` honors the per-code inits. Set both to `false` to keep rollout and loss aligned. |

### Distillation head (prior-distillation pipeline)

| Key | Type | Default | Notes |
|---|---|---|---|
| `use_distillation_head` | bool | `true` | Enable distill-head KL. Adds `_distill_head` MLP `h_t → (mu_d, logvar_d)`. Encoder `(mean, logvar)` are `stop_gradient`'d and serve as the KL target; z_e is **removed** from the action path entirely. Auto-forces `kl_weight=0.0` in the loss (distill KL replaces it). |
| `distillation_head_layer_sizes` | list[int] | `[512, 512, 256, 256]` | MLP sizing. |
| `distill_kl_weight` | float | `1.0` | Weight on `KL(sg(encoder) || distill_head)`. |
| `distillation_encoder_checkpoint` | str / null | `${oc.env:C2A_ROOT,.}/trained_mimic_ckpts/260413_040706_764936/` | Path to a frozen MimicEncoder checkpoint. Required when `use_distillation_head=true`. Architecture must match `encoder_layer_sizes` + `continuous_latent_dim` exactly (the loader splices subtrees by name). |
| `distill_logvar_min` / `distill_logvar_max` | float / null | `null` / `null` | Optional clamps on the distill logvar output. |

### Pretrained-decoder readout

| Key | Type | Default | Notes |
|---|---|---|---|
| `use_pretrained_decoder` | bool | `true` | Replaces the Dense action head with a unified VAE head (`h_t → (mu, logvar) → sample z → [z, proprio] → IntentionDecoder → action_params`). When combined with `use_distillation_head=true`, the VAE head's `(mu, logvar)` ARE the distill prediction — no separate `_distill_head` MLP is instantiated. |
| `decoder_layer_sizes_vae` | list[int] | `[512, 256, 256, 256]` | Layer sizing for the frozen `IntentionDecoder`. **Must match `mimic_encoder.yaml`'s `decoder_layer_sizes`** so the pretrained params load. |

### KL schedule

| Key | Type | Default | Notes |
|---|---|---|---|
| `kl_schedule` | str | `cosine_anneal` | One of `none` / `ramp` / `cosine_anneal`. Auto-forced to `"none"` when `use_distillation_head=true` (distill KL is constant). |
| `kl_schedule_config.start_value` | float | `0.01` | KL weight at schedule start. |
| `kl_schedule_config.end_value` | float | `0.05` | KL weight at schedule end. |
| `kl_schedule_config.start_frac` / `end_frac` | float | `0.0` / `0.5` | Fractions of total training where the schedule ramps (ramp only). |
| `kl_schedule_config.num_cycles` | int | `4` | Cosine-anneal cycle count. |

---

## `train_setup`

| Key | Type | Default | Notes |
|---|---|---|---|
| `agent_name` | str | `rodent` | Walker kind. |
| `run_name` | str | `moseq_decoder_test` | Used for WandB and checkpoint dir names. |
| `checkpoint_to_restore` | str / null | `null` | Absolute path to resume from. |
| `restore_from_run_state` | str / null | `null` | Resume by run ID (track-mjx's `load_from_run_state`). |
| `freeze_decoder` | bool | `true` | Freeze the pretrained `IntentionDecoder` subtree during RL (only the distill head / GRU / action head train). Skip if you want to fine-tune the VAE decoder. |
| `train_test_split_info` | opt | `null` | track-mjx internal. |
| `train_subset_ratio` | float | `0.7` | Fraction of clips for training when `balanced_split_path` is `null`. |
| `eval_every` | int | `20_000_000` | Env steps between eval rollouts. |
| `episode_length` | int | `200` | Env steps per rollout in eval. |
| `reset_every` | int | `20_000_000` | Checkpoint interval. |
| `train_config.*` | various | see yaml | PPO hyperparameters: `num_envs`, `num_timesteps`, `batch_size`, `num_minibatches`, `num_updates_per_batch`, `learning_rate`, `clipping_epsilon`, `discounting`, `action_repeat`, `entropy_cost`, `reward_scaling`, `normalize_observations`, `unroll_length`, `seed`, `get_activation`, `deterministic_eval`. |

---

## `logging_config`

| Key | Type | Default | Notes |
|---|---|---|---|
| `project_name` | str | `moseq_experiments` | WandB project. |
| `group_name` | str | `decoder_rl` | WandB group. |
| `exp_name` | str | `moseq_decoder` | WandB display name. |
| `clip_id` | int | `0` | Debugging clip index. |
| `model_path` | str | `${oc.env:C2A_ROOT,.}/model_checkpoints` | Parent dir for checkpoints. Each run gets a `{TIMESTAMP}_{RUN_NAME}/` subfolder. |

---

## `walker_config`

| Key | Type | Notes |
|---|---|---|
| `walker_name` | `"rodent"` | Only rodent supported today. |
| `torque_actuators` | `true` | |
| `rescale_factor` | `0.9` | Body scale relative to stac-mjx XML. |
| `joint_names`, `body_names`, `end_eff_names` | list[str] | Must match the XML at `c2a/assets/rodent.xml`. |

---

## Auto-override rules (summary)

- `use_distillation_head=true` implies:
  - `kl_weight` is forced to `0.0` in the loss (distill KL replaces it)
  - `z_e_at_action_head` is forced to `false` (z_e is removed from the action path)
  - `kl_schedule` is forced to `"none"`
- `use_pretrained_decoder=true` implies:
  - The VAE head MLP is instantiated and replaces the Dense action head.
  - If `use_distillation_head=true` also, no separate `_distill_head` is built — VAE head does both jobs.
  - `decoder_layer_sizes_vae` MUST match the pretrained checkpoint's decoder layout.
- `use_rnn_decoder=false` disables the whole recurrent path and uses the FF MLP decoder with `decoder_layer_sizes`.
- `e2e_config.e2e_mode=true` implies (see `e2e_config` section below):
  - `kpms_config.code_source` MUST be `"vqvae"` and the loaded `codes_path` MUST come from the same VQ-VAE training run as `pretrained_vqvae_ckpt` (codebook size match is asserted at startup).
  - The policy network's `nn.Embed(kpms_code)` is replaced by the straight-through `z_q` from the live VQ-VAE; the imitation loss flows back into the encoder via STE.
  - `motion_window_stack` of shape `(code_stack_size, window_len, feat_dim)` is added to the obs dict by `MoSeqImitation`.
  - `use_distillation_head` and `use_pretrained_decoder` continue to operate unchanged — they read `task_obs`, orthogonal to the e2e path.

---

## `e2e_config`

| Key | Type | Default | Notes |
|---|---|---|---|
| `e2e_mode` | bool | `false` | Master switch. When false, every other key in this section is ignored. |
| `pretrained_vqvae_ckpt` | str | `${oc.env:C2A_ROOT,.}/outputs/tokenizer/trained_kinematics_vae_ckpts/vqvae/step_0030000` | Directory written by `c2a.codegen.kinematics_vae.vqvae.train` (contains `encoder.msgpack`, `bottleneck.msgpack`, `decoder.msgpack`, `normalizer.npz`, `active_joints.npy`, `meta.json`). Resolved against `C2A_ROOT` when relative. |
| `loss_weights.recon` | float | `0.05` | MSE(decoder(z_st), motion_window_stack). Start small per spec; raise if the recon term collapses to zero or the encoder stops moving. |
| `loss_weights.commit` | float | `0.25` | van den Oord 2017 β — pulls encoder mean toward the chosen codebook entry. |
| `loss_weights.codebook` | float | `1.0` | Pulls codebook entries toward encoder mean; this is the only signal that moves the codebook (gradient-codebook variant). |
| `log_split_grads_at_eval` | bool | `true` | Reserved for the recon-pathway gradient-norm snapshot. Helper exists in `c2a.moseq_losses.compute_e2e_split_grads`; the eval callback hook-in is a follow-up (the live training-step gradient norm into the VQ-VAE is fully captured by the optimizer's update but not currently surfaced as a metric). The recon loss is already logged every step as `e2e/recon_loss`. |
| `vqvae_arch.n_joints` | int | `67` | MUST match the VQ-VAE pretraining `arch.n_joints`. The loader validates via the ckpt's `meta.json` and raises on mismatch. |
| `vqvae_arch.use_qvel` | bool | `false` | Whether the motion features include velocity blocks; same validation. |
| `vqvae_arch.window_len` | int | `10` | Encoder window length `w`. Determines the left-padding of the env's motion buffer. |
| `vqvae_arch.horizon` | int | `5` | Predictor horizon; we don't run the predictor in e2e but the value is stored for provenance. |
| `vqvae_arch.conv_channels` | list[int] | `[64, 128, 256]` | Conv1D channel widths in the encoder backbone. |
| `vqvae_arch.conv_kernel_size` | int | `3` | Conv1D kernel size. |
| `vqvae_arch.conv_head_layers` | list[int] | `[256]` | Shared MLP head after the conv stack. |
| `vqvae_arch.decoder_layer_sizes` | list[int] | `[256, 256, 256]` | Per-branch MLP for the VQ-VAE decoder; must match the ckpt's training. |

### e2e WandB metrics

| Metric | Description |
|--------|-------------|
| `e2e/recon_loss`, `e2e/scaled_recon_loss` | Mean squared error between live-VQ-VAE reconstruction and `motion_window_stack`; the scaled version reflects what the optimizer sees. |
| `e2e/commit_loss`, `e2e/scaled_commit_loss` | Commit term (mean over all encoded steps). |
| `e2e/codebook_loss`, `e2e/scaled_codebook_loss` | Codebook term (mean over all encoded steps). |
| `e2e/live_perplexity` | Codebook perplexity from the live encoder (compare to `moseq/perplexity` which still reflects the pretrained tokenizer). |
| `e2e/live_codebook_utilization` | Fraction of codebook entries used by the live encoder per step. |
| `e2e/live_vs_pretrained_agreement` | Fraction of frames where the live code equals the pretrained code. Drops as the live VQ-VAE drifts; note this conflates **codebook reshuffling** with actual behavioral drift, since reindexing alone changes the integer identity even when the underlying behavior is unchanged. |

### e2e eval video

When `e2e_mode=true`, the eval rendering attaches **two stacked code bars** below the rollout frames using the shared Nature colormap (so identical code indices appear in identical colors across rows):

- **Top row — `Pretrained`** — the precomputed pretrained-VQ-VAE codes from `codes_path` (frozen for the run).
- **Bottom row — `Live`** — the codes emitted by the live VQ-VAE at this evaluation step (changes throughout training as the encoder + codebook are optimized).

Caveat: the live codebook's discrete indices are not anchored to the pretrained codebook's indices. Apparent drift in the bottom row can reflect codebook reshuffling rather than behavioral change — see the agreement metric above and consider a Hungarian post-hoc match before drawing conclusions about "behavior changed."

## WandB metrics

| Metric | Description |
|--------|-------------|
| `moseq/perplexity` | Code usage entropy (higher = more uniform). |
| `moseq/codes_used` | Count of active codes in the eval set. |
| `moseq/codebook_utilization` | Fraction of codes with non-trivial usage. |
| `moseq/eval_transition_rate` | Fraction of steps with a code change. Match ~0.005 for durable KPMS codes; high values (e.g. 0.49) indicate the code-corruption bug pattern — check `kpms_config.codes_path`. |
| `moseq/code_sequence` | WandB image: timeline visualization. |
| `kl_loss`, `scaled_kl_loss` | Continuous-latent KL (zero-weighted when distill head is on). |
| `distill_kl_loss`, `scaled_distill_kl_loss` | `KL(sg(encoder) || distill_head)`, only when `use_distillation_head=true`. |
| `distill_mean_norm`, `distill_logvar_mean` | Diagnostics on the distill prediction. |
| `hidden_state_norm` | Mean GRU hidden L2 norm (RNN decoder only). |
| `z_e_norm`, `z_e_std` | Continuous latent diagnostics. |
| `z_e_scale` | Current z_e scalar multiplier (tracks `z_e_anneal` progress if enabled). |

## Known caveats

1. **Hidden-reset asymmetry (J2 finding):** `generate_unroll_rnn_moseq`
   (rollout) zero-resets hidden on episode boundaries even when
   `learned_hidden_init=true`. The loss's `apply_sequence` honors per-code
   inits. Keep both `reinit_hidden_on_code` and `learned_hidden_init`
   disabled to match rollout and loss exactly — or expect a slight
   distribution shift between the two.
2. **Eval reward bug (pre-existing, RNN decoder only):** `eval/episode_reward`
   metric is broken for the RNN decoder; use `moseq/episode_reward_mean`.
   See `scratch/registry/2026-04-03-rnn-decoder-eval-reward-bug/`.
3. **z_e leakage (pre-existing):** even with distillation + pretrained
   decoder, z_e still accounts for 34–38% of per-step reward in natural
   rollouts. The RNN is not fully autonomous from the encoder.
