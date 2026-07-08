#!/usr/bin/env bash
# =============================================================================
# run_mlb_phaseB.sh — Phase-B orchestration for the mlb (manually-labeled)
# steering strategy.
#
# Runs the FULL experiment + figure + demo pipeline for the mlb controller,
# mirroring exactly what was already done for kpms and kvae (best_vae), and
# pins the checkpoint to STEP 30 (training keeps running past 30, so "load
# latest" would be WRONG).
#
# STEP-30 PIN MECHANISM
# ---------------------
# The eval scripts call `load_moseq_checkpoint(cfg.checkpoint.path)` WITHOUT
# threading `cfg.checkpoint.step`, so they ALWAYS restore orbax `latest_step()`.
# We therefore pin step 30 by building a *snapshot* checkpoint directory that
# contains ONLY `MoSeqPPONetwork_30`; orbax `latest_step()` on that dir returns
# 30. The same pinned path is handed to every experiment, figure and demo, so
# the pin is uniform and code-change-free. (Validated: a snapshot with only
# `MoSeqPPONetwork_10` returns latest_step()=10.)
#
# USAGE
#   bash bash_scripts/training/run_mlb_phaseB.sh                # all stages
#   STAGES="1 2 3" bash bash_scripts/training/run_mlb_phaseB.sh # subset
#   MLB_ROUNDTRIP=0 ... # disable stage 7. Default is 1 now that
#                       #   run_code_roundtrip has the `--segmenter mlb` path
#                       #   (see MLB_RUNBOOK.md — READY).
#   MLB_AR_CODEGEN=1 ...# also train gpt2+mamba on mlb codes so the KID table
#                       #   Mamba/GPT columns fill (optional parity; heavy).
#
# Each stage logs to outputs/demos/_phaseB_logs/<stage>.log and prints a banner.
# Experiment reuse flags are ON, so re-invoking the script skips already-cached
# rollouts/merges/dynamics (idempotent).
# =============================================================================
set -u -o pipefail

# ----------------------------------------------------------------------------
# Environment (mirrors docs/demos.md + train_moseq_decoder.py module-time env)
# ----------------------------------------------------------------------------
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export C2A_ROOT="$PWD"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

PYRUN="python3.12 bash_scripts/_pyrun.py"
LOGDIR="outputs/demos/_phaseB_logs"
mkdir -p "$LOGDIR"

# ----------------------------------------------------------------------------
# Constants — the mlb checkpoint, its step-30 pin snapshot, and its codes
# ----------------------------------------------------------------------------
CKPT_BASE="model_checkpoints/260629_212713_669229"
# Launched early at the user's request (step 27, latest complete) rather than 30.
# Env-overridable: PIN_STEP=NN bash run_mlb_phaseB.sh
PIN_STEP="${PIN_STEP:-27}"
# ABSOLUTE path — orbax restore (behavior_parade/roundtrip/gait/KID loaders)
# rejects relative checkpoint paths ("Checkpoint path should be absolute").
CKPT_PIN="$PWD/${CKPT_BASE}_step${PIN_STEP}"   # snapshot dir => latest_step()==PIN_STEP
MLB_CODES="outputs/tokenizer/mlb_v1/best_codes.npz"
RUN="mlb"

# Optional / blocked stages
MLB_ROUNDTRIP="${MLB_ROUNDTRIP:-1}"          # stage 7 (READY: --segmenter mlb exists)
MLB_AR_CODEGEN="${MLB_AR_CODEGEN:-1}"        # stage 9 Mamba/GPT columns (heavy)

# Which stages to run (default all)
STAGES="${STAGES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13}"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
want() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

banner() {
  echo ""
  echo "############################################################"
  echo "# STAGE $1 — $2"
  echo "# $(date -u +%FT%TZ)"
  echo "############################################################"
}

# run_stage <num> <name> <logfile> <cmd...>
# Runs cmd in a subshell with `set -e`, tees to the per-stage log, and aborts
# the whole orchestration on failure (so the main loop sees the real status).
run_stage() {
  local num="$1" name="$2" log="$3"; shift 3
  banner "$num" "$name"
  echo "+ $*" | tee "$LOGDIR/$log"
  ( set -e; "$@" ) 2>&1 | tee -a "$LOGDIR/$log"
  local rc="${PIPESTATUS[0]}"
  if [ "$rc" -ne 0 ]; then
    echo "!! STAGE $num ($name) FAILED rc=$rc — see $LOGDIR/$log" | tee -a "$LOGDIR/$log"
    exit "$rc"
  fi
  echo "== STAGE $num ($name) OK" | tee -a "$LOGDIR/$log"
}

copy_fig() {  # copy_fig <src-glob...> -> paper/figures/
  for f in "$@"; do
    [ -e "$f" ] && cp -f "$f" paper/figures/ && echo "  copied $f -> paper/figures/"
  done
}

# ----------------------------------------------------------------------------
# STAGE 0 — build the step-30 pin snapshot
# ----------------------------------------------------------------------------
if want 0; then
  banner 0 "pin checkpoint to step ${PIN_STEP}"
  {
    if [ ! -d "$CKPT_PIN/MoSeqPPONetwork_${PIN_STEP}" ]; then
      if [ ! -d "$CKPT_BASE/MoSeqPPONetwork_${PIN_STEP}" ]; then
        echo "FATAL: $CKPT_BASE/MoSeqPPONetwork_${PIN_STEP} does not exist yet."
        echo "       Training has not reached step ${PIN_STEP}. Aborting."
        exit 2
      fi
      mkdir -p "$CKPT_PIN"
      cp -r "$CKPT_BASE/MoSeqPPONetwork_${PIN_STEP}" "$CKPT_PIN/"
      echo "snapshot built: $CKPT_PIN/MoSeqPPONetwork_${PIN_STEP}"
    else
      echo "snapshot already present: $CKPT_PIN"
    fi
    # Verify orbax latest_step() == PIN_STEP on the snapshot.
    python3.12 - "$CKPT_PIN" "$PIN_STEP" <<'PY'
import sys, site, os
site.addsitedir(os.path.join(os.environ["C2A_ROOT"], ".venv", "lib", "python3.12", "site-packages"))
import jax
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None
import orbax.checkpoint as ocp
p, want = sys.argv[1], int(sys.argv[2])
m = ocp.CheckpointManager(p, options=ocp.CheckpointManagerOptions(create=False, step_prefix="MoSeqPPONetwork"))
got = m.latest_step()
print(f"latest_step()={got} (want {want}) all_steps={m.all_steps()}")
assert got == want, f"PIN FAILED: latest_step()={got} != {want}"
print("PIN OK")
PY
  } 2>&1 | tee "$LOGDIR/00_pin.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!! STAGE 0 FAILED"; exit 1; }
fi

# ----------------------------------------------------------------------------
# STAGE 1 — code_grid_rollouts (per-code held-code rollouts + merge + labels)
#           Produces outputs/code_grid/code_grid_rollouts/mlb/{data,merged,labels}
# ----------------------------------------------------------------------------
if want 1; then
  run_stage 1 "code_grid_rollouts" 01_code_grid_rollouts.log \
    $PYRUN c2a.experiments.code_grid.run_code_grid_rollouts \
      checkpoint.path="$CKPT_PIN" \
      checkpoint.codes_path_override="$MLB_CODES" \
      output.run_name="$RUN" \
      reuse_rollouts=true reuse_merge=true reuse_labels=true \
      render_video=false
fi

# ----------------------------------------------------------------------------
# STAGE 2 — code_grid_dynamics (DSA + jPCA + InputDSA over hidden/joint)
#           Produces outputs/code_grid/code_grid_dynamics/mlb/{random_raw,...}
# ----------------------------------------------------------------------------
if want 2; then
  run_stage 2 "code_grid_dynamics" 02_code_grid_dynamics.log \
    $PYRUN c2a.experiments.code_grid.run_code_grid_dynamics \
      checkpoint.path="$CKPT_PIN" \
      output.run_name="$RUN" \
      reuse_dynamics=true
fi

# ----------------------------------------------------------------------------
# STAGE 3 — code_grid_behavior_mmd (per-cluster MMD/KID vs GT behavior classes)
# ----------------------------------------------------------------------------
if want 3; then
  run_stage 3 "code_grid_behavior_mmd" 03_behavior_mmd.log \
    $PYRUN c2a.experiments.code_grid.run_code_grid_behavior_mmd \
      rollouts.run_name="$RUN" \
      output.run_name="$RUN"
fi

# ----------------------------------------------------------------------------
# STAGE 4 — ic_contraction (CPU post-processing; add the mlb condition)
#           Writes outputs/ic_contraction/default/ (kpms + best_vae + mlb).
# ----------------------------------------------------------------------------
if want 4; then
  run_stage 4 "ic_contraction" 04_ic_contraction.log \
    $PYRUN c2a.experiments.ic_contraction.run_ic_contraction \
      'conditions=[{name:kpms,display:Keypoint-MoSeq,group:random},{name:best_vae,display:Kinematics-VAE,group:random},{name:mlb,display:Manually-labeled,group:random}]'
fi

# ----------------------------------------------------------------------------
# STAGE 5 — perturbation_recovery (kick→recovery; auto-pick codes from mlb labels)
#           Writes outputs/perturbation/perturbation_recovery/mlb/
# ----------------------------------------------------------------------------
if want 5; then
  run_stage 5 "perturbation_recovery" 05_perturbation.log \
    $PYRUN c2a.experiments.perturbation.run_perturbation_recovery \
      checkpoint.path="$CKPT_PIN" \
      checkpoint.codes_path_override="$MLB_CODES" \
      output.run_name="$RUN" \
      labels_run="$RUN" \
      'codes=[0,1,2,3,4,5,6,7,8]' \
      reuse_rollouts=true
fi

# ----------------------------------------------------------------------------
# STAGE 6 — behavior_parade (walk→groom→rear); writes to mlb_parade/walk_groom_rear
# ----------------------------------------------------------------------------
if want 6; then
  run_stage 6 "behavior_parade" 06_parade.log \
    $PYRUN c2a.experiments.behavior.run_behavior_parade \
      checkpoint.path="$CKPT_PIN" \
      data.codes_path="$MLB_CODES" \
      output.base_dir=./outputs/behavior/mlb_parade
fi

# ----------------------------------------------------------------------------
# STAGE 7 — roundtrip (D5: mlb re-tokenized by its OWN kinematic labeler)
#           READY: run_code_roundtrip supports --segmenter mlb (driving = the
#           held mlb code's definitional 9-class category; re-inferred = the
#           TopoMimic auto-labeler over each body's settled window x net-yaw).
#           Enabled by default (MLB_ROUNDTRIP=1); set MLB_ROUNDTRIP=0 to skip.
# ----------------------------------------------------------------------------
if want 7; then
  if [ "$MLB_ROUNDTRIP" = "1" ]; then
    run_stage 7 "roundtrip (mlb segmenter)" 07_roundtrip.log \
      $PYRUN c2a.experiments.ic_contraction.run_code_roundtrip \
        --conditions mlb \
        --segmenter mlb \
        --rollouts-base outputs/code_grid/code_grid_rollouts \
        --pose random --n-bodies 60 \
        --out outputs/ic_contraction/code_roundtrip/mlb \
        --reuse
  else
    banner 7 "roundtrip — SKIPPED (MLB_ROUNDTRIP=0)"
    echo "stage 7 disabled via MLB_ROUNDTRIP=0; --segmenter mlb is available, see MLB_RUNBOOK.md" \
      | tee "$LOGDIR/07_roundtrip.log"
  fi
fi

# ----------------------------------------------------------------------------
# STAGE 8 — gait_dynamics (hind-limb PSD vs MIMIC-MJX)
#           Writes outputs/fidelity/moseq_gait_dynamics_mlb/
# ----------------------------------------------------------------------------
if want 8; then
  run_stage 8 "gait_dynamics" 08_gait.log \
    $PYRUN c2a.experiments.fidelity.run_gait_dynamics \
      checkpoint.path="$CKPT_PIN" \
      data.codes_path="$MLB_CODES" \
      output.base_dir=./outputs/fidelity/moseq_gait_dynamics_mlb
fi

# ----------------------------------------------------------------------------
# STAGE 9 — inception_distance (KID). Source-agnostic methods only; the
#           Mamba/GPT KID columns require AR-codegen training on mlb codes
#           (MLB_AR_CODEGEN=1, optional). Writes moseq_inception_distance_mlb/.
# ----------------------------------------------------------------------------
if want 9; then
  AR_OVERRIDES=()
  if [ "$MLB_AR_CODEGEN" = "1" ]; then
    # Optional: train gpt2 + mamba on the mlb codes for KID parity with kvae.
    run_stage 9 "ar_codegen gpt2 (mlb)" 09a_ar_gpt2.log \
      $PYRUN c2a.ar_codegen.train model=gpt2 data.code_source=mlb \
        data.codes_path="$MLB_CODES"
    run_stage 9 "ar_codegen mamba (mlb)" 09b_ar_mamba.log \
      $PYRUN c2a.ar_codegen.train model=mamba data.code_source=mlb \
        data.codes_path="$MLB_CODES"
    GPT_CKPT="$(ls -td outputs/ar_codegen/gpt2/*/ 2>/dev/null | head -1)"
    MAMBA_CKPT="$(ls -td outputs/ar_codegen/mamba/*/ 2>/dev/null | head -1)"
    AR_OVERRIDES=(
      "inception_distance.methods=[decoder_original_codes,mimic_mjx,transition_matrix,gpt2,mamba,uniform_random]"
      "ar_checkpoints.gpt2=$GPT_CKPT"
      "ar_checkpoints.mamba=$MAMBA_CKPT"
    )
  else
    AR_OVERRIDES=(
      "inception_distance.methods=[decoder_original_codes,mimic_mjx,transition_matrix,uniform_random]"
    )
  fi
  run_stage 9 "inception_distance (KID)" 09_inception.log \
    $PYRUN c2a.experiments.fidelity.run_inception_distance \
      checkpoint.path="$CKPT_PIN" \
      data.codes_path="$MLB_CODES" \
      output.base_dir=./outputs/fidelity/moseq_inception_distance_mlb \
      "${AR_OVERRIDES[@]}"
fi

# ----------------------------------------------------------------------------
# STAGE 10 — code_distinctness (within/between MMD, NMI spread; pose=random)
#            Reads code_grid_rollouts/mlb; codes_path drives the P3 NMI.
# ----------------------------------------------------------------------------
if want 10; then
  run_stage 10 "code_distinctness" 10_distinctness.log \
    $PYRUN c2a.experiments.tokenizer.run_code_distinctness \
      output.run_name="$RUN" \
      codes_path="$MLB_CODES" \
      'poses=[random]'

  # 10b — killer kinematic data for plot_kinematic_signatures (--run mlb).
  # run_code_sequence.main() needs a MIMIC ckpt; the killer demo alone does
  # not, so we drive run_killer_demo directly (mirrors the kvae driver noted in
  # scratch/.../README.md). Throwaway driver lives in /tmp per repo convention.
  KILLER_DRIVER="/tmp/run_killer_mlb.py"
  cat > "$KILLER_DRIVER" <<PYDRIVER
import os, sys, site, logging
os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
REPO = "$PWD"
os.environ.setdefault("C2A_ROOT", REPO)
site.addsitedir(os.path.join(REPO, ".venv", "lib", "python3.12", "site-packages"))
sys.path.insert(0, REPO)
import jax, numpy as np
from pathlib import Path
from omegaconf import OmegaConf
logging.basicConfig(level=logging.INFO)
from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from c2a.moseq_env_wrapper import MoSeqImitation
from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint, make_inference_fn
from c2a.experiments.shared.clip_selection import load_balanced_splits
from c2a.experiments.behavior.run_code_sequence import run_killer_demo
CKPT="$CKPT_PIN"; CODES="$MLB_CODES"
SPLIT="data/rodent/rodent_balanced_splits.json"; REF="data/rodent/rodent_reference_clips.h5"
OUT=Path("c2a/figures/data/mlb"); OUT.mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load("c2a/experiments/behavior/configs/code_sequence_exp.yaml")
cfg.killer_demo.K = 10; cfg.killer_demo.max_steps = 500; cfg.killer_demo.seed = 42
cfg.killer_demo.behaviors = ["groom", "walk", "rear"]; cfg.killer_demo.start_heights = ["low", "high"]
ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(CKPT)
use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
num_codes = int(ckpt_cfg.network_config.num_codes)
codes_data = np.load(CODES); test_codes = codes_data["test_codes"]
splits = load_balanced_splits(SPLIT); test_indices = splits["balanced"]["test_indices"]
test_clips = ReferenceClips(data_path=REF, n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
                            keep_clips_idx=np.array(test_indices))
_, _, env_cfg = utils.prepare_config(ckpt_cfg)
env_cfg.start_frame_range = [0, 0]; env_cfg.domain_randomization.use_domain_randomization = False
css = int(ckpt_cfg.network_config.get("code_stack_size", 1))
env = MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes, code_stack_size=css)
inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)
run_killer_demo(cfg, env, inf_fn, (norm_state, policy_params), ppo_networks, use_rnn,
                test_codes, test_clips, splits, num_codes, OUT,
                jit_reset=jax.jit(env.reset), jit_step=jax.jit(env.step))
print("DONE killer mlb ->", OUT)
PYDRIVER
  run_stage 10 "killer kinematic data (mlb)" 10b_killer.log \
    python3.12 "$KILLER_DRIVER"
fi

# ----------------------------------------------------------------------------
# STAGE 11 — figures (all paper producers, mlb panels)
# ----------------------------------------------------------------------------
if want 11; then
  # 11.1 matrix/stat figures (codemat, distinctness, DSA, jPCA, strip-legend) +
  #      shared KID table (picks up moseq_inception_distance_mlb/results.json).
  run_stage 11 "render_paper_figures (mlb)" 11a_paper_figures.log \
    $PYRUN c2a.figures.paper.render_paper_figures --run "$RUN" --pose random --suffix _mlb

  # 11.2 IC-contraction combined figure (kpms+kvae+mlb scatter).
  run_stage 11 "render_ic_contraction_figure" 11b_ic_contraction_fig.log \
    $PYRUN c2a.figures.paper.render_ic_contraction_figure

  # 11.3 per-strategy + averaged round-trip figure (only emits mlb if stage 7 ran).
  run_stage 11 "render_roundtrip_figure" 11c_roundtrip_fig.log \
    $PYRUN c2a.figures.paper.render_roundtrip_figure

  # 11.4 perturbation recovery figure + appendix filmstrip (mlb row fills from S5).
  # mlb Rear column pinned to code 4 (Rear_Right): the first Rear code (3) gets
  # launched airborne by the impulse and reads badly; code 4 recovers cleanly
  # (grounded rear->kick->rear). mlb Walk column pinned to code 2 (Walk_Straight):
  # the default first Walk code (0) has a reared baseline + belly-flop at the
  # impulse and misreads; code 2 stays quadrupedal throughout (clean
  # walk->kick->walk).
  run_stage 11 "render_perturbation_recovery" 11d_perturbation_fig.log \
    $PYRUN c2a.figures.render_perturbation_recovery --film-pick mlb:Rear=4 mlb:Walk=6

  # 11.4b mlb-only 2-panel (recovery frac. | deviation) for the main Fig 3 (c,d):
  #       paper/figures/perturbation_recovery_mlb.pdf.
  run_stage 11 "render_perturbation_recovery (mlb only)" 11d2_perturbation_mlb.log \
    $PYRUN c2a.figures.render_perturbation_recovery --only mlb

  # 11.5 codebook atlas (definitional 3x3, 9 mlb codes) = fig:codemat (d).
  #      render_codebook_atlas tiles the 9 codes in the fixed MLB_GRID layout
  #      titled by definitional category. Body selection: best_reward reads
  #      clearest for 8/9 codes (tall grounded rears for 3/4, clean locomotion
  #      arcs for 1/2, circling turns for 6/7); code 0 (Walk_L) collapses into an
  #      overlapping pile under best_reward, so it is overridden to the medoid
  #      body (clean 2-body walk). Writes outputs/figures/codebook_atlas/.
  run_stage 11 "render_codebook_atlas (mlb)" 11e_atlas.log \
    $PYRUN c2a.figures.render_codebook_atlas \
      --run "$RUN" --checkpoint "$CKPT_PIN" \
      --select best_reward --select-overrides 0=far_left 1=far_right 2=far_straight --anchor-mode off --hold 300 --n-copies 5 --cam-azim -95 --hold-overrides 1=170 2=170 --copies-overrides 1=4 2=4
  # definitional atlas is the paper's code_atlas_mlb.pdf -> cp+rename.
  cp -f outputs/figures/codebook_atlas/codebook_atlas_mlb.pdf paper/figures/code_atlas_mlb.pdf \
    && echo "  copied codebook_atlas_mlb.pdf -> paper/figures/code_atlas_mlb.pdf"
  cp -f outputs/figures/codebook_atlas/codebook_atlas_mlb.png paper/figures/code_atlas_mlb.png \
    && echo "  copied codebook_atlas_mlb.png -> paper/figures/code_atlas_mlb.png"

  # 11.5b behavior-parade trail (Fig 2c, appendix). Consumes the stage-6 mlb
  #        parade rollouts. Writes paper/figures/behavior_parade_trail_mlb.png.
  run_stage 11 "render_behavior_parade_trail (mlb)" 11e2_parade_trail.log \
    $PYRUN c2a.figures.render_behavior_parade_trail \
      --run "$RUN" --checkpoint "$CKPT_PIN" --codes "$MLB_CODES" \
      --parade-dir outputs/behavior/mlb_parade/walk_groom_rear

  # 11.6 gait PSD overlay (mlb). Writes outputs/figures/psd_overlay/.
  run_stage 11 "plot_psd_overlay (mlb)" 11f_psd.log \
    $PYRUN c2a.figures.plot_psd_overlay --run "$RUN"

  # 11.7 kinematic signatures (mlb). Reads c2a/figures/data/mlb/killer_*.npz.
  run_stage 11 "plot_kinematic_signatures (mlb)" 11g_kinematics.log \
    $PYRUN c2a.figures.plot_kinematic_signatures --run "$RUN"

  # 11.8 stage outputs that do NOT write directly into paper/figures -> copy in.
  banner 11 "copy figures -> paper/figures"
  # (code_atlas_mlb.{pdf,png} is copied in stage 11.5 from render_codebook_atlas.)
  copy_fig outputs/figures/behavior_parade_trail/behavior_parade_trail_mlb.pdf \
           outputs/figures/behavior_parade_trail/behavior_parade_trail_mlb.png
  copy_fig outputs/figures/psd_overlay/psd_overlay_left_mlb.pdf \
           outputs/figures/psd_overlay/psd_overlay_right_mlb.pdf \
           outputs/figures/psd_overlay/psd_overlay_left_mlb.png \
           outputs/figures/psd_overlay/psd_overlay_right_mlb.png
  copy_fig outputs/figures/kinematic_signatures/kinematic_signatures_low_mlb.pdf \
           outputs/figures/kinematic_signatures/kinematic_signatures_high_mlb.pdf \
           outputs/figures/kinematic_signatures/kinematic_signatures_low_mlb.png \
           outputs/figures/kinematic_signatures/kinematic_signatures_high_mlb.png
fi

# ----------------------------------------------------------------------------
# STAGE 12 — presentation demos (auto-include mlb via strategies.available())
#            No --strategy => render every available strategy (incl. mlb).
# ----------------------------------------------------------------------------
if want 12; then
  run_stage 12 "demo: kick_recovery" 12a_demo_kick.log \
    $PYRUN c2a.demos.kick_recovery_demo
  run_stage 12 "demo: code_atlas" 12b_demo_atlas.log \
    $PYRUN c2a.demos.code_atlas_demo
  run_stage 12 "demo: parade_motion" 12c_demo_parade.log \
    $PYRUN c2a.demos.parade_motion_demo --panel-w 1280 --panel-h 960
  run_stage 12 "demo: hidden_dynamics" 12d_demo_hidden.log \
    $PYRUN c2a.demos.hidden_dynamics_demo --body-w 880 --body-h 760
fi

# ----------------------------------------------------------------------------
# STAGE 13 — compile the paper (tectonic)
#            NOTE: the .tex still contains mlb_tbd_wide.pdf placeholders. Apply
#            the figure-include edits in MLB_RUNBOOK.md BEFORE this step for the
#            real mlb panels to appear; otherwise it compiles with placeholders.
# ----------------------------------------------------------------------------
if want 13; then
  run_stage 13 "tectonic compile" 13_compile.log \
    bash -c 'cd "$C2A_ROOT/paper" && tectonic neurips_2026.tex'
fi

echo ""
echo "############################################################"
echo "# mlb Phase-B orchestration finished. Logs: $LOGDIR/"
echo "############################################################"
