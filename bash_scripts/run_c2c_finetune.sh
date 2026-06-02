#!/bin/bash
# =============================================================================
# Code2Code finetune-mode RL
# =============================================================================
# Loads a pretrained pure-tracking checkpoint and adds c2c + diversity
# on top using the curriculum's late-stage reward composition:
#
#   r_total(τ) = (1 - α(τ)) · r_track + α(τ) · (alpha · match) + β(τ) · log(VS)
#
# With finetune defaults:
#   α(τ) ramps 0 → 0.75 over the first 5% of training, then holds at 0.75
#   β(τ) ramps 0 → 0.10 over the first 5% of training, then holds at 0.10
#   α_max = 0.75 hard cap (same as curriculum — c2c is root-invariant;
#                          tracking floor of 0.25 must remain forever)
#
# Default checkpoint: model_checkpoints/260513_062522_599062
#   (trained 2026-05-13, predates c2c integration → guaranteed pure-tracking)
#
# Default tokenizer: K=30 default vae (matches the original training of
# the default ckpt — DO NOT swap to a different K, the policy's code-
# embedding table is sized to 50 entries and assumes the K=30 semantics).
#
# Lower lr (3e-5) and shorter timesteps (1.5B) than curriculum because
# no bootstrap is needed — the loaded policy is already converged on
# tracking.
#
# Usage:
#   bash bash_scripts/run_c2c_finetune.sh
#   bash bash_scripts/run_c2c_finetune.sh \
#       train_setup.checkpoint_to_restore=/abs/path/to/other_ckpt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$SCRIPT_DIR")"
export C2A_ROOT="$CODE2ACT_ROOT"

# WandB — caller must supply WANDB_API_KEY
export WANDB_MODE="${WANDB_MODE:-online}"
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set. Run 'wandb login' or export it." >&2
  exit 1
fi

# MuJoCo headless rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Pretrained tracking checkpoint
# ---------------------------------------------------------------------------
# Override with `train_setup.checkpoint_to_restore=...` on the CLI.
CKPT="model_checkpoints/260513_062522_599062"

# ---------------------------------------------------------------------------
# Tokenizer — MUST match what the loaded checkpoint was trained with.
# The default ckpt used K=30 default vae (the unswept vae/ dir, NOT the
# K=25 kl=0.001 sweep-best used by curriculum mode).
# ---------------------------------------------------------------------------
TOKENIZER_DIR="outputs/tokenizer/trained_kinematics_vae_ckpts/vae"
TOKENIZER_CKPT="${TOKENIZER_DIR}/step_0030000"
CODES_PATH="${TOKENIZER_DIR}/best_codes.npz"
VAE_PKL="${TOKENIZER_DIR}/tokenizer.pkl"

# Sanity-check that everything exists before invoking Python.
for f in "$CODES_PATH" "$VAE_PKL" "$TOKENIZER_CKPT" "$CKPT"; do
  if [ ! -e "$CODE2ACT_ROOT/$f" ]; then
    echo "ERROR: missing required path: $CODE2ACT_ROOT/$f" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
cd "$CODE2ACT_ROOT"
python3.12 bash_scripts/_pyrun.py c2a.train_moseq_decoder \
    kpms_config.code_source=kinematics_vae \
    kpms_config.codes_path="${CODES_PATH}" \
    code2code.reward_mode=finetune \
    code2code.tokenizer_kind=vae \
    code2code.tokenizer_ckpt="${TOKENIZER_CKPT}" \
    code2code.vae_tokenizer_pkl="${VAE_PKL}" \
    code2code.alpha=1.0 \
    code2code.finetune_config.alpha=0.75 \
    code2code.finetune_config.warmup_fraction=0.05 \
    code2code.finetune_config.expected_num_codes=50 \
    network_config.distill_kl_weight=1.0 \
    code2code.finetune_config.default_checkpoint="${CKPT}" \
    train_setup.train_config.num_timesteps=1500000000 \
    train_setup.train_config.learning_rate=3e-5 \
    train_setup.run_name=c2c_finetune_from_260513 \
    "$@"
