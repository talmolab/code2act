#!/bin/bash
# =============================================================================
# Code2Code curriculum-mode decoder RL
# =============================================================================
# Schedule:
#   r_total(τ) = (1 - α(τ)) · r_track + α(τ) · (alpha · match) + β(τ) · log(VS)
#   τ = current_env_step / num_timesteps ∈ [0, 1]
#   α(τ): tanh ramp α_min → α_max around τ_mid (sharpness τ_scale)
#   β(τ): tanh ramp 0 → β_max around τ_div_start (sharpness τ_div_scale)
#
# Bootstrap phase (τ < τ_mid − τ_scale): tracking-dominant. ≥ 1e9 env steps
# required — validation in c2a/code2code/schedules.py rejects shorter runs.
# Transition: α tanh-anneals into c2c-shaper regime. α_max ≤ 0.75 hard cap
# (tracking floor 0.25 must remain forever — c2c is root-invariant and
# would let policy drift if α_max → 1).
# Late phase: diversity bonus β·log(VS) shapes within-class behavior.
#
# Uses the K=25 kl=0.001 VAE tokenizer (sweep-best, same as the
# replace-distill runs).  num_timesteps default: 5e9 (gives 1.25e9
# bootstrap @ τ_mid=0.30, plenty for tracking convergence).
#
# WandB panels populated: curriculum/{alpha, r_track, r_c2c_raw,
# eval/avg_episode_length, eval/episode_terminations/root_too_far, …}.
#
# Usage:
#   bash bash_scripts/run_c2c_curriculum.sh
#   bash bash_scripts/run_c2c_curriculum.sh \
#       code2code.curriculum_config.tau_mid=0.25
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
# Tokenizer (K=25, kl=0.001 — best from the 2026-05-26 VAE sweep)
# ---------------------------------------------------------------------------
TOKENIZER_DIR="outputs/tokenizer/trained_kinematics_vae_ckpts/vae_sweep/kl_0p001"
TOKENIZER_CKPT="${TOKENIZER_DIR}/step_0030000"
TOKENIZE_DIR="${TOKENIZER_DIR}/tokenize_k25"
CODES_PATH="${TOKENIZE_DIR}/best_codes.npz"
VAE_PKL="${TOKENIZE_DIR}/tokenizer.pkl"

# Sanity-check the files exist before invoking Python (cheap fail-fast).
for f in "$CODES_PATH" "$VAE_PKL" "${TOKENIZER_CKPT}"; do
  if [ ! -e "$CODE2ACT_ROOT/$f" ]; then
    echo "ERROR: missing required file: $CODE2ACT_ROOT/$f" >&2
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
    code2code.reward_mode=curriculum \
    code2code.tokenizer_kind=vae \
    code2code.tokenizer_ckpt="${TOKENIZER_CKPT}" \
    code2code.vae_tokenizer_pkl="${VAE_PKL}" \
    code2code.alpha=1.0 \
    code2code.curriculum_config.alpha_min=0.05 \
    code2code.curriculum_config.alpha_max=0.75 \
    code2code.curriculum_config.tau_mid=0.30 \
    code2code.curriculum_config.tau_scale=0.05 \
    network_config.distill_kl_weight=1.0 \
    train_setup.train_config.num_timesteps=5000000000 \
    train_setup.run_name=c2c_curriculum_k25_kl1 \
    "$@"
