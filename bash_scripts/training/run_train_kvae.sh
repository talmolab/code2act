#!/bin/bash
# =============================================================================
# Code2Act — train the KVAE steering strategy (kinematics-VAE, c2c additive)
# =============================================================================
#
# Trains a code-conditioned policy with distillation + additive c2c reward:
#
#     r = c2c_track_weight * r_track  +  c2c_alpha * match
#
# where `match = exp(-d/τ)` and `d` is the cosine distance between the
# encoder's z(current pose) and the codebook entry for the reference code.
#
# This is the simplified pipeline after we deleted curriculum, finetune,
# replace, and the skill-diversity terms (distillation fade,
# action-distinguishability).  The policy is anchored to the demo
# manifold by:
#   - constant distillation KL pull  (network_config.distill_kl_weight)
#   - additive c2c reward bonus      (code2code.alpha * match)
#   - hard tracking reward floor     (r_track always present)
#
# Tokenizer: K=30 default kinematics VAE (the canonical pretrained
# tokenizer; same K and ckpt the prior runs used).  Override by setting
# the TOKENIZER_DIR / CODES_PATH env vars before invocation.
#
# Usage:
#   cd code2act
#   bash bash_scripts/training/run_train_kvae.sh
#
#   # Override common settings on the CLI (Hydra overrides):
#   bash bash_scripts/training/run_train_kvae.sh \
#       code2code.alpha=0.7 \
#       train_setup.train_config.num_timesteps=1500000000
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export C2A_ROOT="$CODE2ACT_ROOT"

# WandB: load credentials from ~/.netrc.  Never hardcode the token here.
# The two getline calls are required: ~/.netrc stores the key on the
# line AFTER 'login' which is itself one line after the machine entry.
export WANDB_MODE="${WANDB_MODE:-online}"
if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ ! -r "$HOME/.netrc" ]; then
        echo "ERROR: ~/.netrc not readable and WANDB_API_KEY is not set." >&2
        exit 1
    fi
    WANDB_API_KEY="$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")"
    if [ -z "$WANDB_API_KEY" ]; then
        echo "ERROR: could not extract WandB API key from ~/.netrc." >&2
        exit 1
    fi
    export WANDB_API_KEY
fi

# MuJoCo headless rendering for the eval video logger.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Tokenizer paths (K=30 default kinematics VAE; sweep tokenizers override
# by setting TOKENIZER_DIR / TOKENIZER_KIND before invocation).
#
# TOKENIZER_KIND is `vae`. It drives both code2code.tokenizer_kind and
# kpms_config.code_source. `vae` requires a tokenizer.pkl (the K-Means
# centroids).
# ---------------------------------------------------------------------------
: "${TOKENIZER_KIND:=vae}"
: "${TOKENIZER_DIR:=outputs/tokenizer/trained_kinematics_vae_ckpts/vae}"
: "${TOKENIZER_CKPT:=${TOKENIZER_DIR}/step_0030000}"
: "${CODES_PATH:=${TOKENIZER_DIR}/best_codes.npz}"
: "${VAE_PKL:=${TOKENIZER_DIR}/tokenizer.pkl}"

case "$TOKENIZER_KIND" in
    vae)      CODE_SOURCE="kinematics_vae" ;;
    *) echo "ERROR: TOKENIZER_KIND=$TOKENIZER_KIND not in {vae}" >&2; exit 1 ;;
esac

REQUIRED=("$CODES_PATH" "$TOKENIZER_CKPT")
[ "$TOKENIZER_KIND" = "vae" ] && REQUIRED+=("$VAE_PKL")
for f in "${REQUIRED[@]}"; do
    if [ ! -e "$CODE2ACT_ROOT/$f" ]; then
        echo "ERROR: missing tokenizer artifact: $CODE2ACT_ROOT/$f" >&2
        exit 1
    fi
done

cd "$CODE2ACT_ROOT"

# ---------------------------------------------------------------------------
# Launch — from-scratch training (no checkpoint_to_restore).
# ---------------------------------------------------------------------------
RUN_NAME="c2c_additive_$(date +%y%m%d_%H%M%S)"

EXTRA_TOK_ARGS=()
[ "$TOKENIZER_KIND" = "vae" ] && EXTRA_TOK_ARGS+=("code2code.vae_tokenizer_pkl=${VAE_PKL}")

python3.12 bash_scripts/_pyrun.py c2a.train_moseq_decoder \
    kpms_config.code_source="${CODE_SOURCE}" \
    kpms_config.codes_path="${CODES_PATH}" \
    code2code.reward_mode=additive \
    code2code.tokenizer_kind="${TOKENIZER_KIND}" \
    code2code.tokenizer_ckpt="${TOKENIZER_CKPT}" \
    "${EXTRA_TOK_ARGS[@]}" \
    code2code.alpha=0.5 \
    code2code.track_weight=1.0 \
    code2code.temperature=auto \
    network_config.distill_kl_weight=1.0 \
    train_setup.run_name="${RUN_NAME}" \
    "$@"
