#!/bin/bash
# =============================================================================
# Code2Act KPMS baseline training (reward_mode=off, distillation on)
# =============================================================================
#
# Pure-tracking PPO with KPMS discrete codes as the segmenter.  No c2c
# reward — the policy sees only the tracking reward from the Imitation env.
# Distillation KL keeps the RNN latent aligned to the pretrained VAE encoder.
#
# Segmenter: K=100 ARHMM (s100_k1e+02_l10 seed4) — best model from sweep.
# GRU input: [code_emb, proprio] — no continuous latent z_e on action path
#             (use_distillation_head forces z_e_at_action_head=False).
#
# Usage:
#   cd code2act
#   WANDB_API_KEY=<key> bash bash_scripts/run_kpms_baseline.sh
#
#   # Extra Hydra overrides can be appended as positional args:
#   WANDB_API_KEY=<key> bash bash_scripts/run_kpms_baseline.sh \
#       train_setup.train_config.num_timesteps=3000000000
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$SCRIPT_DIR")"
export C2A_ROOT="$CODE2ACT_ROOT"

# WandB: load credentials from ~/.netrc.  Never hardcode the token here.
# This script is checked into the repo — any token committed becomes leaked.
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
# KPMS codes (K=100, s100_k1e+02_l10_arhmm seed4 — sweep best).
# num_codes=100 is auto-detected from the npz; no CLI override needed.
# ---------------------------------------------------------------------------
CODES_PATH="outputs/tokenizer/kpms_sweep/s100_k1e+02_l10_arhmm/seed4/best_codes.npz"

if [ ! -e "$CODE2ACT_ROOT/$CODES_PATH" ]; then
    echo "ERROR: KPMS codes not found: $CODE2ACT_ROOT/$CODES_PATH" >&2
    echo "  Run: python3.12 bash_scripts/_pyrun.py c2a.codegen.kpms.generate_codes \\" >&2
    echo "         --sweep-results outputs/tokenizer/kpms_sweep/sweep_results.json \\" >&2
    echo "         --balanced-split data/rodent/rodent_balanced_splits.json \\" >&2
    echo "         --output $CODES_PATH" >&2
    exit 1
fi

cd "$CODE2ACT_ROOT"

# ---------------------------------------------------------------------------
# Launch — from-scratch training, reward_mode=off (pure tracking).
# ---------------------------------------------------------------------------
RUN_NAME="kpms_baseline_$(date +%y%m%d_%H%M%S)"

python3.12 bash_scripts/_pyrun.py c2a.train_moseq_decoder \
    kpms_config.code_source=kpms_raw \
    kpms_config.codes_path="${CODES_PATH}" \
    code2code.reward_mode=off \
    network_config.distill_kl_weight=1.0 \
    train_setup.run_name="${RUN_NAME}" \
    "$@"
