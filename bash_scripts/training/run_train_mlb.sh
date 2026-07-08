#!/bin/bash
# =============================================================================
# Train the MLB steering-strategy controller (9 flat behavior x direction codes).
# =============================================================================
# Stage A: build the codes npz (one-time, ~5 s) unless --skip-build is passed.
# Stage B: launch PPO training using c2a/configs/moseq_decoder_mlb.yaml.
#
# Usage:
#   bash bash_scripts/training/run_train_mlb.sh                # full train
#   bash bash_scripts/training/run_train_mlb.sh --smoke        # ~10M steps
#   bash bash_scripts/training/run_train_mlb.sh --skip-build   # codes exist
#   bash bash_scripts/training/run_train_mlb.sh --deadband 0.15
#
# Any other extra arguments are forwarded as Hydra overrides to training.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export C2A_ROOT="$CODE2ACT_ROOT"
cd "$CODE2ACT_ROOT"

# ----- WandB & XLA env (same pattern as other launchers) -----
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -f "$HOME/.netrc" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    WANDB_API_KEY="$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")"
    if [ -n "$WANDB_API_KEY" ]; then
        export WANDB_API_KEY
    fi
fi
export WANDB_MODE="${WANDB_MODE:-online}"

# ----- arg parsing -----
SKIP_BUILD=false
SMOKE=false
DEADBAND="0.10"
EXTRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --smoke)      SMOKE=true; shift ;;
        --deadband)   DEADBAND="$2"; shift 2 ;;
        *) EXTRA_OVERRIDES+=("$1"); shift ;;
    esac
done

CODES_NPZ="outputs/tokenizer/mlb_v1/best_codes.npz"

# ----- Stage A: build codes -----
if [ "$SKIP_BUILD" = "false" ]; then
    echo "=== Stage A: building MLB codes (deadband=$DEADBAND) ==="
    python3.12 bash_scripts/_pyrun.py c2a.codegen.mlb_segmenter \
        --output "$CODES_NPZ" \
        --deadband "$DEADBAND"
else
    echo "=== Stage A: skipped (using existing $CODES_NPZ) ==="
fi

if [ ! -f "$CODES_NPZ" ]; then
    echo "ERROR: $CODES_NPZ does not exist." >&2
    exit 1
fi

# ----- Stage B: train -----
echo
echo "=== Stage B: launching PPO training ==="

OVERRIDES=()
if [ "$SMOKE" = "true" ]; then
    OVERRIDES+=(
        "train_setup.train_config.num_envs=256"
        "train_setup.train_config.num_timesteps=10000000"
        "train_setup.train_config.num_minibatches=4"
        "train_setup.run_name=mlb_v1_smoke"
    )
    export WANDB_MODE=disabled
fi

for o in "${EXTRA_OVERRIDES[@]}"; do
    OVERRIDES+=("$o")
done

python3.12 bash_scripts/_pyrun.py c2a.train_moseq_decoder \
    --config-name=moseq_decoder_mlb \
    "${OVERRIDES[@]}"
