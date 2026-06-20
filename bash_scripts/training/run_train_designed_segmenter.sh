#!/bin/bash
# =============================================================================
# Train policy with the designed multi-scale segmenter (K=25 + UNL).
# =============================================================================
# Stage A: build the codes npz (one-time, ~5 s) unless --skip-build is passed.
# Stage B: launch PPO training using c2a/configs/moseq_decoder_designed.yaml.
#
# Usage:
#   bash bash_scripts/run_train_designed_segmenter.sh                # full train
#   bash bash_scripts/run_train_designed_segmenter.sh --smoke        # ~10M steps
#   bash bash_scripts/run_train_designed_segmenter.sh --skip-build   # codes exist
#
# Any extra arguments are forwarded as Hydra overrides to the training script.
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
EXTRA_OVERRIDES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --smoke)      SMOKE=true; shift ;;
        *) EXTRA_OVERRIDES+=("$1"); shift ;;
    esac
done

PATTERN_YAML="c2a/codegen/patterns/multi_scale_v1.yaml"
CODES_NPZ="outputs/tokenizer/designed_v1/best_codes.npz"

# ----- Stage A: build codes -----
if [ "$SKIP_BUILD" = "false" ]; then
    echo "=== Stage A: building designed codes ==="
    python3.12 bash_scripts/_pyrun.py c2a.codegen.designed_segmenter \
        --pattern_yaml "$PATTERN_YAML" \
        --output "$CODES_NPZ"
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
    # Smoke: few minutes, ~10 PPO iters, just verify the loop runs.
    OVERRIDES+=(
        "train_setup.train_config.num_envs=256"
        "train_setup.train_config.num_timesteps=10000000"
        "train_setup.train_config.num_minibatches=4"
        "train_setup.run_name=designed_v1_smoke"
    )
    export WANDB_MODE=disabled
fi

# Append user-provided overrides last so they win.
for o in "${EXTRA_OVERRIDES[@]}"; do
    OVERRIDES+=("$o")
done

python3.12 bash_scripts/_pyrun.py c2a.train_moseq_decoder \
    --config-name=moseq_decoder_designed \
    "${OVERRIDES[@]}"
