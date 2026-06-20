#!/bin/bash
# =============================================================================
# Code2Act — forage-HEIGHT transfer training (Option-B, height-bonus reward)
# =============================================================================
#
# Variable-height arena foraging: some treats are placed HIGH and can only be
# collected by REARING (skull within radius requires a rear); low treats are
# collected normally. A height-bonus reward pays MORE for high treats so the
# policy learns the harder approach->rear instead of grabbing only easy lows.
#
# This is the validated `forage_height_v3` setup (treats 0.84, high-treat
# collection 0.27 at 50M and still climbing) — here with LONGER training so it
# can run directly on another machine. Frozen GRU decoder (decoder-only) + frozen
# Mamba code prior; only the small categorical high-level policy is trained.
#
# Usage:
#   export WANDB_API_KEY=...            # or have ~/.netrc with api.wandb.ai
#   cd code2act
#   bash bash_scripts/run_forage_height.sh
#
# Override per machine via env vars, e.g.:
#   NUM_TIMESTEPS=200000000 NUM_ENVS=4096 RUN_NAME=forage_height_long \
#     bash bash_scripts/run_forage_height.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export C2A_ROOT="$CODE2ACT_ROOT"

# ---- wandb (NEVER hardcode the key; read from env or ~/.netrc) ----
export WANDB_MODE="${WANDB_MODE:-online}"
if [ "$WANDB_MODE" != "disabled" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    if [ -r "$HOME/.netrc" ]; then
        WANDB_API_KEY="$(awk '/api.wandb.ai/{getline; getline; print $2}' "$HOME/.netrc")"
        export WANDB_API_KEY
    fi
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "ERROR: set WANDB_API_KEY env (or ~/.netrc), or WANDB_MODE=disabled." >&2
        exit 1
    fi
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---- knobs (override via env) ----
NUM_TIMESTEPS="${NUM_TIMESTEPS:-150000000}"   # LONGER than the 50M dev run
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_EVALS="${NUM_EVALS:-40}"                   # ~ every 3.75M steps at 150M
RUN_NAME="${RUN_NAME:-forage_height_long}"

cd "$CODE2ACT_ROOT"

python3.12 bash_scripts/_pyrun.py c2a.transfer.train \
    --task forage --prior mamba --dwell 2 \
    --treat_radius 0.09 --num_treats 6 --spawn_radius 0.45 --min_spawn_radius 0.2 \
    --treat_height 0.04 --high_treat_height 0.16 --frac_high_treats 0.5 \
    --scent_weight 1.5 --scent_sigma 0.25 --treat_weight 10.0 --high_treat_mult 4.0 \
    --test_num_treats 10 --test_spawn_radius 1.0 \
    --num_envs "$NUM_ENVS" --num_timesteps "$NUM_TIMESTEPS" --episode_length 400 \
    --batch_size 512 --num_minibatches 16 --num_evals "$NUM_EVALS" --num_eval_envs 64 \
    --entropy_cost 3e-2 --discounting 0.99 \
    --log_video --video_every 3 --video_top_n 6 --video_steps 1000 \
    --run_name "$RUN_NAME"
