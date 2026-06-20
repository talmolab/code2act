#!/bin/bash
# =============================================================================
# Code2Act — forage-DWELL transfer training
# =============================================================================
#
# Variable-height arena foraging with SUSTAINED-posture harvesting.
# Optimal strategy (by reward design):
#
#     walk on all fours -> arrive at patch
#       -> rear (HIGH treat) OR crouch-still (LOW treat)
#       -> sustain for dwell_steps
#       -> harvest fires (large bonus)
#       -> drop / move
#       -> walk to next patch
#
# Reward terms:
#   - dwell_occupancy   : per-step reward while satisfying patch-appropriate
#                         posture INSIDE a patch (rear@high or still@low).
#   - dwell_harvest     : bonus when dwell_steps satisfied (one-time per patch),
#                         high treats pay high_treat_reward_mult x more.
#   - scent_proximity   : potential-based shaping with scent_at_torso=True
#                         (rearing does NOT inflate scent reading -> no rear-hack).
#   - rear_transit_cost : SMALL per-step cost when reared but NOT in XY range
#                         of any active HIGH patch. Forces the "drop after
#                         harvest" pattern instead of the "constant rear+walk"
#                         degenerate equilibrium. Default 0.5.
#
# Defaults to the VAE K=25 decoder + matching K=25 Mamba prior. Override per
# machine via env vars, e.g.:
#
#   DECODER_CKPT=model_checkpoints/260611_073323_750085 \
#   MAMBA_PATH=outputs/ar_codegen/mamba/260612_054031_devuser \
#   RUN_NAME=forage_dwell_kpms_long \
#     bash bash_scripts/run_forage_dwell.sh
#
# Usage:
#   export WANDB_API_KEY=...            # or have ~/.netrc with api.wandb.ai
#   cd code2act
#   bash bash_scripts/run_forage_dwell.sh
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
NUM_TIMESTEPS="${NUM_TIMESTEPS:-150000000}"
NUM_ENVS="${NUM_ENVS:-2048}"
NUM_EVALS="${NUM_EVALS:-40}"
RUN_NAME="${RUN_NAME:-forage_dwell_long}"

# Default: VAE K=25 decoder + matching K=25 Mamba prior. Override for KPMS.
DECODER_CKPT="${DECODER_CKPT:-model_checkpoints/260603_180852_140475}"
MAMBA_PATH="${MAMBA_PATH:-outputs/ar_codegen/mamba/260611_081253_devuser}"

# Reward shape (the design described at the top of this script).
DWELL_STEPS="${DWELL_STEPS:-10}"               # control-steps of sustain to harvest
OCCUPANCY_WEIGHT="${OCCUPANCY_WEIGHT:-1.0}"    # per-step on-patch reward
HARVEST_WEIGHT="${HARVEST_WEIGHT:-10.0}"       # harvest bonus
REAR_TRANSIT_COST="${REAR_TRANSIT_COST:-0.5}"  # transit-rear penalty (0=off)
IMMOBILE_THRESH="${IMMOBILE_THRESH:-0.06}"     # torso speed below = immobile

cd "$CODE2ACT_ROOT"

python3.12 bash_scripts/_pyrun.py c2a.transfer.train \
    --task forage_dwell --prior mamba --dwell 2 \
    --treat_radius 0.09 --num_treats 6 --spawn_radius 0.45 --min_spawn_radius 0.2 \
    --treat_height 0.04 --high_treat_height 0.16 --frac_high_treats 0.5 \
    --scent_weight 1.5 --scent_sigma 0.25 --high_treat_mult 4.0 \
    --occupancy_weight "$OCCUPANCY_WEIGHT" \
    --harvest_weight "$HARVEST_WEIGHT" \
    --rear_transit_cost "$REAR_TRANSIT_COST" \
    --dwell_steps "$DWELL_STEPS" \
    --immobile_thresh "$IMMOBILE_THRESH" \
    --test_num_treats 10 --test_spawn_radius 1.0 \
    --num_envs "$NUM_ENVS" --num_timesteps "$NUM_TIMESTEPS" --episode_length 400 \
    --batch_size 512 --num_minibatches 16 --num_evals "$NUM_EVALS" --num_eval_envs 64 \
    --entropy_cost 3e-2 --discounting 0.99 \
    --log_video --video_every 3 --video_top_n 6 --video_steps 1000 \
    --run_name "$RUN_NAME" \
    --decoder_ckpt "$DECODER_CKPT" \
    --mamba_path "$MAMBA_PATH"
