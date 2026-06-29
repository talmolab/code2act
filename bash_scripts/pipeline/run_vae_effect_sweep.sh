#!/usr/bin/env bash
# =============================================================================
# run_vae_effect_sweep.sh — effect of the kinematics-VAE tokenizer on the controller
# =============================================================================
# The kinematics-VAE counterpart of run_kpms_effect_sweep.sh. A CONTROLLED 2-D
# tokenizer comparison over a grid of (KL weight beta x vocabulary size K):
#
#   * beta  (loss_weights.kl) is a TRAIN-time param  -> one VAE per beta (retrain)
#   * K     (num_codes, K-Means)  is POST-HOC          -> one VAE tokenizes at all K
#
# So we train |KL_VALUES| VAEs, tokenize each at every K, and train one decoder
# per (beta, K) tokenizer with a BYTE-IDENTICAL production decoder config. Any
# downstream difference is attributable to the tokenizer, not the decoder.
#
#   Stage 1  train one kinematics-VAE per beta
#   Stage 2  tokenize each beta-VAE at every K            -> |KL| x |K| code npz
#   Stage 3  ONE full-budget decoder per (beta, K), identical config, only the
#            code stream varies. Emits ready-to-submit job commands (default)
#            or runs them sequentially here (--launch).
#
# Usage (from the code2act/ repo root):
#   bash bash_scripts/pipeline/run_vae_effect_sweep.sh             # train + emit jobs
#   bash bash_scripts/pipeline/run_vae_effect_sweep.sh --launch    # train + run locally
#   bash bash_scripts/pipeline/run_vae_effect_sweep.sh --skip-train --skip-tokenize  # reuse codes
#
# NOTE: the `vae` variant is the paper's learned-tokenizer family. Its
# per-tokenizer hyperparameter sweep lives in bash_scripts/sweeps/sweep_vae.sh.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export C2A_ROOT="$CODE2ACT_ROOT"
cd "$CODE2ACT_ROOT"

PYRUN="python3.12 bash_scripts/_pyrun.py"

# --- sweep grid: KL weight beta (retrain) x vocabulary size K (post-hoc) ------
# Centered on best_vae 260603_180852_140475 (kinematics-VAE, K=25, kl=0.001).
# beta spans an order of magnitude around the figured-out best (0.001); the
# config default is 0.005. K brackets 25 from coarse to fine.
# Default grid = 3 beta x 4 K = 12 controllers. Edit either array to taste.
KL_VALUES=(0.0003 0.001 0.005)
K_VALUES=(12 25 48 100)
VAE_STEPS=30000
PCA_DIM=8

# --- paths -------------------------------------------------------------------
VAE_BASE="$CODE2ACT_ROOT/trained_kinematics_vae_ckpts/vae_effect"
CODES_DIR="outputs/tokenizer/vae_effect_sweep/codes"
JOBS_FILE="outputs/tokenizer/vae_effect_sweep/decoder_jobs.sh"
WANDB_GROUP="vae_effect_grid"
CODE_SOURCE="kinematics_vae"

# Per-beta VAE checkpoint dir (slugify the dot in beta for a clean path).
vae_workdir() { echo "$VAE_BASE/kl${1//./p}"; }
vae_ckpt()    { echo "$(vae_workdir "$1")/step_$(printf '%07d' "$VAE_STEPS")"; }
codes_npz()   { echo "$CODES_DIR/codes_kl${1//./p}_k${2}.npz"; }

# --- decoder config: the PRODUCTION setup, unchanged -------------------------
# Every tokenizer is decoded by the IDENTICAL production controller defined in
# c2a/configs/moseq_decoder.yaml (distillation head vs
# trained_mimic_ckpts/260413_040706_764936, pretrained decoder, GRU
# [512,512,256,256], code_stack=5, continuous_latent_dim=16, cosine KL,
# num_timesteps=5e9). NOTHING in that config is overridden except the code
# source — only the VAE code stream varies, isolating the tokenizer's effect.

# --- args --------------------------------------------------------------------
SKIP_TRAIN=false
SKIP_TOKENIZE=false
LAUNCH=false
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-train)    SKIP_TRAIN=true ;;
        --skip-tokenize) SKIP_TOKENIZE=true ;;
        --launch)        LAUNCH=true ;;
        --steps)         VAE_STEPS="$2"; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

N_GRID=$(( ${#KL_VALUES[@]} * ${#K_VALUES[@]} ))
echo "=============================================="
echo "  kinematics-VAE effect sweep (2-D grid)"
echo "  beta (kl):   ${KL_VALUES[*]}   (retrain per value)"
echo "  K values:    ${K_VALUES[*]}   (post-hoc per VAE)"
echo "  grid:        ${#KL_VALUES[@]} beta x ${#K_VALUES[@]} K = $N_GRID controllers"
echo "  vae train:   steps=$VAE_STEPS pca_dim=$PCA_DIM  (best_vae arch)"
echo "  decoder:     production c2a/configs/moseq_decoder.yaml (unchanged; distill vs 260413)"
echo "  skip-train:  $SKIP_TRAIN   skip-tokenize: $SKIP_TOKENIZE   launch: $LAUNCH"
echo "=============================================="

# --- Stage 1: train one kinematics-VAE per beta ------------------------------
if [ "$SKIP_TRAIN" = false ]; then
    echo "=== Stage 1: train ${#KL_VALUES[@]} kinematics-VAE(s), one per beta ==="
    for B in "${KL_VALUES[@]}"; do
        echo "  train VAE  kl(beta)=$B  ($VAE_STEPS steps)"
        $PYRUN c2a.codegen.kinematics_vae.vae.train \
            train.num_steps="$VAE_STEPS" \
            train.ckpt_dir="$(vae_workdir "$B")" \
            loss_weights.kl="$B"
    done
else
    echo "=== Stage 1: SKIPPED (--skip-train) ==="
fi

# --- Stage 2: tokenize each beta-VAE at every K ------------------------------
mkdir -p "$CODES_DIR"
if [ "$SKIP_TOKENIZE" = false ]; then
    echo "=== Stage 2: tokenize $N_GRID (beta, K) cells ==="
    for B in "${KL_VALUES[@]}"; do
        for K in "${K_VALUES[@]}"; do
            echo "  tokenize  kl=$B  K=$K"
            $PYRUN c2a.codegen.kinematics_vae.vae.tokenize \
                --ckpt-dir "$(vae_ckpt "$B")" \
                --output "$(codes_npz "$B" "$K")" \
                --num-codes "$K" --pca-dim "$PCA_DIM"
        done
    done
else
    echo "=== Stage 2: SKIPPED (--skip-tokenize) ==="
fi

# --- Stage 3: one decoder per (beta, K), identical config --------------------
echo "=== Stage 3: build decoder jobs (one per grid cell) ==="
: > "$JOBS_FILE"
echo "#!/usr/bin/env bash" >> "$JOBS_FILE"
echo "# Auto-generated decoder jobs — one full-budget run per (beta, K) VAE tokenizer." >> "$JOBS_FILE"
echo "# Submit each line as a job, or run this file to execute sequentially." >> "$JOBS_FILE"
echo "set -euo pipefail; cd \"$CODE2ACT_ROOT\"" >> "$JOBS_FILE"

N=0
for B in "${KL_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        CODES_NPZ="$(codes_npz "$B" "$K")"
        if [ ! -f "$CODES_NPZ" ] && [ "$SKIP_TOKENIZE" = true ]; then
            echo "  WARNING: missing $CODES_NPZ (skipped tokenize); skipping kl=$B K=$K" >&2
            continue
        fi
        RUN_NAME="vae_effect_kl${B//./p}_k${K}"

        CMD="$PYRUN c2a.train_moseq_decoder \
kpms_config.code_source=$CODE_SOURCE \
kpms_config.codes_path=$CODES_NPZ \
network_config.num_codes=$K \
train_setup.run_name=$RUN_NAME \
logging_config.group_name=$WANDB_GROUP \
logging_config.exp_name=$RUN_NAME"

        echo "$CMD" >> "$JOBS_FILE"
        N=$((N + 1))
        echo "  [$N] $RUN_NAME  (codes=$CODES_NPZ, kl=$B, num_codes=$K)"
    done
done

echo ""
echo "Wrote $N decoder job(s) to: $JOBS_FILE"

if [ "$LAUNCH" = true ]; then
    echo "=== --launch: running $N decoders sequentially ==="
    bash "$JOBS_FILE"
else
    echo "Submit each line of $JOBS_FILE as a job, or re-run with --launch."
fi
