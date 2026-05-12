#!/bin/bash
# =============================================================================
# Code2Act kinematics-VAE pipeline:
#   pretrain encoder -> tokenize codes -> decoder RL training
# =============================================================================
#
# Variants:
#   vae       -> vanilla Gaussian VAE; codes via K-Means on latent
#   vqvae     -> single-codebook VQVAE (van den Oord 2017; gradient codebook)
#   ps_vqvae  -> product-space VQVAE (VQ-MAP, Li et al. 2025; per-group EMA-VQ
#                with dead-code reseed)
#
# Usage:
#   cd code2act
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> [num_steps]
#   bash bash_scripts/run_pipeline_kinematics_vae.sh vae
#   bash bash_scripts/run_pipeline_kinematics_vae.sh vqvae 5000
#   bash bash_scripts/run_pipeline_kinematics_vae.sh ps_vqvae 30000
#
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> --skip-pretrain  \
#       --ckpt-dir trained_kinematics_vae_ckpts/<variant>/step_0030000
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> --skip-pretrain  \
#       --skip-tokenize
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> --dry-run
#
# Each stage runs in its own process for clean Hydra / JAX bring-up.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$CODE2ACT_ROOT/.venv/bin/python"
export C2A_ROOT="$CODE2ACT_ROOT"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VARIANT=""
NUM_STEPS=""
CKPT_DIR=""
SKIP_PRETRAIN=false
SKIP_TOKENIZE=false
SKIP_DECODER=false
DRY_RUN=false

show_help() {
    cat <<EOF
Usage: bash run_pipeline_kinematics_vae.sh <variant> [num_steps] [flags]

Variants:
  vae        Vanilla Gaussian VAE + K-Means tokenization
  vqvae      Vanilla VQVAE (gradient codebook + STE)
  ps_vqvae   Product-space VQVAE (per-group EMA codebooks)

Flags:
  --ckpt-dir DIR     Use a specific pretrain ckpt for tokenize / decoder stages.
                     Default: trained_kinematics_vae_ckpts/<variant>/step_<num_steps>.
  --skip-pretrain    Skip Stage 1.
  --skip-tokenize    Skip Stage 2.
  --skip-decoder     Skip Stage 3.
  --dry-run          Print commands without executing.
  -h, --help         Show this help.

Examples:
  bash run_pipeline_kinematics_vae.sh vae
  bash run_pipeline_kinematics_vae.sh ps_vqvae 5000 --skip-decoder
EOF
}

if [[ $# -eq 0 ]]; then show_help; exit 0; fi

VARIANT="$1"; shift
case "$VARIANT" in
    vae|vqvae|ps_vqvae) ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown variant: $VARIANT (expected vae|vqvae|ps_vqvae)"; exit 1 ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt-dir) CKPT_DIR="$2"; shift 2 ;;
        --skip-pretrain) SKIP_PRETRAIN=true; shift ;;
        --skip-tokenize) SKIP_TOKENIZE=true; shift ;;
        --skip-decoder)  SKIP_DECODER=true;  shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        -h|--help) show_help; exit 0 ;;
        *)
            # First positional arg after variant treated as num_steps.
            if [[ -z "$NUM_STEPS" && "$1" =~ ^[0-9]+$ ]]; then
                NUM_STEPS="$1"; shift
            else
                echo "Unknown option: $1"; exit 1
            fi
            ;;
    esac
done

# Defaults
: "${NUM_STEPS:=30000}"
WORKDIR="$CODE2ACT_ROOT/trained_kinematics_vae_ckpts/$VARIANT"
DEFAULT_CKPT="$WORKDIR/step_$(printf '%07d' "$NUM_STEPS")"
: "${CKPT_DIR:=$DEFAULT_CKPT}"

CODES_NPZ="$WORKDIR/best_codes.npz"

# ---------------------------------------------------------------------------
# Tokenize hyperparams (sensible defaults per variant)
# ---------------------------------------------------------------------------
case "$VARIANT" in
    vae)
        TOKENIZE_ARGS=(--num-codes 50 --pca-dim 8)
        ;;
    vqvae)
        TOKENIZE_ARGS=(--codebook-size 128 --code-dim 24)
        ;;
    ps_vqvae)
        TOKENIZE_ARGS=(--group-dims 4 8 12 --group-codebook-sizes 8 8 16)
        ;;
esac

# ---------------------------------------------------------------------------
# Logging / runner
# ---------------------------------------------------------------------------
log() { echo -e "\n=== $* ==="; }

run() {
    if $DRY_RUN; then echo "[dry-run] $*"; else "$@"; fi
}

log "Kinematics-VAE pipeline — variant=$VARIANT, num_steps=$NUM_STEPS"
echo "  C2A_ROOT  : $C2A_ROOT"
echo "  ckpt dir  : $CKPT_DIR"
echo "  codes npz : $CODES_NPZ"

# ---------------------------------------------------------------------------
# Stage 1 — pretrain
# ---------------------------------------------------------------------------
if ! $SKIP_PRETRAIN; then
    log "Stage 1 — pretrain $VARIANT for $NUM_STEPS steps"
    run "$VENV_PYTHON" -m "c2a.codegen.kinematics_vae.$VARIANT.train" \
        train.num_steps="$NUM_STEPS" \
        train.ckpt_dir="$WORKDIR"
else
    log "Stage 1 — SKIPPED (--skip-pretrain)"
fi

# ---------------------------------------------------------------------------
# Stage 2 — tokenize (write the unified codes-npz)
# ---------------------------------------------------------------------------
if ! $SKIP_TOKENIZE; then
    log "Stage 2 — tokenize using ckpt $CKPT_DIR"
    run "$VENV_PYTHON" -m "c2a.codegen.kinematics_vae.$VARIANT.tokenize" \
        --ckpt-dir "$CKPT_DIR" \
        --output "$CODES_NPZ" \
        "${TOKENIZE_ARGS[@]}"
else
    log "Stage 2 — SKIPPED (--skip-tokenize)"
fi

# ---------------------------------------------------------------------------
# Stage 3 — decoder RL (uses the unified code_source dispatch)
# ---------------------------------------------------------------------------
if ! $SKIP_DECODER; then
    log "Stage 3 — decoder RL with code_source=$VARIANT"
    run "$VENV_PYTHON" -m c2a.train_moseq_decoder \
        kpms_config.code_source="$VARIANT" \
        kpms_config.codes_path="$CODES_NPZ" \
        train_setup.run_name="kinematics_vae_${VARIANT}_$(date +%Y%m%d_%H%M%S)"
else
    log "Stage 3 — SKIPPED (--skip-decoder)"
fi

log "Pipeline complete."
