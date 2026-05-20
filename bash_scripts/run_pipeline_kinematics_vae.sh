#!/bin/bash
# =============================================================================
# Code2Act kinematics-VAE pipeline:
#   pretrain encoder -> tokenize codes -> render bands/mp4s -> decoder RL
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
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> [num_steps] [flags]
#   bash bash_scripts/run_pipeline_kinematics_vae.sh vae
#   bash bash_scripts/run_pipeline_kinematics_vae.sh vqvae 5000
#   bash bash_scripts/run_pipeline_kinematics_vae.sh ps_vqvae 30000
#
#   # Change the number of codes at tokenize time (vae only — no retraining needed):
#   bash bash_scripts/run_pipeline_kinematics_vae.sh vae --num-codes 50
#
#   # Change the codebook size (vqvae — affects BOTH training AND tokenization):
#   bash bash_scripts/run_pipeline_kinematics_vae.sh vqvae --num-codes 64
#
#   # Change per-group codebook sizes (ps_vqvae — affects BOTH training AND tokenization).
#   # Pass as a comma-separated string; total codes = K1*K2*K3:
#   bash bash_scripts/run_pipeline_kinematics_vae.sh ps_vqvae --group-codebook-sizes 8,8,16
#
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> --skip-pretrain  \
#       --ckpt-dir trained_kinematics_vae_ckpts/<variant>/step_0030000
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> --skip-pretrain  \
#       --skip-tokenize
#   bash bash_scripts/run_pipeline_kinematics_vae.sh <variant> --dry-run
#
# Each stage runs in its own process for clean Hydra / JAX bring-up.
#
# NOTE on codebook size and checkpoints:
#   For vqvae and ps_vqvae the codebook size is baked into the trained model,
#   so --num-codes / --group-codebook-sizes is forwarded to BOTH the training
#   and tokenization stages. Running with a different size than the default
#   (30 for vqvae, 4,4,4 for ps_vqvae) will overwrite any existing checkpoint
#   in the same --ckpt-dir. To keep multiple codebook sizes in parallel, pass
#   a size-specific --ckpt-dir, e.g.:
#     --ckpt-dir trained_kinematics_vae_ckpts/vqvae_K64
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
# User-supplied code-count overrides (empty = use per-variant default below).
NUM_CODES_OVERRIDE=""       # --num-codes N  (vae / vqvae only)
GROUP_CB_SIZES_OVERRIDE=""  # --group-codebook-sizes K1,K2,K3  (ps_vqvae only)
SKIP_PRETRAIN=false
SKIP_TOKENIZE=false
SKIP_RENDER=false
SKIP_DECODER=false
DRY_RUN=false

show_help() {
    cat <<EOF
Usage: bash run_pipeline_kinematics_vae.sh <variant> [num_steps] [flags]

Variants:
  vae        Vanilla Gaussian VAE + K-Means tokenization
  vqvae      Vanilla VQVAE (gradient codebook + STE)
  ps_vqvae   Product-space VQVAE (per-group EMA codebooks)

Code-count flags (override defaults of 30 / 30 / 4,4,4):
  --num-codes N                   vae / vqvae only.
                                  For vae: K-Means cluster count (post-hoc,
                                    no retraining needed).
                                  For vqvae: codebook size K — forwarded to
                                    BOTH training and tokenization; the trained
                                    model has this size baked in.
  --group-codebook-sizes K1,K2,K3 ps_vqvae only (comma-separated, no spaces).
                                  Per-group codebook sizes; total codes =
                                    K1*K2*K3. Forwarded to BOTH training and
                                    tokenization.

Other flags:
  --ckpt-dir DIR     Use a specific pretrain ckpt for tokenize / decoder stages.
                     Default: trained_kinematics_vae_ckpts/<variant>/step_<num_steps>.
                     Tip: specify a size-specific path when sweeping codebook
                     sizes to avoid overwriting checkpoints, e.g.:
                       --ckpt-dir trained_kinematics_vae_ckpts/vqvae_K64/step_0030000
  --skip-pretrain    Skip Stage 1 (pretrain encoder/codebook).
                     If --num-codes / --group-codebook-sizes is set, the
                     specified sizes must match the existing checkpoint or
                     the tokenize step will fail with a shape error.
  --skip-tokenize    Skip Stage 2 (write best_codes.npz + viz PNGs).
  --skip-render      Skip Stage 3 (per-clip band figures + MP4s).
  --skip-decoder     Skip Stage 4 (PPO decoder RL).
  --dry-run          Print commands without executing.
  -h, --help         Show this help.

Examples:
  bash run_pipeline_kinematics_vae.sh vae
  bash run_pipeline_kinematics_vae.sh vae --num-codes 50
  bash run_pipeline_kinematics_vae.sh vqvae --num-codes 64
  bash run_pipeline_kinematics_vae.sh ps_vqvae --group-codebook-sizes 8,8,16
  bash run_pipeline_kinematics_vae.sh ps_vqvae 5000 --skip-decoder
  bash run_pipeline_kinematics_vae.sh vqvae --num-codes 64 \
      --ckpt-dir trained_kinematics_vae_ckpts/vqvae_K64
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
        --num-codes) NUM_CODES_OVERRIDE="$2"; shift 2 ;;
        --group-codebook-sizes) GROUP_CB_SIZES_OVERRIDE="$2"; shift 2 ;;
        --skip-pretrain) SKIP_PRETRAIN=true; shift ;;
        --skip-tokenize) SKIP_TOKENIZE=true; shift ;;
        --skip-render)   SKIP_RENDER=true;   shift ;;
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

# ---------------------------------------------------------------------------
# Validate flag / variant compatibility
# ---------------------------------------------------------------------------
if [[ -n "$NUM_CODES_OVERRIDE" && "$VARIANT" == "ps_vqvae" ]]; then
    echo "Error: --num-codes is not valid for ps_vqvae."
    echo "  Use --group-codebook-sizes K1,K2,K3 (comma-separated, total = K1*K2*K3)."
    exit 1
fi
if [[ -n "$GROUP_CB_SIZES_OVERRIDE" && "$VARIANT" != "ps_vqvae" ]]; then
    echo "Error: --group-codebook-sizes is only valid for ps_vqvae."
    echo "  Use --num-codes N for $VARIANT."
    exit 1
fi
if [[ -n "$NUM_CODES_OVERRIDE" ]] && ! [[ "$NUM_CODES_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --num-codes must be a positive integer; got '$NUM_CODES_OVERRIDE'."
    exit 1
fi
if [[ -n "$GROUP_CB_SIZES_OVERRIDE" ]] && ! [[ "$GROUP_CB_SIZES_OVERRIDE" =~ ^[1-9][0-9]*(,[1-9][0-9]*)+$ ]]; then
    echo "Error: --group-codebook-sizes must be comma-separated positive integers"
    echo "  (e.g., 4,8,16 or 8,8,8); got '$GROUP_CB_SIZES_OVERRIDE'."
    exit 1
fi

# ---------------------------------------------------------------------------
# Apply defaults and compute effective code-count settings
# ---------------------------------------------------------------------------
# Defaults
: "${NUM_STEPS:=30000}"
WORKDIR="$CODE2ACT_ROOT/trained_kinematics_vae_ckpts/$VARIANT"
DEFAULT_CKPT="$WORKDIR/step_$(printf '%07d' "$NUM_STEPS")"
: "${CKPT_DIR:=$DEFAULT_CKPT}"

CODES_NPZ="$WORKDIR/best_codes.npz"

# Effective code counts (user override or per-variant default).
# For vae/vqvae: NUM_CODES is the total codebook size (K-Means K or VQ K).
# For ps_vqvae:  GROUP_CB_SIZES is comma-separated K1,K2,...  total = prod(K_i).
case "$VARIANT" in
    vae|vqvae)
        NUM_CODES="${NUM_CODES_OVERRIDE:-30}"
        ;;
    ps_vqvae)
        GROUP_CB_SIZES="${GROUP_CB_SIZES_OVERRIDE:-4,4,4}"
        ;;
esac

# ---------------------------------------------------------------------------
# Build tokenize args and render args for each variant
# ---------------------------------------------------------------------------
case "$VARIANT" in
    vae)
        TOKENIZE_ARGS=(--num-codes "$NUM_CODES" --pca-dim 8)
        RENDER_ARGS=()
        ;;
    vqvae)
        # code_dim = sum(arch.branch_latent_dims) = 4+8+12 = 24 (fixed architecture).
        TOKENIZE_ARGS=(--codebook-size "$NUM_CODES" --code-dim 24)
        RENDER_ARGS=()
        ;;
    ps_vqvae)
        # group_dims must equal arch.branch_latent_dims = [4, 8, 12] (fixed architecture).
        # Convert "K1,K2,K3" -> (K1 K2 K3) array for Python nargs.
        read -ra GCS_ARRAY <<< "${GROUP_CB_SIZES//,/ }"
        TOKENIZE_ARGS=(--group-dims 4 8 12 --group-codebook-sizes "${GCS_ARRAY[@]}")
        RENDER_ARGS=(--all-groups)
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
case "$VARIANT" in
    vae)      echo "  num-codes : $NUM_CODES (K-Means clusters; no retraining needed)" ;;
    vqvae)    echo "  num-codes : $NUM_CODES (codebook size K; baked into training)" ;;
    ps_vqvae) echo "  group-cb  : $GROUP_CB_SIZES (per-group sizes; baked into training)" ;;
esac

# ---------------------------------------------------------------------------
# Stage 1 — pretrain
# For vqvae/ps_vqvae the codebook size is baked into the model; forward it
# as a Hydra override so training and tokenization stay consistent.
# For vae, codebook size is a post-hoc K-Means decision — no training change.
# ---------------------------------------------------------------------------
if ! $SKIP_PRETRAIN; then
    log "Stage 1 — pretrain $VARIANT for $NUM_STEPS steps"
    case "$VARIANT" in
        vqvae)
            run "$VENV_PYTHON" -m "c2a.codegen.kinematics_vae.$VARIANT.train" \
                train.num_steps="$NUM_STEPS" \
                train.ckpt_dir="$WORKDIR" \
                codebook.size="$NUM_CODES"
            ;;
        ps_vqvae)
            # Hydra list syntax: product_bottleneck.group_codebook_sizes=[K1,K2,K3]
            run "$VENV_PYTHON" -m "c2a.codegen.kinematics_vae.$VARIANT.train" \
                train.num_steps="$NUM_STEPS" \
                train.ckpt_dir="$WORKDIR" \
                "product_bottleneck.group_codebook_sizes=[$GROUP_CB_SIZES]"
            ;;
        vae)
            run "$VENV_PYTHON" -m "c2a.codegen.kinematics_vae.$VARIANT.train" \
                train.num_steps="$NUM_STEPS" \
                train.ckpt_dir="$WORKDIR"
            ;;
    esac
else
    log "Stage 1 — SKIPPED (--skip-pretrain)"
    # For VQ variants, warn if non-default sizes were specified: the tokenize
    # step will fail with a shape error if the sizes don't match the checkpoint.
    if [[ "$VARIANT" != "vae" && ( -n "$NUM_CODES_OVERRIDE" || -n "$GROUP_CB_SIZES_OVERRIDE" ) ]]; then
        echo "  WARNING: codebook sizes were specified with --skip-pretrain."
        echo "  The tokenize step will fail if the sizes do not match the existing checkpoint."
        echo "  Checkpoint: $CKPT_DIR"
    fi
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
# Stage 3 — top-K band figures + MP4s (kpms parity)
# ---------------------------------------------------------------------------
if ! $SKIP_RENDER; then
    log "Stage 3 — render top-K bands for $VARIANT"
    run "$VENV_PYTHON" -m "c2a.codegen.kinematics_vae.$VARIANT.render_topk_bands" \
        --codes-npz "$CODES_NPZ" \
        --output-dir "$WORKDIR" \
        --format both --fps 30 \
        "${RENDER_ARGS[@]}"
else
    log "Stage 3 — SKIPPED (--skip-render)"
fi

# ---------------------------------------------------------------------------
# Stage 4 — decoder RL (uses the unified code_source dispatch)
# ---------------------------------------------------------------------------
# Map variant slug -> code_source token. The VAE variant's dispatch
# token is "kinematics_vae" (not "vae"); the other two match the slug.
case "$VARIANT" in
    vae)      CODE_SOURCE="kinematics_vae" ;;
    vqvae)    CODE_SOURCE="vqvae" ;;
    ps_vqvae) CODE_SOURCE="ps_vqvae" ;;
esac

if ! $SKIP_DECODER; then
    log "Stage 4 — decoder RL with code_source=$CODE_SOURCE"
    run "$VENV_PYTHON" -m c2a.train_moseq_decoder \
        kpms_config.code_source="$CODE_SOURCE" \
        kpms_config.codes_path="$CODES_NPZ" \
        train_setup.run_name="kinematics_vae_${VARIANT}_$(date +%Y%m%d_%H%M%S)"
else
    log "Stage 4 — SKIPPED (--skip-decoder)"
fi

log "Pipeline complete."
