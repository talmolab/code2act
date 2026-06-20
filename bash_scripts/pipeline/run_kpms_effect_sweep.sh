#!/usr/bin/env bash
# =============================================================================
# run_kpms_effect_sweep.sh — effect of the KPMS tokenizer on the controller
# =============================================================================
# A CONTROLLED tokenizer comparison: fit several KPMS tokenizers that differ
# only in kappa (syllable stickiness), then train one decoder per tokenizer
# with a BYTE-IDENTICAL decoder config. Any downstream difference is then
# attributable to the tokenizer, not to the decoder.
#
#   Stage 1  fit 10 KPMS tokenizers   (K=100, 10 log-spaced kappa, arhmm, 1 seed)
#   Stage 2  generate per-setting codes  -> 10 npz + manifest.json
#   Stage 3  ONE full-budget decoder per tokenizer, identical config, only the
#            code stream varies. Emits 10 ready-to-submit job commands (default)
#            or runs them sequentially here (--launch).
#
# Usage (from the code2act/ repo root):
#   bash bash_scripts/pipeline/run_kpms_effect_sweep.sh            # fit + emit jobs
#   bash bash_scripts/pipeline/run_kpms_effect_sweep.sh --launch   # fit + run locally
#   bash bash_scripts/pipeline/run_kpms_effect_sweep.sh --skip-fit # reuse codes, re-emit jobs
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export C2A_ROOT="$CODE2ACT_ROOT"
cd "$CODE2ACT_ROOT"

PYRUN="python3.12 bash_scripts/_pyrun.py"

# --- paths (Stage 1/2) ---
CONFIG="c2a/configs/kpms_effect_sweep.yaml"
SWEEP_RESULTS="outputs/tokenizer/kpms_effect_sweep/sweep_results.json"
CODES_OUTPUT="outputs/tokenizer/kpms_effect_sweep/codes"
BALANCED_SPLIT="$CODE2ACT_ROOT/data/rodent/rodent_balanced_splits.json"
MANIFEST="$CODES_OUTPUT/manifest.json"
JOBS_FILE="outputs/tokenizer/kpms_effect_sweep/decoder_jobs.sh"
WANDB_GROUP="kpms_effect_k100"

# --- decoder config: the PRODUCTION setup, unchanged -------------------------
# Every tokenizer is decoded by the IDENTICAL production controller defined in
# c2a/configs/moseq_decoder.yaml: distillation head against
# trained_mimic_ckpts/260413_040706_764936, pretrained decoder, GRU
# [512,512,256,256], code_stack=5, continuous_latent_dim=16, cosine-annealed KL,
# num_timesteps=5e9. NOTHING in that config is overridden here — only the KPMS
# code stream varies — so any downstream difference is attributable to the
# tokenizer alone. (num_codes is also auto-set from the codes npz at load time;
# we pass it explicitly to be safe.)

# --- args --------------------------------------------------------------------
SKIP_FIT=false
LAUNCH=false
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-fit) SKIP_FIT=true ;;
        --launch)   LAUNCH=true ;;
        --config)   CONFIG="$2"; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

echo "=============================================="
echo "  KPMS-effect sweep"
echo "  config:      $CONFIG"
echo "  decoder:     production c2a/configs/moseq_decoder.yaml (unchanged; distill vs 260413)"
echo "  skip-fit:    $SKIP_FIT      launch: $LAUNCH"
echo "=============================================="

# --- Stage 1 + 2: fit tokenizers and generate codes --------------------------
if [ "$SKIP_FIT" = false ]; then
    echo "=== Stage 1: KPMS sweep (jax_enable_x64) ==="
    $PYRUN c2a.codegen.run_sweep --config "$CONFIG"

    echo "=== Stage 2: per-setting code generation ==="
    $PYRUN c2a.codegen.kpms.generate_all_codes \
        --sweep-results "$SWEEP_RESULTS" \
        --balanced-split "$BALANCED_SPLIT" \
        --output-dir "$CODES_OUTPUT"
else
    echo "=== Stage 1+2: SKIPPED (--skip-fit) ==="
fi

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found at $MANIFEST (run without --skip-fit first)" >&2
    exit 1
fi

# --- Stage 3: one decoder per tokenizer, identical config --------------------
echo "=== Stage 3: build decoder jobs (one per tokenizer) ==="
SETTINGS=$(python3.12 -c "import json,sys; print('\n'.join(sorted(json.load(open('$MANIFEST')))))")

: > "$JOBS_FILE"
echo "#!/usr/bin/env bash" >> "$JOBS_FILE"
echo "# Auto-generated decoder jobs — one full-budget run per KPMS tokenizer." >> "$JOBS_FILE"
echo "# Submit each line as a job, or run this file to execute sequentially." >> "$JOBS_FILE"
echo "set -euo pipefail; cd \"$CODE2ACT_ROOT\"" >> "$JOBS_FILE"

N=0
for setting in $SETTINGS; do
    CODES_PATH=$(python3.12 -c "import json; print(json.load(open('$MANIFEST'))['$setting']['codes_path'])")
    NUM_CODES=$(python3.12 -c "import json; print(json.load(open('$MANIFEST'))['$setting']['num_codes'])")
    RUN_NAME="kpms_effect_${setting}"

    CMD="$PYRUN c2a.train_moseq_decoder \
kpms_config.codes_path=$CODES_PATH \
network_config.num_codes=$NUM_CODES \
train_setup.run_name=$RUN_NAME \
logging_config.group_name=$WANDB_GROUP \
logging_config.exp_name=$RUN_NAME"

    echo "$CMD" >> "$JOBS_FILE"
    N=$((N + 1))
    echo "  [$N] $RUN_NAME  (codes=$CODES_PATH, num_codes=$NUM_CODES)"
done

echo ""
echo "Wrote $N decoder job(s) to: $JOBS_FILE"

if [ "$LAUNCH" = true ]; then
    echo "=== --launch: running $N decoders sequentially ==="
    bash "$JOBS_FILE"
else
    echo "Submit each line of $JOBS_FILE as a job, or re-run with --launch."
fi
