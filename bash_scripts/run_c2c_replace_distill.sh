#!/bin/bash
# =============================================================================
# Code2Code replace-mode decoder RL, on top of the canonical distillation
# backbone (frozen MIMIC encoder + frozen pretrained VAE decoder).
#
# Differences from the prior `vae_k30_c2c_replace_…` run (which trained
# from scratch and failed to converge):
#   - distillation head ON           (use_distillation_head=True)
#   - pretrained VAE decoder loaded  (use_pretrained_decoder=True)
#   - decoder frozen                 (freeze_decoder=True)
#   - termination criteria = default tight values (0.2 m drift, 6.0 pose L2)
#
# Differences from the current `c2c_vae_k30_additive_alpha5` run:
#   - reward_mode  : additive -> replace   (no tracking reward at all)
#   - alpha        : 5.0      -> 1.0       (replace sets absolute scale)
#   - alive_bonus  : 0.0      -> 0.1       (soft floor; base-config default)
#   - tokenizer    : vae K=30 (default)  -> vae_sweep/kl_0p001 K=25 (best from sweep)
#
# `network_config.num_codes` is auto-resolved from the codes_path at load
# time (see c2a/configs/moseq_decoder.yaml:121) and does not need overriding.
#
# Usage:
#   bash bash_scripts/run_c2c_replace_distill.sh                 # default
#   bash bash_scripts/run_c2c_replace_distill.sh code2code.alpha=2.0  # extra overrides
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$CODE2ACT_ROOT/.venv/bin/python"
export C2A_ROOT="$CODE2ACT_ROOT"

# ---------------------------------------------------------------------------
# Tokenizer (K=25, kl=0.001 — best from the 2026-05-26 VAE sweep)
# ---------------------------------------------------------------------------
TOKENIZER_DIR="outputs/tokenizer/trained_kinematics_vae_ckpts/vae_sweep/kl_0p001"
TOKENIZER_CKPT="${TOKENIZER_DIR}/step_0030000"
TOKENIZE_DIR="${TOKENIZER_DIR}/tokenize_k25"
CODES_PATH="${TOKENIZE_DIR}/best_codes.npz"
VAE_PKL="${TOKENIZE_DIR}/tokenizer.pkl"

# Sanity-check the files exist before invoking Python (cheap fail-fast).
for f in "$CODES_PATH" "$VAE_PKL" "${TOKENIZER_CKPT}"; do
  if [ ! -e "$CODE2ACT_ROOT/$f" ]; then
    echo "ERROR: missing required file: $CODE2ACT_ROOT/$f" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
"$VENV_PYTHON" -m c2a.train_moseq_decoder \
    kpms_config.code_source=kinematics_vae \
    kpms_config.codes_path="${CODES_PATH}" \
    code2code.reward_mode=replace \
    code2code.tokenizer_kind=vae \
    code2code.tokenizer_ckpt="${TOKENIZER_CKPT}" \
    code2code.vae_tokenizer_pkl="${VAE_PKL}" \
    code2code.alpha=1.0 \
    code2code.replace_alive_bonus=0.1 \
    train_setup.run_name=c2c_vae_k25_kl001_replace_distill \
    "$@"
