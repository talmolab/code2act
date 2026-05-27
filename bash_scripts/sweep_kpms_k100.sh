#!/bin/bash
# KPMS K=100 sweep — 8 kappa × 2 model_type × 5 seeds = 80 fits.
# Runs in its own process because keypoint_moseq sets jax_enable_x64
# irreversibly per process.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2A_ROOT="$(dirname "$SCRIPT_DIR")"
export C2A_ROOT
export PYTHONPATH="$C2A_ROOT/.venv/lib/python3.12/site-packages:$C2A_ROOT/.venv/lib64/python3.12/site-packages:$C2A_ROOT:$C2A_ROOT/track-mjx"

echo "===== KPMS K=100 SWEEP — $(date) ====="

cd "$C2A_ROOT"
python3.12 -m c2a.codegen.run_sweep --config "$C2A_ROOT/c2a/configs/kpms_sweep_k100.yaml" 2>&1

echo ""
echo "===== KPMS K=100 SWEEP COMPLETE — $(date) ====="
touch "$C2A_ROOT/outputs/tokenizer/kpms_sweep/.done"
