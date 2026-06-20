#!/bin/bash
# Sweep: VAE — 4 KL retrains × 6 post-hoc K tokenizations = 24 evaluable configs.
# Each KL retrain takes ~1 hr on GPU; each tokenize ~2 min.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2A_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
export C2A_ROOT
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$C2A_ROOT/.venv/lib/python3.12/site-packages:$C2A_ROOT/.venv/lib64/python3.12/site-packages:$C2A_ROOT:$C2A_ROOT/track-mjx"

# A zombie CUDA context from an earlier sibling run is holding ~34 GB of
# GPU memory; we have ~10 GB free. JAX defaults preallocate 75% of the
# total — disable that and cap allocations so we fit in what's free.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.18

BASE_OUT="$C2A_ROOT/outputs/tokenizer/trained_kinematics_vae_ckpts"
mkdir -p "$BASE_OUT/vae_sweep"

KL_VALUES=(0.001 0.005 0.01 0.05)
K_VALUES=(10 15 20 25 30 40)

echo "===== VAE SWEEP — $(date) ====="
echo "  KL values: ${KL_VALUES[*]}"
echo "  K  values: ${K_VALUES[*]}"

for kl in "${KL_VALUES[@]}"; do
    kl_tag=$(echo "$kl" | tr '.' 'p')
    ckpt_dir="$BASE_OUT/vae_sweep/kl_${kl_tag}"
    mkdir -p "$ckpt_dir"

    echo ""
    echo "===== VAE pretrain  kl=$kl  ckpt=$ckpt_dir  $(date) ====="
    python3.12 -m c2a.codegen.kinematics_vae.vae.train \
        loss_weights.kl="$kl" \
        train.ckpt_dir="$ckpt_dir" \
        2>&1 | tee "$ckpt_dir/train.log"

    # Final ckpt is the latest step_NNNNN subdir.
    last_step_dir=$(ls -d "$ckpt_dir"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "$last_step_dir" ]; then
        echo "ERROR: no step_* directory found in $ckpt_dir" >&2
        continue
    fi
    echo "  Using final ckpt: $last_step_dir"

    for k in "${K_VALUES[@]}"; do
        tok_out="$ckpt_dir/tokenize_k${k}"
        mkdir -p "$tok_out"
        echo ""
        echo "===== VAE tokenize  kl=$kl  K=$k  $(date) ====="
        python3.12 -m c2a.codegen.kinematics_vae.vae.tokenize \
            --ckpt-dir "$last_step_dir" \
            --output "$tok_out/best_codes.npz" \
            --num-codes "$k" \
            2>&1 | tee "$tok_out/tokenize.log" || \
        echo "  tokenize failed for K=$k (continuing)"
    done
done

echo ""
echo "===== VAE SWEEP COMPLETE — $(date) ====="
touch "$BASE_OUT/vae_sweep/.done"
