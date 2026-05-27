#!/bin/bash
# Sweep: VQ-VAE — 6 K × 3 commit = 18 retrains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2A_ROOT="$(dirname "$SCRIPT_DIR")"
export C2A_ROOT
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$C2A_ROOT/.venv/lib/python3.12/site-packages:$C2A_ROOT/.venv/lib64/python3.12/site-packages:$C2A_ROOT:$C2A_ROOT/track-mjx"

# Zombie CUDA context constrains us to ~10 GB free GPU; don't preallocate.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.18

BASE_OUT="$C2A_ROOT/outputs/tokenizer/trained_kinematics_vae_ckpts/vqvae_sweep"
mkdir -p "$BASE_OUT"

K_VALUES=(8 12 16 20 24 30)
COMMIT_VALUES=(0.1 0.25 0.5)

echo "===== VQ-VAE SWEEP — $(date) ====="
echo "  K values: ${K_VALUES[*]}"
echo "  commit values: ${COMMIT_VALUES[*]}"

for k in "${K_VALUES[@]}"; do
    for beta in "${COMMIT_VALUES[@]}"; do
        beta_tag=$(echo "$beta" | tr '.' 'p')
        ckpt_dir="$BASE_OUT/k${k}_commit${beta_tag}"
        mkdir -p "$ckpt_dir"

        echo ""
        echo "===== VQ-VAE  K=$k  β=$beta  $(date) ====="
        python3.12 -m c2a.codegen.kinematics_vae.vqvae.train \
            codebook.size="$k" \
            loss_weights.commit="$beta" \
            train.ckpt_dir="$ckpt_dir" \
            2>&1 | tee "$ckpt_dir/train.log" || \
        { echo "  TRAIN FAILED for K=$k β=$beta — continuing"; continue; }

        last_step_dir=$(ls -d "$ckpt_dir"/step_* 2>/dev/null | sort -V | tail -1)
        if [ -z "$last_step_dir" ]; then
            echo "  ERROR: no step_* directory in $ckpt_dir" >&2
            continue
        fi

        echo ""
        echo "===== VQ-VAE tokenize  K=$k  β=$beta  $(date) ====="
        python3.12 -m c2a.codegen.kinematics_vae.vqvae.tokenize \
            --ckpt-dir "$last_step_dir" \
            --output "$ckpt_dir/best_codes.npz" \
            --codebook-size "$k" \
            2>&1 | tee "$ckpt_dir/tokenize.log" || \
        echo "  tokenize failed for K=$k β=$beta"
    done
done

echo ""
echo "===== VQ-VAE SWEEP COMPLETE — $(date) ====="
touch "$BASE_OUT/.done"
