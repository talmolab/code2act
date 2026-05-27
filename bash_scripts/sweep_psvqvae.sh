#!/bin/bash
# Sweep: PS-VQ-VAE — staged.
#   Stage A: 6 architectures (group_codebook_sizes)
#   Stage B: 3 decay × 3 reset_threshold on Stage A's winner
#   Stage C: 3 commit weights on Stage B's winner
# Stage B and C select the "best" by writing a marker file with the winning ckpt dir.
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

BASE_OUT="$C2A_ROOT/outputs/tokenizer/trained_kinematics_vae_ckpts/ps_vqvae_sweep"
mkdir -p "$BASE_OUT"

# Stage A — architectures (group_dims fixed at [4,8,12]; vary group_codebook_sizes)
# Format: "tag|g0,g1,g2"
declare -a STAGE_A=(
  "k_2_2_2|2,2,2"
  "k_2_2_4|2,2,4"
  "k_2_4_4|2,4,4"
  "k_4_4_4|4,4,4"
  "k_2_4_8|2,4,8"
  "k_4_4_8|4,4,8"
)

train_one_psvqvae() {
    # train_one_psvqvae <ckpt_dir> <g0,g1,g2> [decay] [reset_threshold] [commit]
    local out="$1"
    local gcs="$2"
    local decay="${3:-0.99}"
    local rt="${4:-1.0}"
    local commit="${5:-0.02}"
    mkdir -p "$out"
    # Convert "2,4,4" to "[2,4,4]" for Hydra list syntax
    local gcs_list="[${gcs}]"
    echo ""
    echo "===== PS-VQ-VAE  gcs=$gcs_list  decay=$decay  rt=$rt  commit=$commit  $(date) ====="
    python3.12 -m c2a.codegen.kinematics_vae.ps_vqvae.train \
        product_bottleneck.group_codebook_sizes="$gcs_list" \
        product_bottleneck.decay="$decay" \
        product_bottleneck.reset_threshold="$rt" \
        loss_weights.commit="$commit" \
        train.ckpt_dir="$out" \
        2>&1 | tee "$out/train.log" || \
    { echo "  TRAIN FAILED for $out — continuing"; return 1; }

    local last_step=$(ls -d "$out"/step_* 2>/dev/null | sort -V | tail -1)
    if [ -z "$last_step" ]; then
        echo "  ERROR: no step_* in $out" >&2
        return 1
    fi
    # group_codebook_sizes args for tokenize:
    local gd_str="$gcs"  # used for --group-codebook-sizes
    echo "===== PS-VQ-VAE tokenize  out=$out  $(date) ====="
    python3.12 -m c2a.codegen.kinematics_vae.ps_vqvae.tokenize \
        --ckpt-dir "$last_step" \
        --output "$out/best_codes.npz" \
        --group-dims 4 8 12 \
        --group-codebook-sizes ${gcs//,/ } \
        2>&1 | tee "$out/tokenize.log" || \
    echo "  tokenize failed"
}

# ============================================================================
# Stage A — architecture sweep
# ============================================================================
echo "===== PS-VQ-VAE STAGE A (architectures) — $(date) ====="
for entry in "${STAGE_A[@]}"; do
    tag="${entry%%|*}"
    gcs="${entry##*|}"
    train_one_psvqvae "$BASE_OUT/stageA_${tag}" "$gcs" || true
done

# Stage A winner: for now, use a heuristic — pick the architecture with
# the LARGEST active_at_0p1pct / K_total ratio (best uniformity at this
# stage of the sweep). After diagnostics runs, the diagnostics script
# will pick the actual best per the Goldilocks score. We just need a
# deterministic pick for Stage B continuation.
echo ""
echo "===== PS-VQ-VAE STAGE A done — picking winner for Stage B  $(date) ====="
WINNER_A=$(python3.12 -c "
import json
from pathlib import Path
import numpy as np

base = Path('$BASE_OUT')
best_score = -1.0
best_dir = None
best_gcs = None
for entry in [('stageA_k_2_2_2','2,2,2'), ('stageA_k_2_2_4','2,2,4'),
              ('stageA_k_2_4_4','2,4,4'), ('stageA_k_4_4_4','4,4,4'),
              ('stageA_k_2_4_8','2,4,8'), ('stageA_k_4_4_8','4,4,8')]:
    npz = base / entry[0] / 'best_codes.npz'
    if not npz.exists():
        continue
    d = np.load(str(npz), allow_pickle=False)
    if 'all_codes' not in d:
        continue
    codes = d['all_codes']
    K = int(d.get('num_states', codes.max() + 1).item() if hasattr(d.get('num_states', 0), 'item') else codes.max() + 1)
    flat = codes[codes >= 0].ravel()
    freq = np.bincount(flat, minlength=K)[:K]
    threshold = max(int(flat.size * 0.001), 1)
    active = int((freq >= threshold).sum())
    score = active / max(K, 1)
    if score > best_score:
        best_score = score
        best_dir = entry[0]
        best_gcs = entry[1]
print(f'{best_gcs}|{best_dir}|{best_score:.3f}')
" 2>&1 | tail -1)

WINNER_GCS=$(echo "$WINNER_A" | cut -d'|' -f1)
WINNER_DIR=$(echo "$WINNER_A" | cut -d'|' -f2)
echo "Stage A winner: $WINNER_DIR (gcs=$WINNER_GCS)"

# ============================================================================
# Stage B — decay × reset_threshold on Stage A's winner
# ============================================================================
echo ""
echo "===== PS-VQ-VAE STAGE B (dynamics)  winner=$WINNER_GCS  $(date) ====="
DECAYS=(0.95 0.99 0.999)
RTS=(0.5 1.0 2.0)
for decay in "${DECAYS[@]}"; do
    for rt in "${RTS[@]}"; do
        decay_tag=$(echo "$decay" | tr '.' 'p')
        rt_tag=$(echo "$rt" | tr '.' 'p')
        out="$BASE_OUT/stageB_decay${decay_tag}_rt${rt_tag}"
        train_one_psvqvae "$out" "$WINNER_GCS" "$decay" "$rt" 0.02 || true
    done
done

# Stage B winner pick (same heuristic).
echo ""
echo "===== PS-VQ-VAE STAGE B done — picking winner for Stage C  $(date) ====="
WINNER_B=$(python3.12 -c "
import json
from pathlib import Path
import numpy as np
base = Path('$BASE_OUT')
best_score = -1.0
best_decay = None
best_rt = None
for decay in ['0p95', '0p99', '0p999']:
    for rt in ['0p5', '1p0', '2p0']:
        npz = base / f'stageB_decay{decay}_rt{rt}' / 'best_codes.npz'
        if not npz.exists():
            continue
        d = np.load(str(npz), allow_pickle=False)
        if 'all_codes' not in d:
            continue
        codes = d['all_codes']
        K = int(d.get('num_states', codes.max() + 1).item() if hasattr(d.get('num_states', 0), 'item') else codes.max() + 1)
        flat = codes[codes >= 0].ravel()
        freq = np.bincount(flat, minlength=K)[:K]
        threshold = max(int(flat.size * 0.001), 1)
        active = int((freq >= threshold).sum())
        score = active / max(K, 1)
        if score > best_score:
            best_score = score
            best_decay = decay.replace('p', '.')
            best_rt = rt.replace('p', '.')
print(f'{best_decay}|{best_rt}|{best_score:.3f}')
" 2>&1 | tail -1)
WINNER_DECAY=$(echo "$WINNER_B" | cut -d'|' -f1)
WINNER_RT=$(echo "$WINNER_B" | cut -d'|' -f2)
echo "Stage B winner: decay=$WINNER_DECAY rt=$WINNER_RT"

# ============================================================================
# Stage C — commit weight on Stage B's winner
# ============================================================================
echo ""
echo "===== PS-VQ-VAE STAGE C (commit weight)  winner_arch=$WINNER_GCS decay=$WINNER_DECAY rt=$WINNER_RT  $(date) ====="
COMMITS=(0.02 0.1 0.25)
for commit in "${COMMITS[@]}"; do
    commit_tag=$(echo "$commit" | tr '.' 'p')
    out="$BASE_OUT/stageC_commit${commit_tag}"
    train_one_psvqvae "$out" "$WINNER_GCS" "$WINNER_DECAY" "$WINNER_RT" "$commit" || true
done

echo ""
echo "===== PS-VQ-VAE SWEEP COMPLETE — $(date) ====="
touch "$BASE_OUT/.done"
