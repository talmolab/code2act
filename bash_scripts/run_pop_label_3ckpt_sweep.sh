#!/bin/bash
# =============================================================================
# v8 sweep: run v7-style population_label_patterns on the three new ckpts
#           with two init modes each = 6 folders total.
# =============================================================================
# Per-ckpt mode codes + Format-B patterns live in the override configs:
#   c2a/experiments/configs/pop_label_180723.yaml  (vqvae k=8)
#   c2a/experiments/configs/pop_label_180727.yaml  (ps_vqvae stageA k_2_2_4)
#   c2a/experiments/configs/pop_label_180852.yaml  (VAE k=25 kl=0.001)
#
# Each ckpt × init_mode gets its own folder:
#   outputs/population/pop_label_180723_same_pos/
#   outputs/population/pop_label_180723_walking_pose/
#   outputs/population/pop_label_180727_same_pos/
#   outputs/population/pop_label_180727_walking_pose/
#   outputs/population/pop_label_180852_same_pos/
#   outputs/population/pop_label_180852_walking_pose/
#
# Each folder = 24 videos (12 cells × 2 formats), 750 frames @ 50 fps.
# Total budget: ~3h data + ~18h render = ~21h sequential.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE2ACT_ROOT="$(dirname "$SCRIPT_DIR")"
export C2A_ROOT="$CODE2ACT_ROOT"
cd "$CODE2ACT_ROOT"

CKPTS=("180723" "180727" "180852")
INIT_MODES=("same_pos" "walking_pose")

# --- Stage 1: data generation, all 6 jobs serial -----------------------------
for CKPT in "${CKPTS[@]}"; do
  for MODE in "${INIT_MODES[@]}"; do
    OUT="outputs/population/pop_label_${CKPT}_${MODE}"
    mkdir -p "$OUT"
    echo "============================================================"
    echo " DATA   ckpt=$CKPT init=$MODE  →  $OUT"
    echo "============================================================"
    python3.12 bash_scripts/_pyrun.py c2a.experiments.population.run_population_label_patterns \
        --config-name="pop_label_${CKPT}" \
        output.base_dir="$OUT" \
        population.init_mode="$MODE" \
        2>&1 | tee "$OUT/run.log"
  done
done

# --- Stage 2: video renders, all 6 jobs serial -------------------------------
for CKPT in "${CKPTS[@]}"; do
  for MODE in "${INIT_MODES[@]}"; do
    OUT="outputs/population/pop_label_${CKPT}_${MODE}"
    echo "============================================================"
    echo " RENDER ckpt=$CKPT init=$MODE  →  $OUT"
    echo "============================================================"
    python3.12 bash_scripts/_pyrun.py c2a.experiments.population.render_population_v5_videos \
        output.base_dir="$OUT" \
        2>&1 | tee "$OUT/run_render.log"
  done
done

echo "============================================================"
echo " ALL 6 FOLDERS COMPLETE"
echo "============================================================"
