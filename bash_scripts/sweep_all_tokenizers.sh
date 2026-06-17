#!/bin/bash
# Master orchestrator for the full tokenizer sweep.
# Layout:
#   - sweep_kpms  screen: KPMS (CPU, ~30h)
#   - sweep_neural screen: VAE → VQ-VAE → PS-VQ-VAE sequentially (~41h GPU)
#   - sweep_watcher screen: waits for both to complete, then runs the
#       tokenizer-diagnostics pipeline on the new outputs.
#
# Logs:
#   outputs/tokenizer/_logs/{kpms,neural,watcher}.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C2A_ROOT="$(dirname "$SCRIPT_DIR")"
export C2A_ROOT

LOG_DIR="$C2A_ROOT/outputs/tokenizer/_logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Screen 1: KPMS (CPU)
# ---------------------------------------------------------------------------
echo "Launching screen 'sweep_kpms' …"
screen -dmS sweep_kpms bash -c "
  bash '$SCRIPT_DIR/sweep_kpms_k100.sh' 2>&1 | tee '$LOG_DIR/kpms.log'
  echo '=== sweep_kpms exited at \$(date) ==='
  sleep 86400  # keep screen alive for inspection
"

# ---------------------------------------------------------------------------
# Screen 2: Neural sweeps (GPU, serial)
# ---------------------------------------------------------------------------
echo "Launching screen 'sweep_neural' …"
screen -dmS sweep_neural bash -c "
  set -uo pipefail
  bash '$SCRIPT_DIR/sweep_vae.sh' 2>&1 | tee '$LOG_DIR/vae.log'
  bash '$SCRIPT_DIR/sweep_vqvae.sh' 2>&1 | tee '$LOG_DIR/vqvae.log'
  bash '$SCRIPT_DIR/sweep_psvqvae.sh' 2>&1 | tee '$LOG_DIR/psvqvae.log'
  touch '$C2A_ROOT/outputs/tokenizer/trained_kinematics_vae_ckpts/.neural_done'
  echo '=== sweep_neural exited at \$(date) ==='
  sleep 86400
"

# ---------------------------------------------------------------------------
# Screen 3: Watcher — waits for both, runs diagnostics
# ---------------------------------------------------------------------------
echo "Launching screen 'sweep_watcher' …"
screen -dmS sweep_watcher bash -c "
  export C2A_ROOT='$C2A_ROOT'
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export PYTHONPATH='$C2A_ROOT/.venv/lib/python3.12/site-packages:$C2A_ROOT/.venv/lib64/python3.12/site-packages:$C2A_ROOT:$C2A_ROOT/track-mjx'

  echo '=== watcher: waiting for KPMS and neural sweeps to complete ==='
  while [ ! -f '$C2A_ROOT/outputs/tokenizer/kpms_sweep/.done' ] || \
        [ ! -f '$C2A_ROOT/outputs/tokenizer/trained_kinematics_vae_ckpts/.neural_done' ]; do
    sleep 60
  done
  echo '=== watcher: both sweeps complete at \$(date), running diagnostics ==='
  cd '$C2A_ROOT'
  python3.12 -m c2a.experiments.tokenizer.run_tokenizer_diagnostics 2>&1 | tee '$LOG_DIR/diagnostics.log'
  touch '$C2A_ROOT/outputs/tokenizer/.all_done'
  echo '=== watcher: ALL DONE at \$(date) ==='
  sleep 86400
" 2>&1 | tee -a "$LOG_DIR/watcher.log"

echo ""
echo "All three screens launched. Status:"
screen -ls 2>&1 | grep -E "sweep_" || true
echo ""
echo "Log directory: $LOG_DIR"
echo "To monitor:"
echo "  tail -f $LOG_DIR/kpms.log"
echo "  tail -f $LOG_DIR/vae.log  (then vqvae.log, then psvqvae.log)"
echo "Or attach:"
echo "  screen -r sweep_kpms"
echo "  screen -r sweep_neural"
echo "  screen -r sweep_watcher"
