#!/usr/bin/env bash
# Train every method of one dataset sequentially. Skips runs that already finished.
#   bash scripts/run_all.sh snapmogen|humanml3d
set -u
DATASET="${1:?usage: run_all.sh snapmogen|humanml3d}"
cd "$(dirname "$0")/.."
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "logs/$DATASET"

for cfg in configs/"$DATASET"/*.yaml; do
  stem=$(basename "$cfg" .yaml); [ "$stem" = base ] && continue
  log="logs/$DATASET/$stem.log"
  if grep -q "epoch: 0999\|epoch: 999" "$log" 2>/dev/null; then echo "skip $stem (done)"; continue; fi
  python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "[ABORT] no CUDA device before $stem" >&2; exit 1; }
  echo ">>> $(date '+%F %T') $stem"
  python3 scripts/train.py --config "$cfg" > "$log" 2>&1
  echo "<<< $(date '+%F %T') $stem (rc=$?)"
done
