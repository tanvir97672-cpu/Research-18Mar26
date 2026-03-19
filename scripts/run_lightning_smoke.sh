#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/teamspace/studios/this_studio"
REPO_DIR="$WORKDIR/Research 18Mar26"

cd "$WORKDIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/tanvir97672-cpu/Research-18Mar26.git "Research 18Mar26"
fi

cd "$REPO_DIR"
git pull origin main

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip install wandb pyyaml

if [ -z "${WANDB_API_KEY:-}" ]; then
  read -rsp "Enter NEW WandB API Key: " WANDB_API_KEY
  echo
  export WANDB_API_KEY
fi

python - << 'PY'
import os, re, sys
k = os.environ.get("WANDB_API_KEY", "").strip()
if not re.fullmatch(r"[A-Za-z0-9_]{40,}", k):
    print("Invalid WANDB_API_KEY format. Get a fresh key from https://wandb.ai/authorize")
    sys.exit(1)
print("WANDB_API_KEY format ok")
PY

wandb login --relogin "$WANDB_API_KEY"

DATASET_DIR="${DATASET_DIR:-}"

# Ignore placeholder values often copied from examples.
if [[ "$DATASET_DIR" == *"your/actual"* ]] || [[ "$DATASET_DIR" == "/your/actual/lora_dataset_folder" ]]; then
  DATASET_DIR=""
fi

if [ -z "$DATASET_DIR" ]; then
  for p in \
    /teamspace/datasets/lora_2025 \
    /teamspace/studios/this_studio/datasets/lora_2025 \
    /teamspace/studios/this_studio/data/lora_2025
  do
    if [ -d "$p" ]; then
      DATASET_DIR="$p"
      break
    fi
  done
fi

if [ -z "$DATASET_DIR" ]; then
  SAMPLE_NPY="$(find /teamspace -maxdepth 8 -type f -path '*/device_*/*.npy' 2>/dev/null | head -n 1 || true)"
  if [ -n "$SAMPLE_NPY" ]; then
    DATASET_DIR="$(dirname "$(dirname "$SAMPLE_NPY")")"
  fi
fi

if [ -z "$DATASET_DIR" ]; then
  echo "No LoRa dataset directory found under /teamspace"
  echo "Expected layout: <root>/device_0/*.npy, <root>/device_1/*.npy, ..."
  echo "Try: find /teamspace -maxdepth 8 -type f -path '*/device_*/*.npy' | head"
  echo "Then rerun with: export DATASET_DIR=\"<root_that_contains_device_folders>\""
  exit 1
fi

python scripts/prepare_smoke_config.py --dataset-dir "$DATASET_DIR" --config configs/smoke_l4.yaml

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1

mkdir -p logs
LOG="logs/smoke_l4_$(date +%Y%m%d_%H%M%S).log"

nohup bash -lc "
set -euo pipefail
cd '/teamspace/studios/this_studio/Research 18Mar26'
python scripts/check_codebase.py
python scripts/preflight_check.py
python scripts/estimate_workload.py --config configs/smoke_l4.yaml
python scripts/train.py --config configs/smoke_l4.yaml --dry-run
python scripts/train.py --config configs/smoke_l4.yaml
" > "$LOG" 2>&1 &

echo "Smoke test started in background"
echo "Log: $LOG"
echo "Watch: tail -f $LOG"
