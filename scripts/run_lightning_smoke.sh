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

NETRC_FILE="${HOME}/.netrc"
if [ -z "${WANDB_API_KEY:-}" ]; then
  if [ -f "$NETRC_FILE" ] && grep -q "api.wandb.ai" "$NETRC_FILE"; then
    echo "Found existing W&B login in $NETRC_FILE, continuing without prompt."
  else
    read -rsp "Enter NEW WandB API Key: " WANDB_API_KEY
    echo
    export WANDB_API_KEY
  fi
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  python - << 'PY'
import os, re, sys
k = os.environ.get("WANDB_API_KEY", "").strip()
if not re.fullmatch(r"[A-Za-z0-9_]{40,}", k):
    print("Invalid WANDB_API_KEY format. Get a fresh key from https://wandb.ai/authorize")
    sys.exit(1)
print("WANDB_API_KEY format ok")
PY
  wandb login --relogin "$WANDB_API_KEY"
else
  echo "WANDB_API_KEY not provided, using existing wandb credentials from netrc."
fi

DATASET_DIR="${DATASET_DIR:-}"

# Ignore common placeholder values and fall back to auto-discovery.
case "${DATASET_DIR}" in
  ""|"/your/actual/lora_dataset_folder"|"/path/to/lora_dataset"|"CHANGE_ME")
    DATASET_DIR=""
    ;;
esac

if [ -n "$DATASET_DIR" ] && [ ! -d "$DATASET_DIR" ]; then
  echo "Provided DATASET_DIR does not exist: $DATASET_DIR"
  echo "Falling back to automatic dataset discovery..."
  DATASET_DIR=""
fi

if [ -n "$DATASET_DIR" ] && ! find "$DATASET_DIR" -maxdepth 3 -type f -path "*/device_*/*.npy" | head -n 1 | grep -q .; then
  echo "Provided DATASET_DIR does not contain expected device_*/.npy layout: $DATASET_DIR"
  echo "Falling back to automatic dataset discovery..."
  DATASET_DIR=""
fi

if [ -z "$DATASET_DIR" ]; then
  for p in \
    /teamspace/datasets/lora_2025 \
    /teamspace/studios/this_studio/datasets/lora_2025 \
    /teamspace/studios/this_studio/data/lora_2025
  do
    if [ -d "$p" ] && find "$p" -maxdepth 3 -type f -path "*/device_*/*.npy" | head -n 1 | grep -q .; then
      DATASET_DIR="$p"
      break
    fi
  done
fi

if [ -z "$DATASET_DIR" ]; then
  FIRST_SAMPLE="$(find /teamspace -maxdepth 8 -type f -path "*/device_*/*.npy" 2>/dev/null | head -n 1 || true)"
  if [ -n "$FIRST_SAMPLE" ]; then
    DATASET_DIR="$(dirname "$(dirname "$FIRST_SAMPLE")")"
  fi
fi

if [ -z "$DATASET_DIR" ]; then
  FIRST_BIN="$(find /teamspace -maxdepth 8 -type f -iname "*.bin" 2>/dev/null | head -n 1 || true)"
  if [ -n "$FIRST_BIN" ]; then
    BIN_ROOT="$(dirname "$FIRST_BIN")"
    CONVERT_OUT="$REPO_DIR/data/converted_npy"
    echo "Found raw .bin files. Converting real captures to .npy chunks for smoke test..."
    python scripts/convert_bin_to_npy.py \
      --input-dir "$BIN_ROOT" \
      --output-dir "$CONVERT_OUT" \
      --samples-per-chunk 4096 \
      --dtype int16 \
      --max-chunks-per-device 200
    if find "$CONVERT_OUT" -maxdepth 3 -type f -path "*/device_*/*.npy" | head -n 1 | grep -q .; then
      DATASET_DIR="$CONVERT_OUT"
    fi
  fi
fi

if [ -z "$DATASET_DIR" ]; then
  echo "No LoRa dataset directory found under /teamspace"
  echo "Run one of these:"
  echo "  find /teamspace -maxdepth 8 -type f -path '*/device_*/*.npy' | head"
  echo "  find /teamspace -maxdepth 8 -type f -iname '*.bin' | head"
  echo "  export DATASET_DIR=/absolute/path/to/dataset_root"
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
