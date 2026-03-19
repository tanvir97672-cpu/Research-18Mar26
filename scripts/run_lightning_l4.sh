#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/Research\ 18Mar26 || cd /teamspace/studios/this_studio

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pip install wandb
read -sp "Enter your WandB API Key: " WANDB_API_KEY && export WANDB_API_KEY && echo ""

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_ENABLED=1

python scripts/check_codebase.py
python scripts/preflight_check.py
python scripts/train.py --dry-run --disable-wandb

python scripts/train.py --config configs/default.yaml
