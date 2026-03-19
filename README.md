# Real-Only Open-Set LoRa RFFI (L4 Optimized)

This codebase is prepared for Lightning AI with an Nvidia L4 GPU and follows a strict real-data-only policy.

## What is implemented

- Open-set pipeline with known-device ID and unknown-device rejection.
- Multi-candidate top-K verification with score fusion (beyond single-argmax JRFFP-SC behavior).
- Calibration-only threshold selection to avoid test leakage.
- L4 throughput optimizations: AMP, TF32, pinned memory, persistent workers, channels-last, optional `torch.compile`.
- Preflight and full codebase checks to fail fast before expensive training runs.

## Expected dataset layout

The loader currently expects `.npy` per packet/segment:

- `device_0/*.npy`
- `device_1/*.npy`
- ...

Each `.npy` can be:

- complex IQ vector, or
- shape `[N, 2]` IQ (I,Q), or
- flat interleaved IQ `[2N]`.

## Run order

1. `scripts/preflight_check.py`
2. `scripts/check_codebase.py`
3. `scripts/train.py --dry-run`
4. full training

## Lightning L4 smoke test (lowest-cost real-only)

- Use `configs/smoke_l4.yaml`.
- This keeps only a tiny real subset via:
	- `sample_fraction: 0.01`
	- `max_files_per_device: 20`
- No synthetic generation/augmentation is used.

Recommended order for smoke testing:

1. `python scripts/check_codebase.py`
2. `python scripts/preflight_check.py`
3. `python scripts/estimate_workload.py --config configs/smoke_l4.yaml`
4. `python scripts/train.py --config configs/smoke_l4.yaml --dry-run --disable-wandb`
5. `python scripts/train.py --config configs/smoke_l4.yaml --disable-wandb`

If auto-detection does not find your dataset path on Lightning, set it explicitly:

`export DATASET_DIR="/path/to/root_with_device_folders"`

Required dataset layout:

- `<root>/device_0/*.npy`
- `<root>/device_1/*.npy`
- ...

## Notes

- No synthetic augmentations are used by this codebase.
- If your DataPort files are `.bin`, add a converter script to produce `.npy` windows first.
- Start with `max_files_per_device` in `configs/default.yaml` to control cost.
