from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def has_npy_layout(root: Path) -> bool:
    return any(root.glob("device_*/*.npy"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch smoke config with dataset path and wandb enabled")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", default="configs/smoke_l4.yaml")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    if not has_npy_layout(dataset_dir):
        raise SystemExit(
            "Dataset directory does not match expected layout device_*/.npy files. "
            f"Checked: {dataset_dir}. "
            "Run: find <dataset_root> -maxdepth 3 -type f -path '*/device_*/*.npy' | head"
        )

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})["root_dir"] = str(dataset_dir)
    cfg.setdefault("runtime", {})["use_wandb"] = True

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Updated {cfg_path} with root_dir={dataset_dir} and use_wandb=True")


if __name__ == "__main__":
    main()
