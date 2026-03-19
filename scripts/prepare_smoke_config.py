from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def has_npy_layout(root: Path) -> bool:
    return any(root.glob("device_*/*.npy"))


def resolve_dataset_root(root: Path) -> Path | None:
    if has_npy_layout(root):
        return root
    # Search a few levels below in case user points to a parent datasets folder.
    for depth in range(1, 4):
        pattern = "*/" * depth + "device_*/*.npy"
        hit = next(root.glob(pattern), None)
        if hit is not None:
            # hit is .../<candidate_root>/device_x/file.npy
            return hit.parent.parent
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch smoke config with dataset path and wandb enabled")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--config", default="configs/smoke_l4.yaml")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    resolved_root = resolve_dataset_root(dataset_dir)
    if resolved_root is None:
        raise SystemExit(
            "Dataset directory does not match expected layout device_*/.npy files. "
            f"Checked: {dataset_dir}"
        )
    dataset_dir = resolved_root

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
