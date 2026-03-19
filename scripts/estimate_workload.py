from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rffi.config import load_config
from rffi.data.iq_dataset import discover_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate workload from dataset/config")
    parser.add_argument("--config", type=str, default="configs/smoke_l4.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    records = discover_samples(cfg.data)
    counts = Counter(r.label for r in records)

    num_samples = len(records)
    num_devices = len(counts)
    steps_per_epoch = math.ceil(max(1, int(0.7 * num_samples)) / cfg.train.batch_size)

    report = {
        "config": args.config,
        "total_samples_after_fraction": num_samples,
        "num_devices_after_fraction": num_devices,
        "min_samples_per_device": min(counts.values()) if counts else 0,
        "max_samples_per_device": max(counts.values()) if counts else 0,
        "batch_size": cfg.train.batch_size,
        "epochs": cfg.train.epochs,
        "estimated_steps_per_epoch": steps_per_epoch,
        "estimated_total_steps": steps_per_epoch * cfg.train.epochs,
        "sample_fraction": cfg.data.sample_fraction,
        "max_files_per_device": cfg.data.max_files_per_device,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
