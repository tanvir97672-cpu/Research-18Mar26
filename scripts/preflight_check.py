from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from rffi.config import load_config
from rffi.data.iq_dataset import discover_samples


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    report: dict[str, object] = {}

    report["python"] = sys.version
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_device_count"] = torch.cuda.device_count()

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        report["gpu_name"] = torch.cuda.get_device_name(idx)
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        report["total_vram_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2)

    dataset_ok = False
    dataset_message = ""
    try:
        samples = discover_samples(cfg.data)
        dataset_ok = len(samples) > 0
        dataset_message = f"found {len(samples)} samples"
    except Exception as ex:
        dataset_message = str(ex)

    report["dataset_ok"] = dataset_ok
    report["dataset_message"] = dataset_message

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
