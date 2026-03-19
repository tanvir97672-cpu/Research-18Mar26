from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root_dir: str
    file_glob: str = "device_*/*.npy"
    representation: str = "stft"  # stft | iq | fft
    sample_rate: int = 1_000_000
    target_iq_len: int = 4096
    sample_fraction: float = 1.0
    stft_nperseg: int = 256
    stft_noverlap: int = 128
    stft_nfft: int = 256
    max_files_per_device: int = 0
    calibration_ratio: float = 0.15


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    grad_accum_steps: int = 1
    compile_model: bool = True
    channels_last: bool = True
    amp_dtype: str = "bf16"  # bf16 | fp16
    deterministic: bool = False
    seed: int = 42
    log_every: int = 25


@dataclass
class ModelConfig:
    num_classes: int = 20
    embedding_dim: int = 128
    width_mult: float = 1.0
    top_k: int = 3
    score_alpha: float = 0.25


@dataclass
class RuntimeConfig:
    output_dir: str = "outputs"
    run_name: str = "lora_open_set"
    use_wandb: bool = False
    wandb_project: str = "lora-open-set-rffi"
    wandb_entity: str = ""


@dataclass
class AppConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig
    runtime: RuntimeConfig


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _merge_dict(raw, overrides)
    return AppConfig(
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(**raw["model"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )
