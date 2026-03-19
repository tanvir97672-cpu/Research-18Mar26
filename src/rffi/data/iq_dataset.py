from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.signal import stft
from torch.utils.data import Dataset

from rffi.config import DataConfig


@dataclass
class SampleRecord:
    path: Path
    label: int
    device_name: str


def _label_from_device_name(device_name: str) -> int:
    if device_name.startswith("device_"):
        return int(device_name.split("_")[-1])
    raise ValueError(f"Unexpected device directory format: {device_name}")


def discover_samples(cfg: DataConfig) -> list[SampleRecord]:
    root = Path(cfg.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    files = sorted(root.glob(cfg.file_glob))
    if not files:
        raise FileNotFoundError(
            f"No files found with pattern '{cfg.file_glob}' under {root}"
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    for file_path in files:
        device_name = file_path.parent.name
        grouped[device_name].append(file_path)

    records: list[SampleRecord] = []
    for device_name, paths in grouped.items():
        capped_paths = paths
        if cfg.max_files_per_device > 0:
            capped_paths = paths[: cfg.max_files_per_device]
        if cfg.sample_fraction <= 0 or cfg.sample_fraction > 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if cfg.sample_fraction < 1.0:
            keep_n = max(1, int(len(capped_paths) * cfg.sample_fraction))
            capped_paths = capped_paths[:keep_n]
        label = _label_from_device_name(device_name)
        records.extend(
            SampleRecord(path=sample_path, label=label, device_name=device_name)
            for sample_path in capped_paths
        )
    return records


def _to_complex_vector(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        vec = arr.astype(np.complex64).reshape(-1)
        return vec

    if arr.ndim == 2 and arr.shape[-1] == 2:
        return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)

    if arr.ndim == 1 and arr.size % 2 == 0:
        iq = arr.reshape(-1, 2)
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)

    raise ValueError(
        "Input .npy must be complex vector, [N,2] IQ, or flat interleaved IQ of even length"
    )


def _fix_iq_length(iq: np.ndarray, target_len: int) -> np.ndarray:
    if iq.shape[0] >= target_len:
        return iq[:target_len]
    out = np.zeros((target_len,), dtype=np.complex64)
    out[: iq.shape[0]] = iq
    return out


def _iq_to_stft(
    iq: np.ndarray,
    sample_rate: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> np.ndarray:
    _, _, zxx = stft(
        iq,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
        padded=False,
    )
    spec = np.log10(np.abs(zxx) ** 2 + 1e-12).astype(np.float32)
    return spec


def _iq_to_fft(iq: np.ndarray) -> np.ndarray:
    spec = np.fft.fftshift(np.fft.fft(iq))
    mag = np.log10(np.abs(spec) + 1e-12).astype(np.float32)
    return mag[None, :]


class LoRaIQDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: Iterable[SampleRecord], cfg: DataConfig):
        self.records = list(records)
        self.cfg = cfg
        if not self.records:
            raise ValueError("No sample records were provided to dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rec = self.records[index]
        arr = np.load(rec.path)

        if self.cfg.representation == "stft":
            iq = _to_complex_vector(arr)
            iq = _fix_iq_length(iq, self.cfg.target_iq_len)
            x = _iq_to_stft(
                iq,
                sample_rate=self.cfg.sample_rate,
                nperseg=self.cfg.stft_nperseg,
                noverlap=self.cfg.stft_noverlap,
                nfft=self.cfg.stft_nfft,
            )
            x = x[None, :, :]
        elif self.cfg.representation == "fft":
            iq = _to_complex_vector(arr)
            iq = _fix_iq_length(iq, self.cfg.target_iq_len)
            x = _iq_to_fft(iq)
        elif self.cfg.representation == "iq":
            iq = _to_complex_vector(arr)
            iq = _fix_iq_length(iq, self.cfg.target_iq_len)
            x = np.stack([np.real(iq), np.imag(iq)], axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unsupported representation: {self.cfg.representation}")

        tensor = torch.from_numpy(x).float()
        label = int(rec.label)
        return tensor, label
