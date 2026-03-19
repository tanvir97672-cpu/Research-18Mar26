from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RuntimeEnv:
    device: torch.device
    amp_dtype: torch.dtype


def pick_device(amp_dtype_name: str = "bf16") -> RuntimeEnv:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        bf16_supported = torch.cuda.is_bf16_supported()
        if amp_dtype_name == "bf16" and bf16_supported:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        return RuntimeEnv(device=device, amp_dtype=amp_dtype)

    return RuntimeEnv(device=torch.device("cpu"), amp_dtype=torch.float32)
