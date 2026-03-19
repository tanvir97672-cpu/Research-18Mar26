from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrogramEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, width_mult: float = 1.0, embedding_dim: int = 128):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        c3 = int(128 * width_mult)
        c4 = int(192 * width_mult)

        self.features = nn.Sequential(
            ConvBlock(in_ch, c1, stride=2),
            ConvBlock(c1, c1),
            ConvBlock(c1, c2, stride=2),
            ConvBlock(c2, c2),
            ConvBlock(c2, c3, stride=2),
            ConvBlock(c3, c4, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(c4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        emb = self.proj(feat)
        emb = nn.functional.normalize(emb, dim=1)
        return emb


class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)
