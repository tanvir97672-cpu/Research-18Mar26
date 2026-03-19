from __future__ import annotations

import torch
from torch import nn

from rffi.models.backbones import ClassifierHead, SpectrogramEncoder


class JRFFPSCPlus(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, embedding_dim: int, width_mult: float = 1.0):
        super().__init__()
        self.encoder = SpectrogramEncoder(
            in_ch=in_ch,
            width_mult=width_mult,
            embedding_dim=embedding_dim,
        )
        self.classifier = ClassifierHead(embedding_dim=embedding_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(x)
        logits = self.classifier(emb)
        return logits, emb

    @staticmethod
    def siamese_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.norm(a - b, dim=-1)
