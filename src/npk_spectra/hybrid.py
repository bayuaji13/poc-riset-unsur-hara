"""Small convolution + attention regressor used by the transfer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class HybridConfig:
    channels: tuple[int, ...] = (16, 32, 64)
    kernels: tuple[int, ...] = (9, 7, 5)
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15


class HybridSpectralRegressor(nn.Module):
    def __init__(self, n_targets: int = 3, config: HybridConfig | None = None):
        super().__init__()
        self.config = config or HybridConfig()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel in zip(self.config.channels, self.config.kernels, strict=True):
            layers.extend((nn.Conv1d(in_channels, out_channels, kernel, stride=2, padding=kernel // 2), nn.BatchNorm1d(out_channels), nn.GELU()))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        layer = nn.TransformerEncoderLayer(in_channels, self.config.n_heads, in_channels * 2, self.config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, self.config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(in_channels)
        self.head = nn.Sequential(nn.Dropout(self.config.dropout), nn.Linear(in_channels, n_targets))

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(spectra.unsqueeze(1)).transpose(1, 2)
        return self.head(self.norm(self.attention(tokens)).mean(dim=1))

    def freeze_encoder(self) -> None:
        for parameter in [*self.encoder.parameters(), *self.attention.parameters(), *self.norm.parameters()]:
            parameter.requires_grad = False

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
