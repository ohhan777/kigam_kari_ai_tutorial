"""Channel selection wrappers for fair comparison experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

# 12ch: S2 bands only (no slope/DEM)
S2_INDICES = list(range(12))  # [0..11]

# 6ch: HLS bands only (same as Prithvi)
HLS_INDICES = [1, 2, 3, 8, 10, 11]  # B2, B3, B4, B8A, B11, B12


class ChannelSelectWrapper(nn.Module):
    """Wraps a model to select specific input channels from 14ch input."""

    def __init__(self, model: nn.Module, channel_indices: list[int]):
        super().__init__()
        self.model = model
        self.channel_indices = channel_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, self.channel_indices])
