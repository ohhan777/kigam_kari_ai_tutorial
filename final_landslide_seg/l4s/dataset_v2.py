"""LandslideH5Dataset with optional augmentation support."""

from __future__ import annotations

from pathlib import Path

import torch

from l4s.dataset import LandslideH5Dataset


class LandslideAugDataset(LandslideH5Dataset):
    """LandslideH5Dataset wrapper that applies optional augmentation."""

    def __init__(self, data_root: Path | str, split: str, transform=None):
        super().__init__(data_root, split)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = super().__getitem__(index)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label
