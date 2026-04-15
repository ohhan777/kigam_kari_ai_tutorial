"""Augmentation transforms for segmentation (image + mask pairs)."""

from __future__ import annotations

import random

import torch


class SegmentationAugmentation:
    """Random flip + rotation augmentation for image-mask pairs."""

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            image = torch.flip(image, [-1])
            mask = torch.flip(mask, [-1])
        if random.random() > 0.5:
            image = torch.flip(image, [-2])
            mask = torch.flip(mask, [-2])
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
        return image, mask
