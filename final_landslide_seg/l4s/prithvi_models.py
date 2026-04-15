"""Prithvi Foundation Model wrappers for Landslides4Sense segmentation.

All Prithvi models use 6 HLS bands: B2, B3, B4, B8A, B11, B12.
From our 14ch data: indices [1, 2, 3, 8, 10, 11].
"""

from __future__ import annotations

import torch
import torch.nn as nn

# HLS band indices in the Landslides4Sense 14-channel input
HLS_INDICES = [1, 2, 3, 8, 10, 11]


class PrithviSegmentor(nn.Module):
    """Generic Prithvi + UperNet segmentation wrapper."""

    def __init__(self, backbone_name: str, num_classes: int = 2, is_swin: bool = False):
        super().__init__()
        import terratorch  # noqa: F401
        from terratorch.models import EncoderDecoderFactory

        kw: dict = {"in_chans": 6, "pretrained": True}
        if not is_swin:
            kw["num_frames"] = 1

        factory = EncoderDecoderFactory()
        self.model = factory.build_model(
            task="segmentation",
            backbone=backbone_name,
            decoder="UperNetDecoder",
            backbone_kwargs=kw,
            num_classes=num_classes,
        )
        self.band_indices = HLS_INDICES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]
        return self.model(x).output


# ── Factory functions for each variant ────────────────────────────────────────

def prithvi_v1_100(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_eo_v1_100", num_classes)

def prithvi_v2_tiny(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_eo_v2_tiny_tl", num_classes)

def prithvi_v2_100(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_eo_v2_100_tl", num_classes)

def prithvi_v2_300(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_eo_v2_300_tl", num_classes)

def prithvi_v2_600(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_eo_v2_600_tl", num_classes)

def prithvi_swin_b(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_swin_B", num_classes, is_swin=True)

def prithvi_swin_l(num_classes: int = 2) -> PrithviSegmentor:
    return PrithviSegmentor("prithvi_swin_L", num_classes, is_swin=True)
