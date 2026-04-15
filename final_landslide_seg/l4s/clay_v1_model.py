"""Clay v1 Foundation Model for Landslides4Sense segmentation.

Clay v1: patch_size=8, 256x256 native → 32x32 = 1024 patches, dim=768.
Pretrained on S2 10 bands. Requires waves (cuda tensor) and gsd.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Clay v1 pretrained S2 10-band wavelengths (exact)
CLAY_S2_WAVES = [0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190]
CLAY_S2_10CH_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # L4S → Clay band mapping

# 14ch: all L4S bands with Clay-accurate wavelengths where available
WAVES_14CH = [
    0.443, 0.493, 0.560, 0.665, 0.704, 0.740,  # B1, B2-B6
    0.783, 0.842, 0.865, 0.945,                  # B7, B8, B8A, B9
    1.610, 2.190,                                  # B11, B12
    3.0, 4.0,                                      # slope, DEM (synthetic)
]


class ClayV1Segmentor(nn.Module):
    """Clay v1 encoder (pretrained) + FPN decoder.

    256x256 input → 1024 patches (32x32 grid), dim=768.
    GSD corrected: 10m × 128/256 = 5.0m.
    """

    def __init__(self, num_classes: int = 2, use_14ch: bool = False):
        super().__init__()
        import terratorch  # noqa
        import timm

        self.encoder = timm.create_model("clay_v1_base", pretrained=True, features_only=True)
        self.use_14ch = use_14ch

        if use_14ch:
            self.wavelengths = WAVES_14CH
            self.channel_indices = None
        else:
            self.wavelengths = CLAY_S2_WAVES
            self.channel_indices = CLAY_S2_10CH_INDICES

        self.grid_size = 32  # 256 / 8
        self.corrected_gsd = 10.0 * 128.0 / 256.0  # 5.0m
        embed_dim = 768

        # Multi-scale from 4 transformer depths → FPN
        self.scale_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU()) for _ in range(4)
        ])
        self.fpn_smooths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
            for _ in range(4)
        ])
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.channel_indices is not None:
            x = x[:, self.channel_indices]

        x_256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        waves = torch.tensor(self.wavelengths, device=x.device).float()

        # Clay v1 encoder → list of [B, 1025, 768] (12 layers)
        all_feats = self.encoder(x_256, waves=waves, gsd=self.corrected_gsd)

        # Select 4 evenly-spaced depth features
        n = len(all_feats)
        indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        selected = [all_feats[i][:, 1:, :] for i in indices]  # remove CLS token

        G = self.grid_size
        target_sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]
        scales = []
        for tokens, proj, size in zip(selected, self.scale_projs, target_sizes):
            feat = proj(tokens).reshape(B, G, G, 256).permute(0, 3, 1, 2)
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            scales.append(feat)

        # Top-down FPN
        for i in range(2, -1, -1):
            scales[i] = scales[i] + F.interpolate(
                scales[i + 1], size=scales[i].shape[2:], mode="bilinear", align_corners=False
            )
        outs = [s(f) for f, s in zip(scales, self.fpn_smooths)]
        merged = sum(
            F.interpolate(o, size=outs[0].shape[2:], mode="bilinear", align_corners=False) for o in outs
        )
        merged = F.interpolate(merged, size=(128, 128), mode="bilinear", align_corners=False)
        return self.seg_head(merged)


def clay_v1_10ch(num_classes: int = 2) -> ClayV1Segmentor:
    """Clay v1 with 10 pretrained S2 bands."""
    return ClayV1Segmentor(num_classes, use_14ch=False)


def clay_v1_14ch(num_classes: int = 2) -> ClayV1Segmentor:
    """Clay v1 with all 14 channels."""
    return ClayV1Segmentor(num_classes, use_14ch=True)
