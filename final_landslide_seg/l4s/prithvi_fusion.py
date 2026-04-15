"""Prithvi + slope/DEM fusion strategies for Landslides4Sense.

Strategy A: Patch Embedding Extension (8ch = 6 HLS + slope + DEM)
  - Conv3d(6→1024) 를 Conv3d(8→1024) 으로 확장
  - 기존 6ch pretrained weights 유지, 새 2ch는 zero-init
  - 가장 단순하고 pretrained 정보를 최대한 보존

Strategy B: All-Band Extension (14ch = 12 S2 + slope + DEM)
  - Conv3d(6→1024) 를 Conv3d(14→1024) 으로 확장
  - 기존 6ch pretrained weights를 해당 밴드 위치에 배치, 나머지 zero-init
  - 모든 채널 정보를 활용

Strategy C: Late Fusion (Prithvi 6ch encoder + Aux CNN for slope/DEM)
  - Prithvi encoder가 6ch HLS에서 patch tokens 추출
  - 별도 경량 CNN이 slope+DEM에서 공간 특징 추출 후 token화
  - 두 token stream을 fusion하여 decoder에 전달
  - Pretrained encoder를 전혀 변형하지 않아 안정적

모든 전략은 Prithvi V2 300M TL (best performer) 기준.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HLS_INDICES = [1, 2, 3, 8, 10, 11]  # B2, B3, B4, B8A, B11, B12


def _build_base_prithvi(num_classes: int = 2):
    """Build Prithvi V2 300M TL + UperNet with pretrained weights."""
    import terratorch  # noqa: F401
    from terratorch.models import EncoderDecoderFactory
    factory = EncoderDecoderFactory()
    return factory.build_model(
        task="segmentation",
        backbone="prithvi_eo_v2_300_tl",
        decoder="UperNetDecoder",
        backbone_kwargs={"in_chans": 6, "pretrained": True, "num_frames": 1},
        num_classes=num_classes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy A: Patch Embedding Extension — 8ch (6 HLS + slope + DEM)
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviPatchExtend8ch(nn.Module):
    """Extend patch embedding from 6→8 channels, zero-init new channels."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = _build_base_prithvi(num_classes)

        # Extend Conv3d: [1024, 6, 1, 16, 16] → [1024, 8, 1, 16, 16]
        old_proj = self.model.encoder.patch_embed.proj
        new_proj = nn.Conv3d(
            8, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[:, :6] = old_proj.weight
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.model.encoder.patch_embed.proj = new_proj

        # Band order: 6 HLS + slope + DEM
        self.band_indices = HLS_INDICES + [12, 13]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]  # [B, 8, H, W]
        return self.model(x).output


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy B: All-Band Extension — 14ch (all S2 + slope + DEM)
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviPatchExtend14ch(nn.Module):
    """Extend patch embedding from 6→14 channels.

    Pretrained weights mapped to correct S2 band positions:
      HLS Blue→B2(idx1), Green→B3(idx2), Red→B4(idx3),
      NIR_Narrow→B8A(idx8), SWIR1→B11(idx10), SWIR2→B12(idx11)
    Other channels zero-initialized.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = _build_base_prithvi(num_classes)

        old_proj = self.model.encoder.patch_embed.proj
        new_proj = nn.Conv3d(
            14, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.zero_()
            # Map pretrained HLS weights to correct L4S channel positions
            # HLS band order: [Blue, Green, Red, NIR_Narrow, SWIR1, SWIR2]
            # L4S positions:  [1,    2,     3,   8,          10,    11]
            hls_to_l4s = [1, 2, 3, 8, 10, 11]
            for hls_idx, l4s_idx in enumerate(hls_to_l4s):
                new_proj.weight[:, l4s_idx] = old_proj.weight[:, hls_idx]
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.model.encoder.patch_embed.proj = new_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).output  # all 14 channels directly


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy C: Late Fusion — Prithvi 6ch + Aux CNN (slope/DEM)
# ═══════════════════════════════════════════════════════════════════════════════

class AuxSpatialEncoder(nn.Module):
    """Encode slope+DEM (2ch) into spatial features matching neck output [B, C, H/16, W/16]."""

    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim), nn.GELU(),
        )  # 128→64→32→16→8 (factor 16 downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, embed_dim, 8, 8]


class PrithviLateFusion(nn.Module):
    """Prithvi encoder + neck → spatial features, then gated fusion with slope/DEM.

    Pipeline: encoder(6ch) → neck → spatial [B,1024,8,8]
              aux_cnn(slope+DEM) → spatial [B,1024,8,8]
              gated add → decoder → head
    Pretrained encoder weights are fully preserved.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = _build_base_prithvi(num_classes)
        embed_dim = 1024
        self.aux_encoder = AuxSpatialEncoder(embed_dim=embed_dim)
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hls = x[:, HLS_INDICES]     # [B, 6, H, W]
        aux = x[:, [12, 13]]        # [B, 2, H, W]
        input_size = (x.shape[2], x.shape[3])

        # Prithvi encoder → neck → spatial features
        encoder_outputs = self.model.encoder(hls)
        neck_outputs = self.model.neck(encoder_outputs, image_size=input_size)
        # neck_outputs: list of [B, 1024, 8, 8]

        # Aux encoder → [B, 1024, 8, 8]
        aux_feat = self.aux_encoder(aux)
        gate_weight = self.gate(aux_feat)

        # Gated fusion on last 4 features (decoder inputs)
        n = len(neck_outputs)
        fused = list(neck_outputs)
        for i in range(max(0, n - 4), n):
            fused[i] = fused[i] + gate_weight * aux_feat

        # Decoder → Head
        decoded = self.model.decoder([f.clone() for f in fused])
        mask = self.model.head(decoded)
        if mask.shape[-2:] != input_size:
            mask = F.interpolate(mask, size=input_size, mode="bilinear", align_corners=False)
        return mask
