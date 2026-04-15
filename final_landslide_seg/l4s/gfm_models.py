"""Geospatial Foundation Model wrappers for segmentation on Landslides4Sense.

Landslides4Sense channel ordering (14 channels):
  0:B1  1:B2  2:B3  3:B4  4:B5  5:B6  6:B7  7:B8
  8:B8A 9:B9  10:B11 11:B12 12:slope 13:DEM
  (B10 is excluded from the dataset)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Channel mappings ---
# Prithvi / HLS bands: B2, B3, B4, B8A, B11, B12
HLS_INDICES = [1, 2, 3, 8, 10, 11]

# Satlas S2 bands: B4, B3, B2, B5, B6, B7, B8, B11, B12
SATLAS_INDICES = [3, 2, 1, 4, 5, 6, 7, 10, 11]

# SSL4EO S2 ALL 13 bands: B1..B8,B8A,B9,B10,B11,B12
# Our data has 12 S2 bands (no B10). Pad zero at position 10.

# Sentinel-2 center wavelengths in micrometers (matching our 12 S2 indices + slope/DEM)
WAVELENGTHS_14CH = [
    0.443, 0.490, 0.560, 0.665, 0.705, 0.740,  # B1-B6
    0.783, 0.842, 0.865, 0.945,                  # B7, B8, B8A, B9
    1.610, 2.190,                                  # B11, B12
    0.0, 0.0,                                      # slope, DEM (non-optical)
]


def _pad_b10(x: torch.Tensor) -> torch.Tensor:
    """Convert 14ch input to 13ch S2-ALL by selecting 12 S2 bands and inserting zero B10."""
    s2 = x[:, :12]  # first 12 channels
    left, right = s2[:, :10], s2[:, 10:]
    zero = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
    return torch.cat([left, zero, right], dim=1)  # [B, 13, H, W]


# ─── Simple segmentation decoder for ViT patch tokens ───────────────────────

class PatchTokenDecoder(nn.Module):
    """Upsample ViT patch tokens to pixel-level segmentation."""

    def __init__(self, embed_dim: int, num_classes: int = 2, grid_size: int = 14):
        super().__init__()
        self.grid_size = grid_size
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        x = tokens.reshape(B, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)
        x = self.decode(x)  # [B, C, grid*8, grid*8] = [B, C, 112, 112]
        return F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)


# ─── FPN decoder for hierarchical features ───────────────────────────────────

class SimpleFPN(nn.Module):
    """Lightweight FPN decoder for multi-scale features."""

    def __init__(self, in_channels_list: list[int], num_classes: int = 2, fpn_ch: int = 128):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in in_channels_list])
        self.smooths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_ch, fpn_ch, 3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU())
            for _ in in_channels_list
        ])
        self.head = nn.Sequential(
            nn.Conv2d(fpn_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        lats = [conv(f) for f, conv in zip(features, self.laterals)]
        for i in range(len(lats) - 2, -1, -1):
            lats[i] = lats[i] + F.interpolate(lats[i + 1], size=lats[i].shape[2:], mode="bilinear", align_corners=False)
        outs = [s(l) for l, s in zip(lats, self.smooths)]
        target = outs[0].shape[2:]
        merged = sum(F.interpolate(o, size=target, mode="bilinear", align_corners=False) for o in outs)
        merged = F.interpolate(merged, size=(128, 128), mode="bilinear", align_corners=False)
        return self.head(merged)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 11: DOFA-Base
# ═══════════════════════════════════════════════════════════════════════════════

class DOFASegmentor(nn.Module):
    """DOFA encoder (MAE pretrained) + patch token decoder."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights
        self.encoder = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
        self.encoder.head = nn.Identity()  # remove classifier
        self.wavelengths = WAVELENGTHS_14CH
        self.decoder = PatchTokenDecoder(embed_dim=768, num_classes=num_classes, grid_size=14)

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()
        x, _ = enc.patch_embed(x, wavelist)
        x = x + enc.pos_embed[:, 1:, :]
        cls_token = enc.cls_token + enc.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        for blk in enc.blocks:
            x = blk(x)
        if hasattr(enc, "fc_norm"):
            tokens = enc.fc_norm(x[:, 1:, :])
        else:
            tokens = enc.norm(x)[:, 1:, :]
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        tokens = self._extract_tokens(x)  # [B, 196, 768]
        return self.decoder(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 12: ViT-Small SSL4EO-S2
# ═══════════════════════════════════════════════════════════════════════════════

class ViTSSL4EOSegmentor(nn.Module):
    """ViT-Small with SSL4EO S2 MOCO pretrained + patch token decoder."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchgeo.models import vit_small_patch16_224, ViTSmall16_Weights
        self.encoder = vit_small_patch16_224(weights=ViTSmall16_Weights.SENTINEL2_ALL_MOCO)
        self.encoder.head = nn.Identity()
        self.decoder = PatchTokenDecoder(embed_dim=384, num_classes=num_classes, grid_size=14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = _pad_b10(x)  # [B, 13, H, W]
        x13 = F.interpolate(x13, size=(224, 224), mode="bilinear", align_corners=False)
        tokens = self.encoder.forward_features(x13)  # [B, 197, 384]
        tokens = tokens[:, 1:]  # remove CLS
        return self.decoder(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 13: Swin-v2-B Satlas S2
# ═══════════════════════════════════════════════════════════════════════════════

class SwinSatlasSegmentor(nn.Module):
    """Swin-v2-B with Satlas S2 pretrained + FPN decoder."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchgeo.models import swin_v2_b, get_model_weights
        w = get_model_weights("swin_v2_b")
        self.backbone = swin_v2_b(weights=w.SENTINEL2_MI_MS_SATLAS)
        self.band_indices = SATLAS_INDICES
        # Swin-v2-B feature channels at each stage
        self.fpn = SimpleFPN([128, 256, 512, 1024], num_classes=num_classes)

    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from 4 stages of Swin backbone."""
        feats = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in (1, 3, 5, 7):  # after each stage's transformer blocks
                # x is [B, H, W, C] for swin_v2; convert to [B, C, H, W]
                feats.append(x.permute(0, 3, 1, 2).contiguous())
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]  # select 9 bands
        feats = self._extract_features(x)
        return self.fpn(feats)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 14: ResNet50 SSL4EO-S2
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet50SSL4EOSegmentor(nn.Module):
    """ResNet50 with SSL4EO-S2 MOCO weights + UNet-style decoder."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchgeo.models import resnet50, ResNet50_Weights
        backbone = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)

        # Encoder stages
        self.conv1 = backbone.conv1    # expects 13 channels
        self.bn1 = backbone.bn1
        self.act1 = backbone.act1
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1   # 256ch
        self.layer2 = backbone.layer2   # 512ch
        self.layer3 = backbone.layer3   # 1024ch
        self.layer4 = backbone.layer4   # 2048ch

        self.fpn = SimpleFPN([256, 512, 1024, 2048], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pad_b10(x)  # [B, 13, H, W]
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return self.fpn([c1, c2, c3, c4])


# ═══════════════════════════════════════════════════════════════════════════════
# Model 15: Prithvi Swin B + UperNet (terratorch)
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviSwinSegmentor(nn.Module):
    """Prithvi Swin-B with UperNet decoder via terratorch. Selects 6 HLS bands."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        import terratorch  # noqa: F401 (registers models in timm)
        from terratorch.models import EncoderDecoderFactory
        factory = EncoderDecoderFactory()
        self.model = factory.build_model(
            task="segmentation",
            backbone="prithvi_swin_B",
            decoder="UperNetDecoder",
            backbone_kwargs={"in_chans": 6},
            num_classes=num_classes,
        )
        self.band_indices = HLS_INDICES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]  # select 6 HLS bands
        out = self.model(x)
        return out.output
