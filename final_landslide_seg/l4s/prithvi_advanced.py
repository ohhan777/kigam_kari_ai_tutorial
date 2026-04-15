"""Advanced Prithvi architectures to close the gap with SMP UNet++.

Key problems identified:
1. 128x128 → only 64 ViT patches (8x8) — too few for dense prediction
2. UperNet decoder < UNet++ decoder for this task
3. Overfitting with 80 epochs (best at epoch 7-8)

Strategies:
A. Higher resolution: resize 128→224, get 196 patches (14x14) — 3x more spatial info
B. Multi-scale ViT features + slope/DEM CNN → custom FPN decoder with heavier decoder
C. Frozen Prithvi feature extractor → augment SMP UNet++ input
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HLS_INDICES = [1, 2, 3, 8, 10, 11]


def _build_prithvi_300m(num_classes: int = 2):
    import terratorch  # noqa: F401
    from terratorch.models import EncoderDecoderFactory
    return EncoderDecoderFactory().build_model(
        task="segmentation", backbone="prithvi_eo_v2_300_tl", decoder="UperNetDecoder",
        backbone_kwargs={"in_chans": 6, "pretrained": True, "num_frames": 1},
        num_classes=num_classes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy A: High-Resolution Prithvi + 8ch
# 128→224 resize gives 14x14=196 patches instead of 8x8=64
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviHiRes8ch(nn.Module):
    """Prithvi with 224x224 input resolution for 3x more patches."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = _build_prithvi_300m(num_classes)
        # Extend patch embed: 6→8 channels
        old_proj = self.model.encoder.patch_embed.proj
        new_proj = nn.Conv3d(8, old_proj.out_channels,
                             kernel_size=old_proj.kernel_size, stride=old_proj.stride,
                             bias=old_proj.bias is not None)
        with torch.no_grad():
            new_proj.weight.zero_()
            new_proj.weight[:, :6] = old_proj.weight
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.model.encoder.patch_embed.proj = new_proj
        self.band_indices = HLS_INDICES + [12, 13]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]  # 8ch
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out = self.model(x).output
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy B: Prithvi encoder → SMP UNet++ decoder (best decoder + best encoder)
# Extract Prithvi features, project to multi-scale, use SMP decoder
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviUNetPP8ch(nn.Module):
    """Prithvi encoder + SMP UNet++ decoder + slope/DEM side input.

    1. Prithvi encoder: 6ch HLS → multi-depth features at 8x8
    2. Learned upsampling to create 4 scales: 8x8, 16x16, 32x32, 64x64
    3. slope/DEM CNN → features at each scale
    4. UNet++ style decoder on fused multi-scale features
    """

    def __init__(self, num_classes: int = 2, embed_dim: int = 1024):
        super().__init__()
        base = _build_prithvi_300m(num_classes)
        self.encoder = base.encoder
        self.neck = base.neck

        # Create multi-scale from single-scale Prithvi features
        # Prithvi neck outputs [B, 1024, 8, 8] — we select 4 layers (6, 12, 18, 24)
        ch = [256, 256, 256, 256]
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, c, 1), nn.BatchNorm2d(c), nn.GELU(),
            ) for c in ch
        ])

        # slope/DEM branch
        self.aux_stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.aux_downs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64 if i == 0 else 64, 64, 3, stride=2, padding=1),
                          nn.BatchNorm2d(64), nn.GELU())
            for i in range(4)
        ])  # 128→64, 64→32, 32→16, 16→8

        # Decoder (FPN-style with progressive upsampling)
        fused_ch = [c + 64 for c in ch]  # prithvi + aux at each scale
        self.decoder_blocks = nn.ModuleList()
        decoder_ch = [256, 192, 128, 64]
        for i in range(4):
            in_c = fused_ch[i] if i == 3 else decoder_ch[i + 1] + fused_ch[i]  # skip + from_below
            if i == 3:
                in_c = fused_ch[3]
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_c, decoder_ch[i], 3, padding=1), nn.BatchNorm2d(decoder_ch[i]), nn.GELU(),
                nn.Conv2d(decoder_ch[i], decoder_ch[i], 3, padding=1), nn.BatchNorm2d(decoder_ch[i]), nn.GELU(),
            ))

        self.seg_head = nn.Conv2d(decoder_ch[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hls = x[:, HLS_INDICES]
        aux = x[:, [12, 13]]

        # Prithvi encoder → neck → spatial features
        enc_out = self.encoder(hls)
        neck_out = self.neck(enc_out, image_size=(128, 128))
        # Select 4 representative layers (evenly spaced)
        n = len(neck_out)
        indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        selected = [neck_out[i] for i in indices]  # 4x [B, 1024, 8, 8]

        # Project and upsample to multi-scale
        scales = []
        target_sizes = [(64, 64), (32, 32), (16, 16), (8, 8)]
        for i, (feat, proj) in enumerate(zip(selected, self.scale_projs)):
            f = proj(feat)
            f = F.interpolate(f, size=target_sizes[i], mode="bilinear", align_corners=False)
            scales.append(f)

        # Aux branch: slope/DEM at 4 scales
        aux_feat = self.aux_stem(aux)  # [B, 64, 128, 128]
        aux_scales = []
        a = aux_feat
        for down in self.aux_downs:
            a = down(a)
            aux_scales.append(a)
        # aux_scales: 64x64, 32x32, 16x16, 8x8

        # Fuse at each scale: concat prithvi + aux
        fused = [torch.cat([s, a], dim=1) for s, a in zip(scales, aux_scales)]

        # Decoder: bottom-up with skip connections
        d = self.decoder_blocks[3](fused[3])  # start from 8x8
        d = F.interpolate(d, size=fused[2].shape[2:], mode="bilinear", align_corners=False)
        d = self.decoder_blocks[2](torch.cat([d, fused[2]], dim=1))
        d = F.interpolate(d, size=fused[1].shape[2:], mode="bilinear", align_corners=False)
        d = self.decoder_blocks[1](torch.cat([d, fused[1]], dim=1))
        d = F.interpolate(d, size=fused[0].shape[2:], mode="bilinear", align_corners=False)
        d = self.decoder_blocks[0](torch.cat([d, fused[0]], dim=1))

        d = F.interpolate(d, size=(128, 128), mode="bilinear", align_corners=False)
        return self.seg_head(d)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy C: Frozen Prithvi features → augment SMP UNet++ input
# Best of both worlds: Prithvi spectral knowledge + UNet++ decoder + 14ch
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviGuidedUNetPP(nn.Module):
    """SMP UNet++ with Prithvi feature maps as additional input channels.

    1. Frozen Prithvi encoder: 6ch HLS → [B, 1024, 8, 8]
    2. Project to [B, 32, 128, 128] via learned upsampling
    3. Concat with original 14ch → [B, 46, 128, 128]
    4. SMP UNet++ ResNet34 processes all 46 channels
    """

    def __init__(self, num_classes: int = 2, prithvi_feat_ch: int = 32):
        super().__init__()
        import segmentation_models_pytorch as smp

        base = _build_prithvi_300m(num_classes)
        self.prithvi_encoder = base.encoder
        self.prithvi_neck_fn = base.neck  # neck is a function, not a module
        # Freeze Prithvi encoder
        for p in self.prithvi_encoder.parameters():
            p.requires_grad = False

        self.feat_proj = nn.Sequential(
            nn.Conv2d(1024, 128, 1), nn.GELU(),
            nn.Conv2d(128, prithvi_feat_ch, 1), nn.GELU(),
        )

        self.unetpp = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=14 + prithvi_feat_ch,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hls = x[:, HLS_INDICES]
        with torch.no_grad():
            enc_out = self.prithvi_encoder(hls)
            neck_out = self.prithvi_neck_fn(enc_out, image_size=(128, 128))
        # Use last layer features
        feat = neck_out[-1]  # [B, 1024, 8, 8]
        feat = self.feat_proj(feat)  # [B, 32, 8, 8]
        feat = F.interpolate(feat, size=(128, 128), mode="bilinear", align_corners=False)

        combined = torch.cat([x, feat], dim=1)  # [B, 46, 128, 128]
        return self.unetpp(combined)
