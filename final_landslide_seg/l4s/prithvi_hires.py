"""224x224 High-Resolution variants for Prithvi and other GFMs.

Core insight: 128→224 resize gives 64→196 ViT patches (3x), which was
the key factor in reaching 70%+ F1. Apply this to all promising strategies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HLS_INDICES = [1, 2, 3, 8, 10, 11]
HIRES = (224, 224)

WAVELENGTHS_14CH = [
    0.443, 0.490, 0.560, 0.665, 0.705, 0.740,
    0.783, 0.842, 0.865, 0.945, 1.610, 2.190,
    0.0, 0.0,
]


def _build_prithvi(backbone: str = "prithvi_eo_v2_300_tl", num_classes: int = 2):
    import terratorch  # noqa: F401
    from terratorch.models import EncoderDecoderFactory
    return EncoderDecoderFactory().build_model(
        task="segmentation", backbone=backbone, decoder="UperNetDecoder",
        backbone_kwargs={"in_chans": 6, "pretrained": True, "num_frames": 1},
        num_classes=num_classes,
    )


def _extend_patch_embed(model, new_in_ch: int, pretrained_mapping: dict[int, int] | None = None):
    """Extend Conv3d patch embedding, mapping pretrained weights to correct positions."""
    old = model.encoder.patch_embed.proj
    new = nn.Conv3d(new_in_ch, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        if pretrained_mapping:
            for src_idx, dst_idx in pretrained_mapping.items():
                new.weight[:, dst_idx] = old.weight[:, src_idx]
        else:
            new.weight[:, :old.in_channels] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.encoder.patch_embed.proj = new
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HiRes 14ch — All bands at 224x224
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviHiRes14ch(nn.Module):
    """All 14 channels at 224x224. Pretrained weights mapped to correct S2 positions."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        model = _build_prithvi(num_classes=num_classes)
        # Map: HLS[0→Blue]=L4S[1], HLS[1→Green]=L4S[2], HLS[2→Red]=L4S[3],
        #       HLS[3→NIR]=L4S[8], HLS[4→SWIR1]=L4S[10], HLS[5→SWIR2]=L4S[11]
        mapping = {0: 1, 1: 2, 2: 3, 3: 8, 4: 10, 5: 11}
        self.model = _extend_patch_embed(model, 14, mapping)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=HIRES, mode="bilinear", align_corners=False)
        out = self.model(x).output
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HiRes Late Fusion — Prithvi 6ch@224 + AuxCNN slope/DEM@128
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviHiResLateFusion(nn.Module):
    """Prithvi encoder at 224x224 (high-res) + slope/DEM aux at native 128x128.

    Aux features upsampled to match Prithvi neck output, then gated fusion.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = _build_prithvi(num_classes=num_classes)
        embed_dim = 1024
        self.aux_net = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim), nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hls = x[:, HLS_INDICES]
        aux = x[:, [12, 13]]

        hls_224 = F.interpolate(hls, size=HIRES, mode="bilinear", align_corners=False)
        enc_out = self.model.encoder(hls_224)
        neck_out = self.model.neck(enc_out, image_size=HIRES)  # list of [B,1024,14,14]

        aux_feat = self.aux_net(aux)  # [B,1024,8,8]
        n = len(neck_out)
        fused = list(neck_out)
        for i in range(max(0, n - 4), n):
            target_size = fused[i].shape[2:]
            af = F.interpolate(aux_feat, size=target_size, mode="bilinear", align_corners=False)
            fused[i] = fused[i] + self.gate(af) * af

        decoded = self.model.decoder([f.clone() for f in fused])
        mask = self.model.head(decoded)
        return F.interpolate(mask, size=(128, 128), mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HiRes Multi-scale FPN 8ch — Strategy B at 224x224
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviHiResFPN8ch(nn.Module):
    """Strategy B (multi-scale FPN + slope/DEM CNN) at 224x224 resolution."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from l4s.prithvi_advanced import PrithviUNetPP8ch
        self.inner = PrithviUNetPP8ch(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample spectral bands to 224, keep slope/DEM at 128 for aux branch
        hls = x[:, HLS_INDICES]
        aux = x[:, [12, 13]]
        hls_224 = F.interpolate(hls, size=HIRES, mode="bilinear", align_corners=False)
        # Reconstruct a "fake" 14ch tensor for inner model (it selects HLS+aux)
        # The inner model expects 14ch input but we handle channels specially
        # Just resize full input
        x_224 = F.interpolate(x, size=HIRES, mode="bilinear", align_corners=False)
        out = self.inner(x_224)
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DOFA HiRes + Multi-scale FPN decoder
# ═══════════════════════════════════════════════════════════════════════════════

class DOFAHiResFPN(nn.Module):
    """DOFA (14ch, wavelength-aware) at 224x224 + multi-scale FPN decoder."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchgeo.models import dofa_base_patch16_224, DOFABase16_Weights
        self.encoder = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
        self.encoder.head = nn.Identity()
        self.wavelengths = WAVELENGTHS_14CH
        embed_dim = 768

        # Multi-scale projection from patch tokens
        self.scale_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU()) for _ in range(4)
        ])
        # FPN decoder
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
            for _ in range(4)
        ])
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def _extract_multi_depth(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract tokens from 4 transformer depths."""
        enc = self.encoder
        wavelist = torch.tensor(self.wavelengths, device=x.device).float()
        px, _ = enc.patch_embed(x, wavelist)
        px = px + enc.pos_embed[:, 1:, :]
        cls_tok = (enc.cls_token + enc.pos_embed[:, :1, :]).expand(px.shape[0], -1, -1)
        px = torch.cat([cls_tok, px], dim=1)

        depth = len(enc.blocks)
        checkpoints = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        features = []
        for i, blk in enumerate(enc.blocks):
            px = blk(px)
            if i in checkpoints:
                tokens = px[:, 1:]  # remove CLS
                if hasattr(enc, "fc_norm"):
                    tokens = enc.fc_norm(tokens)
                features.append(tokens)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_224 = F.interpolate(x, size=HIRES, mode="bilinear", align_corners=False)
        multi_tokens = self._extract_multi_depth(x_224)  # 4x [B, 196, 768]

        B = x.shape[0]
        target_sizes = [(56, 56), (28, 28), (14, 14), (7, 7)]
        scales = []
        for tokens, proj, size in zip(multi_tokens, self.scale_projs, target_sizes):
            feat = proj(tokens)  # [B, 196, 256]
            feat = feat.reshape(B, 14, 14, 256).permute(0, 3, 1, 2)  # [B,256,14,14]
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            scales.append(feat)

        # Top-down FPN
        for i in range(2, -1, -1):
            scales[i] = scales[i] + F.interpolate(scales[i + 1], size=scales[i].shape[2:],
                                                    mode="bilinear", align_corners=False)
        outs = [conv(s) for s, conv in zip(scales, self.fpn_convs)]
        merged = sum(F.interpolate(o, size=outs[0].shape[2:], mode="bilinear", align_corners=False) for o in outs)
        merged = F.interpolate(merged, size=(128, 128), mode="bilinear", align_corners=False)
        return self.seg_head(merged)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Prithvi V2 600M HiRes 8ch — Larger model benefits from more patches
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviV2_600M_HiRes8ch(nn.Module):
    """Prithvi V2 600M at 224x224 + 8ch. Larger model needs more patches."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        model = _build_prithvi("prithvi_eo_v2_600_tl", num_classes)
        self.model = _extend_patch_embed(model, 8)
        self.band_indices = HLS_INDICES + [12, 13]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]
        x = F.interpolate(x, size=HIRES, mode="bilinear", align_corners=False)
        out = self.model(x).output
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HiRes 6ch — Pure resolution effect (control experiment)
# ═══════════════════════════════════════════════════════════════════════════════

class PrithviHiRes6ch(nn.Module):
    """Prithvi 300M at 224x224 with 6ch HLS only. Isolates resolution effect."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = _build_prithvi(num_classes=num_classes)
        self.band_indices = HLS_INDICES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.band_indices]
        x = F.interpolate(x, size=HIRES, mode="bilinear", align_corners=False)
        out = self.model(x).output
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)
