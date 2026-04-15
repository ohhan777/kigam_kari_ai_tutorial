"""Clay v1.5 Foundation Model for Landslides4Sense segmentation.

Clay v1.5 key features:
- DynamicEmbedding: wavelength-based, accepts any number of bands
- patch_size=8: 224x224 → 28x28 = 784 patches (vs Prithvi 196)
- Encoder: dim=1024, depth=24, heads=16
- Pretrained on multi-sensor satellite data (HuggingFace: made-with-clay/Clay)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Clay v1.5 pretrained Sentinel-2 L2A wavelengths (10 bands, exact values)
# B02=0.493, B03=0.560, B04=0.665, B05=0.704, B06=0.740,
# B07=0.783, B08=0.842, B8A=0.865, B11=1.610, B12=2.190
#
# L4S 14ch: B1(0), B2(1), B3(2), B4(3), B5(4), B6(5), B7(6), B8(7),
#           B8A(8), B9(9), B11(10), B12(11), slope(12), DEM(13)
#
# Clay pretrained 10 bands correspond to L4S indices: [1,2,3,4,5,6,7,8,10,11]
# B1(idx0) and B9(idx9) are NOT in Clay pretrained → novel wavelengths

# Use Clay's exact pretrained wavelengths for matching bands
WAVELENGTHS_14CH_CLAY = [
    0.443,  # B1 - NOT pretrained (novel)
    0.493,  # B2 - Clay B02
    0.560,  # B3 - Clay B03
    0.665,  # B4 - Clay B04
    0.704,  # B5 - Clay B05
    0.740,  # B6 - Clay B06
    0.783,  # B7 - Clay B07
    0.842,  # B8 - Clay B08
    0.865,  # B8A - Clay B8A
    0.945,  # B9 - NOT pretrained (novel)
    1.610,  # B11 - Clay B11
    2.190,  # B12 - Clay B12
    3.000,  # slope - synthetic
    4.000,  # DEM - synthetic
]

# 10ch: only Clay pretrained bands (L4S indices)
CLAY_10CH_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
WAVELENGTHS_10CH_CLAY = [0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190]

# 12ch S2 only (no slope/DEM)
WAVELENGTHS_12CH_CLAY = WAVELENGTHS_14CH_CLAY[:12]


def _load_clay_encoder(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4, patch_size=8):
    """Load Clay v1.5 encoder with pretrained weights from HuggingFace."""
    from terratorch.models.backbones.clay_v15.model import Encoder
    from huggingface_hub import hf_hub_download

    encoder = Encoder(
        mask_ratio=0.0,  # no masking for inference
        patch_size=patch_size,
        shuffle=False,
        dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio,
    )

    ckpt_path = hf_hub_download("made-with-clay/Clay", filename="v1.5/clay-v1.5.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Extract encoder weights (strip 'model.encoder.' prefix)
    enc_sd = {}
    for k, v in sd.items():
        if k.startswith("model.encoder."):
            new_key = k[len("model.encoder."):]
            # Map patch_embedding → to_patch_embed (terratorch naming)
            new_key = new_key.replace("patch_embedding.", "to_patch_embed.")
            enc_sd[new_key] = v

    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    loaded = len(enc_sd) - len(unexpected)
    print(f"[Clay v1.5] Loaded {loaded} params, missing={len(missing)}, unexpected={len(unexpected)}")
    return encoder


class ClayV15Segmentor(nn.Module):
    """Clay v1.5 encoder + FPN decoder for segmentation.

    Input: 14ch at 224x224 → 784 patches (28x28 grid)
    Encoder: 24-layer transformer, dim=1024
    Decoder: Multi-depth FPN with progressive upsampling
    """

    def __init__(self, num_classes: int = 2, channel_mode: str = "14ch"):
        """channel_mode: '14ch' (all), '12ch' (S2 only), '10ch' (Clay pretrained only)"""
        super().__init__()
        self.encoder = _load_clay_encoder()
        self.channel_mode = channel_mode
        if channel_mode == "14ch":
            self.wavelengths = WAVELENGTHS_14CH_CLAY
            self.channel_indices = None  # use all 14
        elif channel_mode == "12ch":
            self.wavelengths = WAVELENGTHS_12CH_CLAY
            self.channel_indices = list(range(12))
        elif channel_mode == "10ch":
            self.wavelengths = WAVELENGTHS_10CH_CLAY
            self.channel_indices = CLAY_10CH_INDICES
        else:
            raise ValueError(f"Unknown channel_mode: {channel_mode}")
        self.patch_size = 8
        self.grid_size = 224 // self.patch_size  # 28

        embed_dim = 1024
        # Multi-scale projection from 4 transformer depths
        self.scale_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU()) for _ in range(4)
        ])
        # FPN decoder
        self.fpn_smooths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU())
            for _ in range(4)
        ])
        self.seg_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, num_classes, 1),
        )

    def _encode_multi_depth(self, datacube: dict) -> list[torch.Tensor]:
        """Extract features from 4 transformer depths (like multi-scale)."""
        cube = datacube["pixels"]
        waves = datacube["waves"]
        B, C, H, W = cube.shape

        patches, waves_encoded = self.encoder.to_patch_embed(cube, waves)
        patches = self.encoder.add_encodings(
            patches, datacube["time"], datacube["latlon"], datacube["gsd"],
        )
        # No masking (mask_ratio=0)
        cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        depth = len(self.encoder.transformer.layers)
        checkpoints = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        features = []
        for i, (attn, ff) in enumerate(self.encoder.transformer.layers):
            x = attn(x) + x
            x = ff(x) + x
            if i in checkpoints:
                features.append(self.encoder.transformer.norm(x)[:, 1:])  # remove CLS
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.channel_indices is not None:
            x = x[:, self.channel_indices]

        x_224 = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # GSD correction: 128→224 resize changes effective GSD
        # GSD = original_gsd × (original_size / resized_size) = 10 × 128/224 ≈ 5.71m
        corrected_gsd = 10.0 * 128.0 / 224.0

        datacube = {
            "pixels": x_224,
            "time": torch.zeros(B, 4, device=x.device),
            "latlon": torch.zeros(B, 4, device=x.device),
            "gsd": torch.tensor(corrected_gsd, device=x.device),
            "waves": torch.tensor(self.wavelengths, device=x.device).float(),
        }

        multi_feats = self._encode_multi_depth(datacube)  # 4x [B, 784, 1024]

        G = self.grid_size  # 28
        target_sizes = [(112, 112), (56, 56), (28, 28), (14, 14)]
        scales = []
        for tokens, proj, size in zip(multi_feats, self.scale_projs, target_sizes):
            feat = proj(tokens).reshape(B, G, G, 256).permute(0, 3, 1, 2)
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            scales.append(feat)

        # Top-down FPN
        for i in range(2, -1, -1):
            scales[i] = scales[i] + F.interpolate(scales[i + 1], size=scales[i].shape[2:],
                                                    mode="bilinear", align_corners=False)
        outs = [s(f) for f, s in zip(scales, self.fpn_smooths)]
        merged = sum(F.interpolate(o, size=outs[0].shape[2:], mode="bilinear", align_corners=False) for o in outs)
        merged = F.interpolate(merged, size=(128, 128), mode="bilinear", align_corners=False)
        return self.seg_head(merged)


def clay_v15_14ch(num_classes: int = 2) -> ClayV15Segmentor:
    """Clay v1.5 with all 14 channels (exact Clay wavelengths for S2, synthetic for slope/DEM)."""
    return ClayV15Segmentor(num_classes, channel_mode="14ch")


def clay_v15_12ch(num_classes: int = 2) -> ClayV15Segmentor:
    """Clay v1.5 with 12 S2 channels only."""
    return ClayV15Segmentor(num_classes, channel_mode="12ch")


def clay_v15_10ch(num_classes: int = 2) -> ClayV15Segmentor:
    """Clay v1.5 with only 10 Clay-pretrained S2 bands (best wavelength alignment)."""
    return ClayV15Segmentor(num_classes, channel_mode="10ch")
