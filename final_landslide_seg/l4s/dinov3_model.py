"""DINOv3-SAT Foundation Model for Landslides4Sense segmentation.

DINOv3 (Meta, Aug 2025) pretrained on SAT-493M (493M Maxar RGB satellite images).
- RGB 3-channel input (R=B4, G=B3, B=B2 from L4S)
- ViT-L/16: dim=1024, depth=24, 303M params → full fine-tuning
- ViT-7B/16: dim=4096, depth=40, 6.7B params → frozen encoder + trainable decoder
  (full fine-tuning needs ~80GB optimizer states, exceeds single H100)
- Supports variable input sizes (must be multiple of 16): 224x224, 256x256, etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# L4S band indices: B4(Red)=3, B3(Green)=2, B2(Blue)=1
RGB_INDICES = [3, 2, 1]


class DINOv3SatSegmentor(nn.Module):
    """DINOv3-SAT encoder + multi-scale FPN decoder for segmentation.

    Input: L4S 14ch → select RGB (3ch) → resize to input_size
    Encoder: DINOv3 ViT with output_hidden_states
    Decoder: 4-depth FPN with progressive upsampling
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        load_bf16: bool = False,
        freeze_encoder: bool = False,
        input_size: int = 224,
    ):
        super().__init__()
        from transformers import AutoModel

        kwargs = {}
        if load_bf16:
            kwargs["dtype"] = torch.bfloat16
        self.encoder = AutoModel.from_pretrained(model_name, **kwargs)
        self.freeze_encoder = freeze_encoder
        self.use_bf16 = load_bf16
        self.input_size = input_size
        self.patch_size = self.encoder.config.patch_size  # 16
        self.grid_size = input_size // self.patch_size  # 14 for 224, 16 for 256

        if freeze_encoder:
            self.encoder.requires_grad_(False)

        config = self.encoder.config
        embed_dim = config.hidden_size
        depth = config.num_hidden_layers
        self.num_register = config.num_register_tokens  # 4
        self.depth = depth

        # Multi-depth checkpoint indices (4 evenly spaced)
        # hidden_states[0] = embedding output, [1..depth] = layer outputs
        self.depth_indices = [
            depth // 4, depth // 2, 3 * depth // 4, depth
        ]

        # Multi-scale projection and FPN decoder (always fp32)
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

    def _extract_multi_depth(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract patch tokens from 4 transformer depths."""
        if self.freeze_encoder:
            with torch.no_grad():
                out = self.encoder(x, output_hidden_states=True)
        else:
            out = self.encoder(x, output_hidden_states=True)
        features = []
        skip = 1 + self.num_register  # skip CLS + register tokens
        for idx in self.depth_indices:
            tokens = out.hidden_states[idx][:, skip:]  # [B, N_patches, dim]
            features.append(tokens.float())  # cast to fp32 for decoder
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        G = self.grid_size  # patch grid size (14 for 224, 16 for 256)
        rgb = x[:, RGB_INDICES]  # [B, 3, 128, 128]
        rgb_up = F.interpolate(rgb, size=(self.input_size, self.input_size),
                               mode="bilinear", align_corners=False)

        # For bf16 encoder, cast input to bf16
        if self.use_bf16:
            rgb_up = rgb_up.to(torch.bfloat16)

        multi_feats = self._extract_multi_depth(rgb_up)  # 4x [B, G*G, dim]

        target_sizes = [(G * 4, G * 4), (G * 2, G * 2), (G, G), (G // 2, G // 2)]
        scales = []
        for tokens, proj, size in zip(multi_feats, self.scale_projs, target_sizes):
            feat = proj(tokens)  # [B, G*G, 256]
            feat = feat.reshape(B, G, G, 256).permute(0, 3, 1, 2)  # [B, 256, G, G]
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            scales.append(feat)

        # Top-down FPN
        for i in range(2, -1, -1):
            scales[i] = scales[i] + F.interpolate(
                scales[i + 1], size=scales[i].shape[2:],
                mode="bilinear", align_corners=False
            )
        outs = [s(f) for f, s in zip(scales, self.fpn_smooths)]
        merged = sum(
            F.interpolate(o, size=outs[0].shape[2:], mode="bilinear", align_corners=False)
            for o in outs
        )
        merged = F.interpolate(merged, size=(128, 128), mode="bilinear", align_corners=False)
        return self.seg_head(merged)


def dinov3_vitl_sat(num_classes: int = 2) -> DINOv3SatSegmentor:
    """DINOv3 ViT-L/16 SAT-493M (303M params, dim=1024, 24 layers).
    224x224 input. Full fine-tuning with differential LR.
    """
    return DINOv3SatSegmentor(
        "facebook/dinov3-vitl16-pretrain-sat493m",
        num_classes=num_classes,
        input_size=224,
    )


def dinov3_vitl_sat_256(num_classes: int = 2) -> DINOv3SatSegmentor:
    """DINOv3 ViT-L/16 SAT-493M at native 256x256 (16x16=256 patches).
    SAT-493M was pretrained on 512x512 images, so 256x256 is within native range.
    """
    return DINOv3SatSegmentor(
        "facebook/dinov3-vitl16-pretrain-sat493m",
        num_classes=num_classes,
        input_size=256,
    )


def dinov3_vit7b_sat(num_classes: int = 2) -> DINOv3SatSegmentor:
    """DINOv3 ViT-7B/16 SAT-493M (6.7B params, dim=4096, 40 layers).
    Frozen encoder (bf16) + trainable FPN decoder (fp32).
    Full fine-tuning impossible on single H100 (optimizer states ~80GB).
    """
    return DINOv3SatSegmentor(
        "facebook/dinov3-vit7b16-pretrain-sat493m",
        num_classes=num_classes,
        load_bf16=True,
        freeze_encoder=True,
        input_size=224,
    )
