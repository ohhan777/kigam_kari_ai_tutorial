"""v2 — Prithvi V2 300M HiRes 8ch @224 (HLS + slope/DEM).

plan_v2: 해상도 128→224 + slope/DEM 2채널 추가 (Conv3d 6→8 확장).
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from l4s.prithvi_hires import _build_prithvi, _extend_patch_embed, HLS_INDICES, HIRES
from l4s.train_gfm_advanced import train_gfm_advanced


class PrithviHiRes8ch(nn.Module):
    """Prithvi V2 300M + HLS 6ch + slope/DEM 2ch at 224x224."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        model = _build_prithvi(num_classes=num_classes)
        self.model = _extend_patch_embed(model, 8)  # 6->8, new ch zero-init
        self.band_indices = HLS_INDICES + [12, 13]  # HLS + slope + DEM

    def forward(self, x):
        x = x[:, self.band_indices]
        x = F.interpolate(x, size=HIRES, mode="bilinear", align_corners=False)
        out = self.model(x).output
        return F.interpolate(out, size=(128, 128), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    train_gfm_advanced(
        PrithviHiRes8ch,
        model_name="v2_prithvi8ch_224",
        encoder_lr=5e-5, new_lr=5e-4,
        epochs=80, warmup_epochs=5,
        batch_size=16,
    )
