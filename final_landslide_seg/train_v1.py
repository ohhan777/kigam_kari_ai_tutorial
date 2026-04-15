"""v1 — Prithvi V2 300M (HLS 6ch @128) + Differential LR + DiceCE.

plan_v1: GFM 도입 1단계. 해상도는 baseline과 동일(128). 사전학습 정합성(6ch HLS)의 효과만 분리.
"""
from l4s.prithvi_models import prithvi_v2_300
from l4s.train_gfm_advanced import train_gfm_advanced

if __name__ == "__main__":
    train_gfm_advanced(
        prithvi_v2_300,
        model_name="v1_prithvi6ch_128",
        encoder_lr=5e-5, new_lr=5e-4,
        epochs=80, warmup_epochs=5,
        batch_size=32,
    )
