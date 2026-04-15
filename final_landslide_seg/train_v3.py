"""v3 — DINOv3 ViT-L SAT-493M (RGB 3ch @224) + FPN decoder.

plan_v3: 사전학습 규모(493M Maxar RGB)가 채널 수보다 지배적. RGB 3ch만으로 14ch 모델을 상회.
"""
from l4s.dinov3_model import dinov3_vitl_sat
from l4s.train_gfm_advanced import train_gfm_advanced

if __name__ == "__main__":
    train_gfm_advanced(
        dinov3_vitl_sat,
        model_name="v3_dinov3_vitl_rgb224",
        encoder_lr=5e-5, new_lr=5e-4,
        epochs=80, warmup_epochs=5,
        batch_size=16,
    )
