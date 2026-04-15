# v0 — Baseline UNet (14ch @128, CE, Adam)

## 구성
| 항목 | 값 |
|------|----|
| 모델 | `models/unet.py` (14ch 입력, 2class 출력, from-scratch) |
| 입력 | 14ch @ 128x128 (원본 해상도) |
| 정규화 | 전체 통계 MEAN/STD (`utils/landslides4sense_dataset.py`) |
| Loss | CrossEntropy |
| Optimizer | Adam(lr=1e-3, wd=5e-4) |
| Scheduler | ReduceLROnPlateau(factor=0.5, patience=5) |
| Augmentation | HFlip/VFlip/Affine |
| Epochs | 30 |
| Batch size | 32 |
| AMP | O (cuda) |
| GPU | H100 × 1 (GPU1) |
| 명령 | `CUDA_VISIBLE_DEVICES=1 uv run python train_v0.py --epochs 30 --batch-size 32 --name v0_baseline_unet` |

## 결과 (Validation)
| 지표 | 값 |
|------|----:|
| **Best Valid F1** | **65.28%** |
| Precision | 65.00% |
| Recall | 65.56% |
| Mean IoU | 73.62% |
| Best epoch | **16 / 30** |
| Best val loss | 0.0415 |

## 해석
- 기준점(baseline)으로 채택. 이후 v1~v5 모든 GFM 실험은 이 F1 65.28%를 넘어야 의미가 있다.
- UNet은 30 에폭 전반에서 비교적 안정적으로 수렴(과적합 급격하지 않음). best epoch 16은 전체의 53% 지점.
- P/R 균형이 양호 (Precision 65.00 ≈ Recall 65.56).

## 산출물
- 체크포인트: `weights/v0_baseline_unet_best.pth`, `weights/v0_baseline_unet.pth`
- 로그: `logs/v0.log`
- wandb: `wandb/offline-run-20260415_115337-v0_baseline_unet/`

## 다음 버전 비교 기준
| 버전 | 목표 F1 | 초과해야 할 delta |
|:---:|:------:|:-----------------:|
| v1 (Prithvi 6ch @128) | 65~67% | +0~2pp |
| v2 (Prithvi 8ch @224) | 70~71% | +5~6pp |
| v3 (DINOv3 ViT-L @224) | 74~75% | +9~10pp |
| v4 (+Lovász+MixUp) | 76~77% | +11~12pp |
| v5 (+SepNorm+Self-train) | 77%+ | +12pp 이상 |
