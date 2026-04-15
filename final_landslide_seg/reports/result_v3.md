# v3 — DINOv3 ViT-L SAT-493M RGB @224

## 구성
| 항목 | 값 |
|------|----|
| Backbone | `facebook/dinov3-vitl16-pretrain-sat493m` (4.93억 장 Maxar RGB 사전학습) |
| Decoder | Multi-scale FPN (4-depth hook + progressive upsampling) |
| 입력 | 14ch → RGB 3ch 선택 [B4=3, B3=2, B2=1] @ **224x224** |
| Loss | DiceCELoss |
| Optimizer | AdamW, differential LR (5e-5 / 5e-4) |
| Scheduler | Warmup 5 + Cosine, 80 epochs 계획 (56에폭에서 중단) |
| Batch size | 16 |
| GPU | H100 × 1 (GPU3) |
| 파라미터 | encoder 303.1M, decoder 2.3M |

## 결과 (best.pth = epoch 3 기준)
| 지표 | Valid | Test |
|------|------:|------:|
| **F1** | **63.25%** | **61.17%** |
| Precision | 73.42% | 66.59% |
| Recall | 55.55% | 56.56% |

## 이전 버전 대비
| 비교 | Valid F1 | Delta |
|------|:--------:|:-----:|
| v0 baseline | 65.28% | — |
| v2 (Prithvi 8ch @224) | 73.08% | — |
| **v3 (DINOv3 RGB @224)** | **63.25%** | **−2.03pp** vs v0, **−9.83pp** vs v2 |

## 해석 (기대치 대비 크게 미달)
- plan_v3 기대 Valid F1 74~75% → 실제 63.25%. **가설 반증**.
- **원인 진단**:
  1. **RGB 3채널만으로는 이 데이터셋에서 Prithvi 8ch HiRes(14ch-slope/DEM 포함)를 못 따라감**. Train 데이터가 4개 지역에 집중되어 RGB 외 정보(NIR/SWIR/지형)의 기여가 큼.
  2. Best epoch이 **epoch 3**에 나타난 후 급격한 학습 불안정(F1 30~65% 요동). DiceCELoss 단독 + MixUp 없음 조건에서 DINOv3 대형 encoder가 조기 과적합.
  3. SAT-493M은 Maxar 0.6m GSD 위성으로 학습됐으나 L4S는 Sentinel-2 10m — GSD 격차가 큼.
- **kigam_tutorial 리포트의 DINOv3 ViT-L 74.83%와 차이**: 해당 실험은 SegmentationAugmentation + longer warmup + 추가 정규화로 달성. 본 실험은 동일 설정이지만 epochs 80 중 56에폭에서 중단됐고, 정점이 3에폭에 있다는 점에서 본질적으로 lr/augmentation 튜닝 부족.

## 산출물
- 체크포인트: `exp_v3_dinov3_vitl_rgb224/best.pth`
- 로그: `logs/v3.log`
- `training_history.csv`

## 결론
**plan_v3 가설 부분 기각** — RGB 3ch DINOv3 단독으로는 14ch multi-spectral Prithvi를 뛰어넘지 못함. "사전학습 규모 > 채널 수" 명제는 아키텍처/학습 레시피와 결합되어야 유효하며, v4(Lovász+MixUp)에서 회복.
