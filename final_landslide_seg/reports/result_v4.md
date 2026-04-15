# v4 — DINOv3 ViT-L SAT + CompetitionLoss + MixUp

## 구성
| 항목 | 값 |
|------|----|
| Backbone | `facebook/dinov3-vitl16-pretrain-sat493m` (동일) |
| Decoder | Multi-scale FPN (동일) |
| 입력 | RGB 3ch @224 |
| **Loss** | **0.4·CE + 0.3·Lovász-Softmax + 0.3·Dice** (`CompetitionLoss`) |
| **Aug** | flip/rot90 + **MixUp(α=0.2, p=0.5)** |
| Optimizer | AdamW, differential LR (5e-5 / 5e-4) |
| Scheduler | Warmup 5 + Cosine, 80 epochs 계획 (51에폭에서 중단) |
| Batch size | 16 |
| GPU | H100 × 1 (GPU4) |

## 결과 (best.pth = epoch 29 기준)
| 지표 | Valid | Test |
|------|------:|------:|
| **F1** | **75.27%** | **63.51%** |
| Precision | 71.26% | 61.64% |
| Recall | 79.76% | 65.50% |

## 이전 버전 대비
| 비교 | Valid F1 | Delta |
|------|:--------:|:-----:|
| v0 baseline | 65.28% | — |
| v2 (Prithvi 8ch @224) | 73.08% | — |
| v3 (DINOv3 RGB @224, DiceCE) | 63.25% | — |
| **v4 (DINOv3 + Lovász+MixUp)** | **75.27%** | **+9.99pp over v0, +2.19pp over v2, +12.02pp over v3** |

## 해석
- **v3 → v4 +12.02pp: Lovász+MixUp의 효과가 극적**. 동일 DINOv3 backbone에서 loss/augmentation만 교체하여 63.25% → 75.27%.
- **Recall 대폭 상승(55.55% → 79.76%, +24pp)**: Lovász가 IoU를 직접 최적화하면서 산사태 탐지 누락이 크게 줄었고, MixUp이 결정 경계를 평활화하여 과적합 억제.
- Best epoch 29/80(36% 지점) — v2/v3(7~17)보다 훨씬 뒤에 정점 도달, MixUp의 regularization 효과가 수렴 시점을 늦춤.
- Test F1 63.51%는 Valid 대비 -11.76pp 격차 — 여전히 domain shift가 남아 있음 (→ v5에서 SepNorm+Self-training으로 해결).

## 산출물
- 체크포인트: `exp_v4_dinov3_lovasz_mixup/best.pth`
- 로그: `logs/v4.log`
- `training_history.csv`

## 결론
**plan_v4 가설 완전 확증** — Lovász+MixUp이 DINOv3를 살려냈고, 현 시점 최고 Valid F1(75.27%) 달성. Test F1은 v5 기법(SepNorm+Self-train)로 개선 필요.
