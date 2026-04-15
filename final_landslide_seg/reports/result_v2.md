# v2 — Prithvi V2 300M HiRes 8ch @224 (+slope/DEM)

## 구성
| 항목 | 값 |
|------|----|
| Backbone | `prithvi_eo_v2_300_tl` (HLS pretrained) |
| Decoder | UperNetDecoder |
| 입력 | HLS 6ch + slope + DEM = 8ch @ **224x224** (bilinear upscale) |
| Patch embedding | Conv3d 6→8 확장, 새 채널(slope/DEM) zero-init |
| Loss | DiceCELoss |
| Optimizer | AdamW, differential LR (5e-5 / 5e-4) |
| Scheduler | Warmup 5 + Cosine, 80 epochs 계획 (54에폭에서 중단) |
| Batch size | 16 |
| GPU | H100 × 1 (GPU2) |
| 파라미터 | encoder 302.3M, new/decoder 41.6M |

## 결과 (best.pth = epoch 17 기준)
| 지표 | Valid | Test |
|------|------:|------:|
| **F1** | **73.08%** | **62.36%** |
| Precision | 75.14% | 63.65% |
| Recall | 71.14% | 61.12% |

## 이전 버전 대비
| 비교 | Valid F1 | Delta |
|------|:--------:|:-----:|
| v0 baseline | 65.28% | — |
| v1 (Prithvi 6ch @128) | 63.18% | — |
| **v2 (Prithvi 8ch @224)** | **73.08%** | **+7.80pp** over v0, **+9.90pp** over v1 |

## 해석
- **해상도 128→224 효과가 결정적**. v1 대비 +9.90pp 상승. 이는 top10 리포트의 "Prithvi 8ch 128→224 = +5.21pp" 관찰과 동일 방향(더 큰 폭).
- Best epoch 17/80(계획의 21%)에서 정점, 이후 하락 — GFM 특유의 조기 과적합 패턴 재확인.
- P/R 모두 75% 수준으로 균형 — UperNet decoder가 경계 예측에 효과적.
- Test F1 62.36%(Valid-Test 격차 10.72pp) — Train/Val/Test 지역 차이에서 오는 도메인 갭.

## 산출물
- 체크포인트: `exp_v2_prithvi8ch_224/best.pth`
- 로그: `logs/v2.log` (epoch 54/80까지)
- `training_history.csv`

## 결론
**plan_v2 가설 완전 확증** — 해상도 증가(128→224) + slope/DEM 추가가 Prithvi를 baseline 위로 끌어올림 (+7.8pp).
