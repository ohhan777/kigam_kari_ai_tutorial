# v1 — Prithvi V2 300M HLS 6ch @128 + Differential LR

## 구성
| 항목 | 값 |
|------|----|
| Backbone | `prithvi_eo_v2_300_tl` (terratorch, HLS 6밴드 사전학습) |
| Decoder | UperNetDecoder (terratorch 기본) |
| 입력 | 14ch → HLS 6ch 선택 [1,2,3,8,10,11] @ 128x128 |
| Loss | DiceCELoss (0.5 Dice + 0.5 CE) |
| Optimizer | AdamW, differential LR (encoder 5e-5 / decoder 5e-4), wd=1e-4 |
| Scheduler | Warmup 5 + Cosine, 80 epochs |
| Augmentation | SegmentationAugmentation (flip/rot90) |
| Batch size | 32 |
| GPU | H100 × 1 (GPU0) |
| 파라미터 | encoder 302.3M, new/decoder 41.1M |

## 결과 (80 epochs 완주)
| 지표 | Valid | Test |
|------|------:|------:|
| **F1** | **63.18%** | **55.01%** |
| Precision | 69.60% | 56.54% |
| Recall | 57.84% | 53.57% |

## 해석
- **Baseline(UNet 14ch @128, 65.28%) 대비 Valid −2.1pp 하락, Test −7.9pp 하락**. GFM 도입만으로는 동일 해상도(128)에서 baseline을 능가하지 못함.
- Precision이 Recall보다 크게 우세(69.6 vs 57.8) — 보수적 예측(많은 산사태 누락).
- 이는 "128x128 입력에서 ViT 패치가 64개(8x8)밖에 안 돼 위치 인코딩이 최적화되지 않기 때문"이라는 기존 분석과 일치. GFM을 살리려면 **해상도 증가가 필수**임을 확인 (→ v2).

## 산출물
- 체크포인트: `exp_v1_prithvi6ch_128/best.pth`, `last.pth`
- 로그: `logs/v1.log`
- 학습 곡선: `exp_v1_prithvi6ch_128/training_history.csv`

## 결론
**가설 부분 기각** — 단순 GFM 도입만으로는 baseline을 넘지 못한다. 해상도·pretrained 정합성을 결합해야 효과가 난다.
