# plan_v2 — 해상도 증강 (224x224 HiRes) + 지형 채널 추가(8ch)

## 1. 목적
v1에서 확인한 GFM의 효과를 **"가장 결정적인 단일 요인으로 알려진 입력 해상도(224x224)"** 와 결합하여 성능 변화를 측정한다. 추가로 slope/DEM 2채널을 encoder에 주입(Conv3d 6→8 확장, 새 채널 zero-init)하여 지형 정보의 기여를 분리 관찰한다.

## 2. 가설
1. 128→224 업스케일만으로도 ViT 기반 GFM은 유의미한 상승을 보일 것이다 (패치 수 64→196, 사전학습 해상도 정합).
2. slope/DEM 2채널 추가는 지형 정보로 보조 효과를 줄 것이나, 해상도 효과가 지배적일 것이다.

## 3. 이전 버전 대비 추가된 것
| 구분 | v1 | v2 |
|------|----|----|
| 입력 해상도 | 128x128 (원본) | **224x224** (bilinear upscale) |
| 입력 채널 | HLS 6ch | **8ch = HLS 6 + slope + DEM** |
| Patch embedding | 6ch pretrained 그대로 | **Conv3d 확장 (6→8)**, slope/DEM은 zero-init |
| Position embedding | 224x224 정합 (변경 없음) | interpolate 없이 사전학습 224 positional 유지 |

## 4. 구현 계획
- 파일: `train_v2.py`, `predict_v2.py`
- 채널 확장: Prithvi의 Conv3d patch embedding을 6→8로 재구성, 앞 6채널은 pretrained weight 복사·뒤 2채널(slope/DEM)은 zero-init
- 입력 전처리: 128x128 원본을 bilinear로 224x224 업스케일 후 encoder에 투입, 출력은 128x128로 재-interpolate

## 5. 검증 포인트
- v1 대비 delta 방향성 — "해상도 증가가 주 요인"이라는 가설이 성립하는지
- Precision/Recall 균형 변화 (HiRes에서 Recall이 오르는 경향이 보이는지)
- best epoch 위치와 과적합 시점 — GFM 특유의 조기 정점(7~15 에폭)이 재현되는지
- slope/DEM 채널의 zero-init이 학습 초기 encoder 표현을 교란하는지 (loss 안정성 관찰)

## 6. 보고
`reports/result_v2.md`: v0/v1 대비 delta, epoch별 F1 커브, 과적합 시점, P/R 균형 변화 정리.
