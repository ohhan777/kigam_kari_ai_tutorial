# plan_v1 — GFM 도입 (Prithvi V2 300M, HLS 6ch @128)

## 1. 목적
Baseline(UNet 14ch @128, CE, Adam — `train_v0.py`)에 대해, **지리공간 파운데이션 모델(GFM)** 의 효과를 가장 직접적으로 검증한다. Prithvi V2 300M(HLS 6밴드로 사전학습)을 **weight 수정 없이 그대로 사용**하여 "사전학습 정합성"의 이점을 본다.

## 2. 가설
GeoFM의 대규모 위성 사전학습 표현은, 동일 입력 해상도(128x128)에서도 in-domain 사전학습이 없는 UNet(ImageNet 기반 backbone도 아님)보다 산사태 분할에 우월할 것이다. 만약 성립하지 않는다면, 그 원인(해상도, pretrained-정합 방식 등)을 v2 이후 실험에서 분리 검증할 단서가 된다.

## 3. 이전 버전 대비 추가된 것
| 구분 | v0 (baseline) | v1 |
|------|---------------|----|
| 모델 | UNet (from-scratch, 14ch) | **Prithvi V2 300M** (HLS pretrained, 6ch) |
| 채널 매핑 | 14ch 그대로 | **HLS 6밴드만 선택** (B2,B3,B4,B8,B11,B12 → Blue,Green,Red,NIR,SWIR1,SWIR2) |
| 디코더 | UNet skip | **UperNetDecoder (terratorch)** |
| Optimizer | Adam(1e-3) | **Differential LR** (encoder 5e-5 / decoder 5e-4) |
| Scheduler | ReduceLROnPlateau | **Warmup(5) + Cosine** |
| Loss | CE | CE + Dice (0.5/0.5) |
| Epochs | 30 | 80 (best epoch 추적) |

## 4. 구현 계획
- 파일: `train_v1.py`, `predict_v1.py`
- 라이브러리: `terratorch`의 `EncoderDecoderFactory` 로 Prithvi V2 300M 체크포인트 로딩
- 채널 매핑: 14ch 입력에서 HLS 대응 인덱스 [1,2,3,8,10,11] 선택

## 5. 검증 포인트
- Prithvi 사전학습 weight가 HLS 밴드 순서대로 정확히 로드되는지 (로깅 확인)
- encoder freeze 없이 differential LR로 encoder도 미세하게 업데이트되는지
- best epoch 위치와 학습 안정성 — GFM의 조기 과적합 여부 관찰
- baseline(v0) 대비 Valid F1 방향성. 상승 시 "사전학습 정합성 효과"가, 미달 시 "해상도/Aug 등 보완 요소 필요"가 드러남

## 6. 보고
`reports/result_v1.md`: Valid/Test Precision/Recall/F1, best epoch, v0 대비 delta, 과적합·실패 원인 분석.
