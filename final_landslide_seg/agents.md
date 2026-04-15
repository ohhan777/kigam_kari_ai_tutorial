# Landslide Segmentation Project

## 목표
Landslides4Sense 데이터셋에 대해 **지리공간 파운데이션 모델(Geospatial Foundation Model, GFM)** 을 활용하여 baseline(UNet, `train_v0.py`)보다 산사태(landslide) 클래스의 val/test **F1 score를 향상**시킨다.
- baseline 대비 성능 개선 여부를 1차 평가 기준으로 삼는다.
- GFM 후보: Prithvi (IBM-NASA), DOFA, Clay, DINOv3-Sat 등 Sentinel-2/SAR 대응 모델. `terratorch` / `torchgeo` / HuggingFace 기반 체크포인트를 우선 활용한다.
- 14채널(S2 10ch + S1 VV/VH + DEM + Slope) 입력을 GFM에 맞게 어떻게 매핑·투영할지(채널 선택/확장, patch embedding 재구성 등)를 명시적으로 설계한다.
- 파인튜닝 전략(frozen backbone + decoder, LoRA, full fine-tune 등)을 실험별로 비교한다.

## 데이터셋
- **경로**: `./data/landslides4sense`
- **형식**: HDF5 (image_*.h5, mask_*.h5)
- **구조**: TrainData(3,799쌍) / ValidData(245쌍) / TestData(800쌍)
- **이미지**: 128x128x14 (float64) — Sentinel-2(B1-B10), Sentinel-1 SAR(VV/VH), DEM, Slope
- **마스크**: 128x128 (uint8) — 0: 배경, 1: 산사태 (이진 분류)
- **특성**: 극심한 class imbalance (산사태 픽셀 ~1.7%)
- **정규화**: 채널별 MEAN/STD 상수 사용 (`utils/landslides4sense_dataset.py`에 정의)

## 환경
- **패키지 관리**: uv (`.venv` 내 모든 패키지 설치 및 실행)
- **Python**: 3.12
- **GPU**: NVIDIA RTX 6000 (Blackwell 아키텍쳐)
- **CUDA**: 활용
- **실행 예시**:
  - 싱글 GPU: `uv run python train_v0.py --epochs 30`
  - DDP: `uv run torchrun --nproc_per_node=N train_v0.py --epochs 30`

## 프로젝트 구조
```
my_landslide_seg/
├── models/             # 모델 정의
│   └── unet.py         # UNet (14ch 입력, 2class 출력, baseline)
├── utils/              # 유틸리티 모듈
│   ├── landslides4sense_dataset.py  # Dataset, Augmentation, MEAN/STD
│   ├── loss.py         # CE, Dice, Jaccard loss
│   ├── metrics.py      # ConfusionMatrix (IoU, P/R/F1, DDP sync)
│   └── plots.py        # 시각화 (2x2: Image/GT/Overlay/Pred)
├── train_v0.py         # 학습 (baseline: UNet + Adam + CE)
├── predict_v0.py       # 평가 (ValidData 기준)
├── weights/            # 학습된 모델 저장 (best/last)
├── outputs/            # 시각화 결과, 실험 결과 문서
├── reports/            # 실험 결과 문서
├── data/
│   └── landslides4sense -> (심볼릭 링크)
└── plan_v*.md          # 실험 계획서
```

## 코딩 컨벤션
- `argparse`로 CLI 인자 처리, `opt` 네임스페이스 사용
- `train_v0.py`과 `predict_v0.py` 형식을 따른다. 
- `wandb`로 학습 로깅
- best 모델은 F1 기준으로 저장, `module.` 접두사 없이 state_dict 저장
- 새 모델은 `models/` 디렉토리에 파일 추가
- 새 실험은 `train_v{N}.py` / `predict_v{N}.py` 로 버전 관리

## 사용 라이브러리
- PyTorch, torchvision (기본 학습/모델)
- albumentations (데이터 증강)
- h5py (데이터 로딩)
- wandb (실험 추적)
- segmentation-models-pytorch (smp), torchgeo, terratorch 등 적극 활용 가능

## 실험 결과 관리
- 각 실험(v0, v1, ...)마다 `reports/result_v{N}.md`에 결과 기록
- 성능, 전략, 기존 대비 비교 분석 포함
- 최종 결과를 `reports/overall_result.md`에 통합 업데이트
