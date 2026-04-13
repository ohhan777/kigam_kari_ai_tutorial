# 지리공간 파운데이션 모델 실습 튜토리얼

한국지질자원연구원(KIGAM) 주관 **지리공간 파운데이션 모델** 실습 강의 자료입니다.
TorchGeo와 TerraTorch를 활용하여 위성영상 데이터를 다루는 기초부터, 파운데이션 모델의 미세조정(fine-tuning)을 통한 다운스트림 태스크 성능 향상까지의 과정을 다룹니다.

## 디렉토리 구조

```
kigam_kari_ai_tutorial/
├── torchgeo/                # TorchGeo 기본 사용법
│   ├── landsat_and_cdl.py   # Landsat 8/9 + CDL 데이터셋 결합 및 시각화
│   ├── sentinel2_ndvi.py    # Sentinel-2 데이터 로드 및 NDVI 산출
│   ├── vhr10_data.py        # VHR-10 객체 탐지 데이터셋 로드 및 시각화
│   ├── train_inria.py       # Inria 건물 세그멘테이션 학습 (UNet + ResNet34)
│   ├── predict_inria.py     # Inria 학습 모델 추론 및 시각화
│   ├── utils/
│   │   └── plots.py         # 위성영상·마스크·탐지 결과 시각화 유틸리티
│   └── pyproject.toml
├── terratorch/              # TerraTorch를 활용한 파운데이션 모델 미세조정
│   ├── train.py             # Prithvi V2 기반 화재 흔적 세그멘테이션 학습 (Python)
│   ├── predict.py           # 학습 모델 검증 및 추론 시각화
│   ├── prithvi_v2_eo_300_tl_unet_burnscars.yaml  # YAML 기반 학습 설정
│   └── pyproject.toml
└── landslide_seg/           # 산사태 세그멘테이션 (업데이트 예정)
    └── (Landslides4Sense 데이터셋 기반 파운데이션 모델 미세조정)
```

## 실습 내용

### 1. TorchGeo 기본 사용법 (`torchgeo/`)

[TorchGeo](https://github.com/microsoft/torchgeo)는 지리공간 데이터를 위한 PyTorch 확장 라이브러리로, 위성영상 데이터셋 로드·샘플링·학습 파이프라인을 제공합니다.

| 스크립트 | 설명 |
|---|---|
| `landsat_and_cdl.py` | Landsat 8/9 영상과 USDA CDL(Cropland Data Layer) 토지피복도를 `IntersectionDataset`으로 결합하고, `RandomGeoSampler`를 이용해 학습용 패치를 추출합니다. |
| `sentinel2_ndvi.py` | Sentinel-2 `.SAFE` 데이터를 로드하고, RGB 시각화 및 NDVI(정규식생지수)를 산출하여 저장합니다. |
| `vhr10_data.py` | NWPU VHR-10 초고해상도 객체 탐지 데이터셋을 로드하고, 바운딩 박스 및 인스턴스 마스크를 시각화합니다. |
| `train_inria.py` | Inria Aerial Image Labeling 데이터셋으로 UNet(ResNet34 백본, ImageNet 사전학습)을 학습합니다. Lightning 기반 학습 파이프라인을 구성합니다. |
| `predict_inria.py` | 학습된 체크포인트를 로드하여 검증 데이터에 대한 건물 세그멘테이션 결과를 시각화합니다. |

### 2. TerraTorch를 활용한 파운데이션 모델 미세조정 (`terratorch/`)

[TerraTorch](https://github.com/IBM/terratorch)는 지리공간 파운데이션 모델(Prithvi 등)의 미세조정을 위한 프레임워크입니다.
HLS Burn Scars(화재 흔적) 데이터셋에 대해 **Prithvi V2 (ViT 기반)** 파운데이션 모델을 UNet 디코더와 결합하여 세그멘테이션 태스크로 미세조정하는 과정을 다룹니다.

| 파일 | 설명 |
|---|---|
| `train.py` | Python 코드로 데이터 모듈, 모델(Prithvi V2 300M + UNet 디코더), 학습기를 직접 구성합니다. 백본을 freeze한 상태에서 디코더만 학습하여 효율적으로 미세조정합니다. |
| `predict.py` | 학습된 체크포인트를 로드하여 검증 데이터에 대한 화재 흔적 세그멘테이션 결과를 시각화합니다. |
| `prithvi_v2_eo_300_tl_unet_burnscars.yaml` | 동일한 학습 파이프라인을 YAML 설정 파일로 정의한 예시입니다. `terratorch fit -c <config>.yaml` 명령으로 실행할 수 있습니다. |

### 3. 산사태 세그멘테이션 (`landslide_seg/`) — 업데이트 예정

[Landslides4Sense](https://www.iarai.ac.at/landslide4sense/) 데이터셋을 기반으로 지리공간 파운데이션 모델을 미세조정하여 산사태 탐지 성능을 향상시키는 기법을 다룰 예정입니다.

## 환경 설정

각 디렉토리에 [uv](https://docs.astral.sh/uv/) 환경 구성을 위한 `pyproject.toml`이 포함되어 있습니다.

## 참고 자료

- [TorchGeo Documentation](https://torchgeo.readthedocs.io/)
- [TerraTorch Documentation](https://ibm.github.io/terratorch/)
- [Prithvi EO V2 (Hugging Face)](https://huggingface.co/ibm-nasa-geospatial)
- [Landslides4Sense](https://www.iarai.ac.at/landslide4sense/)

## 작성자

오한 (ohhan@kari.re.kr)
한국항공우주연구원 위성활용연구팀
