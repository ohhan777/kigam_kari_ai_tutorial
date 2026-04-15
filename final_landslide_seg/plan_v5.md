# plan_v5 (최종) — Separated Normalization + Self-Training (2-Round Pseudo Label)

## 1. 목적
v4까지의 레시피(DINOv3 ViT-L SAT @224 + Differential LR + Lovász+CE+Dice + MixUp)에 **도메인 적응 기법 두 가지**를 추가하여, L4S 데이터셋의 본질적 한계인 **Train vs Valid/Test 지역 간 분포 격차**를 직접 공략한다. 두 축:

1. **Separated Normalization** — 도메인(split)별 평균/표준편차를 분리 적용하여 입력 분포를 정렬
2. **Self-Training (2 rounds)** — 학습된 모델로 Valid/Test에 pseudo label을 생성 후 재학습(도메인 적응)

## 2. 가설
- SepNorm 단독 효과는 작을 수 있으나, pseudo label 품질을 크게 높이는 전제조건이 될 것이다.
- Self-training round가 진행될수록 Valid F1은 소폭, Test F1은 폭넓게 상승할 것이다 (도메인 일반화 효과).
- Valid-Test 격차가 v4 대비 유의미하게 축소되는 것이 이 전략의 최종 성공 지표가 된다.

## 3. 이전 버전 대비 추가된 것
| 구분 | v4 | v5 (최종) |
|------|----|-----------|
| Normalization | 전체 Train 통계 | **split별 mean/std 분리 적용** (train/valid/test 각자 표준화) |
| 학습 스테이지 | 단일 스테이지 | **3스테이지**: Round 0 (labeled only) → Round 1 (+pseudo) → Round 2 (정제된 pseudo) |
| Pseudo label | — | **confidence ≥ 0.9 픽셀만 학습에 사용** |
| Combined dataset | Train 3,799 | **Train 3,799 + Valid 245(pseudo) + Test 800(pseudo)** |
| Augmentation | MixUp(p=0.5) | **Round 1,2에서는 MixUp 해제** (pseudo label이 이미 soft) |

## 4. 구현 계획
- 파일: `train_v5.py`, `predict_v5.py`
- 데이터셋: split별 mean/std 테이블을 내장한 `SepNormDataset`, confidence mask 기반 `SepNormPseudoDataset`
- 스크립트 내부에서 Round 0~2를 순차 수행, 각 Round best checkpoint을 다음 Round 초기화로 사용
- Round별 weight: `best_round0.pth`, `best_round1.pth`, `best_round2.pth`, 전체 최고는 `best.pth`

## 5. 검증 포인트
- SepNorm 적용 전/후 입력 분포 정합성 (histogram 비교)
- Round별 pseudo label 평균 confidence 및 threshold(0.9) 통과 픽셀 비율
- **Valid-Test 격차 축소 여부** — v4의 약 10pp 격차가 v5에서 어느 수준까지 줄어드는지
- 각 Round의 best epoch이 점차 앞당겨지는지 (초기 모델 품질 상승 신호)

## 6. 보고
- `reports/result_v5.md`: Round별 표, Valid/Test F1, v0~v4 대비 누적 delta
- `reports/overall_result.md`: 전체 버전 요약 및 **최종 채택 레시피** 기록
- 최종 레시피 요약 (구성안):
  - **Backbone**: DINOv3 ViT-L SAT-493M (RGB 3ch @224)
  - **Decoder**: Multi-scale FPN (4-depth hook)
  - **Optimizer**: Differential LR (encoder 5e-5 / decoder 5e-4), Warmup(3) + Cosine
  - **Loss**: 0.4·CE + 0.3·Lovász-Softmax + 0.3·Dice
  - **Augmentation**: flip/rot90 + MixUp(α=0.2, Round 0에서만)
  - **Normalization**: Separated Norm (split별 mean/std)
  - **Self-Training**: 2 rounds, confidence ≥ 0.9

---

## 부록: 다중 GPU 병렬 실행 구상
각 plan 하나당 GPU 1장을 할당해 동시 실행할 수 있다:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python train_v1.py &
CUDA_VISIBLE_DEVICES=1 uv run python train_v2.py &
CUDA_VISIBLE_DEVICES=2 uv run python train_v3.py &
CUDA_VISIBLE_DEVICES=3 uv run python train_v4.py &
CUDA_VISIBLE_DEVICES=4 uv run python train_v5.py &
wait
```
남은 GPU는 ablation(예: SepNorm off, MixUp off) 또는 seed 교차 검증에 할당 가능.
