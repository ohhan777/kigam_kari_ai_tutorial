# plan_v4 — Loss/Augmentation 고도화 (Lovász + CE + Dice + MixUp)

## 1. 목적
v3에서 사용한 DINOv3 ViT-L SAT backbone은 유지하고, **학습 목표(loss)와 정규화 기법(augmentation)** 만 L4S 경진대회 상위권 팀이 사용한 방식으로 교체한다. 동일 backbone에서 loss/aug 레시피만 바뀌었을 때 성능이 어떻게 달라지는지 분리 측정한다.

## 2. 가설
1. Lovász-Softmax는 IoU의 convex extension이므로 F1 메트릭을 직접 최적화하여 경계 예측을 개선할 것이다.
2. MixUp(α=0.2, 50% 확률)은 3,799개 소규모 데이터에서 과적합을 억제하고 confidence calibration을 개선하여, 이후(v5)의 pseudo label 품질에 기여할 것이다.
3. 복합 손실 **0.4·CE + 0.3·Lovász + 0.3·Dice** 가 단일 손실보다 안정적일 것이다.

## 3. 이전 버전 대비 추가된 것
| 구분 | v3 | v4 |
|------|----|----|
| Loss | CE + Dice (0.5/0.5) | **0.4·CE + 0.3·Lovász-Softmax + 0.3·Dice** |
| Augmentation | flip/rot90 | flip/rot90 + **MixUp(α=0.2, p=0.5)** |

## 4. 구현 계획
- 파일: `train_v4.py`, `predict_v4.py`
- 새 모듈: Lovász-Softmax 및 `CompetitionLoss`(0.4 CE + 0.3 Lovász + 0.3 Dice)
- MixUp: Beta(α, α)에서 혼합 비율 샘플링, 50% 확률로 batch 내 순열 혼합, loss는 두 라벨에 대해 선형 결합

## 5. 검증 포인트
- v3 대비 Valid F1 변화 방향성 — loss/aug 단독으로 backbone을 "살릴 수 있는지"
- CE+Dice 대비 Lovász 추가 시 Precision/Recall 균형 변화 (특히 경계 recall)
- MixUp 적용 시 epoch-F1 커브가 매끄러워지고 best epoch이 뒤로 밀리는지 (과적합 억제 신호)
- Test F1 변화 — Valid만 오르는 과적합인지, 일반화도 함께 개선되는지

## 6. 보고
`reports/result_v4.md`: v3 대비 delta, best epoch 위치, P/R 균형 변화, Valid/Test 격차 기록.
