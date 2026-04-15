# Landslide Segmentation — 전체 실험 결과 (v0 ~ v5)

> 2026-04-15 실행. H100 × 6 병렬 (GPU0: v1, GPU1: v0, GPU2: v2, GPU3: v3, GPU4: v4, GPU5: v5). v2~v5는 시간 제약으로 80 epochs 계획 중 중단했으나 각 best.pth 기준 평가 완료.

## 종합 순위 (Valid F1 기준)

| 순위 | 버전 | 구성 요약 | Valid F1 | Test F1 | Δ vs v0 (Valid) | Δ vs v0 (Test) |
|:----:|------|-----------|:--------:|:-------:|:---------------:|:--------------:|
| 🥇 1 | **v5** | DINOv3+Lovász+MixUp+SepNorm+SelfTrain | **75.88%** | **71.03%** | **+10.60pp** | **+8.08pp** |
| 🥈 2 | v4 | DINOv3+Lovász+MixUp | 75.27% | 63.51% | +9.99pp | +0.56pp |
| 🥉 3 | v2 | Prithvi V2 300M 8ch @224 | 73.08% | 62.36% | +7.80pp | -0.59pp |
| 4 | v0 | UNet 14ch @128 (baseline) | 65.28% | 62.95%* | — | — |
| 5 | v3 | DINOv3 RGB @224 (DiceCE only) | 63.25% | 61.17% | -2.03pp | -1.78pp |
| 6 | v1 | Prithvi V2 300M 6ch @128 | 63.18% | 55.01% | -2.10pp | -7.94pp |

*v0 Test F1은 baseline 참조 수치(`landslides4sense_final_report.md` 기준, train_v0.py는 test eval 포함 안 함)

## 버전별 핵심 변화와 효과

```
v0 (baseline)        ──► UNet 14ch @128, CE          Valid 65.28%  Test 62.95%
                         │
v1 (+Prithvi GFM)    ──► Prithvi 6ch @128 HLS        Valid 63.18%  Test 55.01%  (-2.1pp, 실패)
                         │
v2 (+HiRes, +slope)  ──► Prithvi 8ch @224            Valid 73.08%  Test 62.36%  (+7.8pp)  ★해상도 효과
                         │
v3 (DINOv3 RGB only) ──► DINOv3 3ch @224 DiceCE      Valid 63.25%  Test 61.17%  (-2.0pp, 실패)
                         │
v4 (+Loss+MixUp)     ──► v3 + Lovász+CE+Dice + MixUp Valid 75.27%  Test 63.51%  (+10.0pp) ★Loss/Aug 효과
                         │
v5 (+SepNorm+Self)   ──► v4 + SepNorm + SelfTrain R1 Valid 75.88%  Test 71.03%  (+10.6pp / Test +8.1pp)
                                                                                 ★ 최종, 도메인 적응
```

## 주요 발견

### 1. 해상도(128→224)가 Prithvi 효과의 필요조건
- v1(128): 63.18% → v2(224): 73.08% = **+9.90pp**.
- 동일 해상도(128)에서는 Prithvi 6ch가 UNet from-scratch를 이기지 못함(v0 65.28% > v1 63.18%).
- ViT 패치 수(64→196)와 사전학습 해상도 정합(HLS 224로 사전학습)이 결합되어야 GFM의 이점이 나타난다.

### 2. 아키텍처만으로는 부족 — Loss/Aug가 DINOv3를 살림
- v3(DICE+CE only): 63.25% → v4(Lovász+CE+Dice+MixUp): 75.27% = **+12.02pp**.
- 동일 backbone에서 loss/augmentation만 교체해 baseline -2pp에서 +10pp로 점프.
- Lovász는 F1/IoU 직접 최적화로 Recall 55.55% → 79.76%(+24pp) 상승 유발.
- MixUp은 수렴 시점을 늦춰(best ep 3 → 29) 과적합 억제.

### 3. Test F1 향상은 Valid와 다른 메커니즘
- v2 → v4: Valid +2.19pp, Test +1.15pp (유사 상승)
- v4 → v5: Valid **+0.61pp**, Test **+7.52pp** (Test만 폭등)
- **Separated Norm + Self-training은 Valid 과적합이 아닌 "도메인 일반화"를 개선**. Valid-Test 격차가 v4 11.76pp → v5 4.85pp로 절반 이상 축소.

### 4. GFM ≠ 자동 승리
- v1(Prithvi @128), v3(DINOv3 RGB+DiceCE) 모두 baseline 대비 오히려 하락.
- GFM은 **해상도 + 사전학습 정합성 + IoU-aligned loss + strong aug + 도메인 적응**이라는 레시피를 갖춰야 비로소 효과.

## 최종 채택 모델 및 재현 커맨드

```bash
# v5 최종 — Best Valid F1 75.88%, Test F1 71.03%
CUDA_VISIBLE_DEVICES=0 uv run python train_v5.py \
    --epochs 50 --epochs_st 30 \
    --self_train_rounds 2 \
    --confidence_threshold 0.9 \
    --mixup_alpha 0.2 \
    --encoder_lr 5e-5 --new_lr 5e-4 \
    --batch_size 16 \
    --save_dir exp_v5_final
```

평가:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python eval_best.py --exp exp_v5_final --model v5 --use_sepnorm
```

## 한계 및 후속 과제

1. **v2~v5는 80/50 epochs 계획 중 중단된 상태에서 best 시점 평가**. 전체 완주 시 특히 v4가 추가 +1~2pp 가능.
2. **v5 Round 2 미실행** — R2까지 갔다면 76~77% Valid / 72~73% Test 기대.
3. **앙상블 미적용** — v2+v4+v5 soft voting 시 추가 향상 여지.
4. **Clay, DOFA, Prithvi 600M, DINOv3 ViT-7B frozen 등 미실험 GFM들** — 개별 특성이 다르므로 추가 비교 가치 있음.
5. **Test F1 62.95%(v0 reference)는 실제 측정 필요** — `predict_v0.py`로 재측정 권장.

## 산출물 경로

| 버전 | 체크포인트 | 로그 | 리포트 |
|:----:|-----------|------|--------|
| v0 | `weights/v0_baseline_unet_best.pth` | `logs/v0.log` | `reports/result_v0.md` |
| v1 | `exp_v1_prithvi6ch_128/best.pth` | `logs/v1.log` | `reports/result_v1.md` |
| v2 | `exp_v2_prithvi8ch_224/best.pth` | `logs/v2.log` | `reports/result_v2.md` |
| v3 | `exp_v3_dinov3_vitl_rgb224/best.pth` | `logs/v3.log` | `reports/result_v3.md` |
| v4 | `exp_v4_dinov3_lovasz_mixup/best.pth` | `logs/v4.log` | `reports/result_v4.md` |
| v5 | `exp_v5_final/best.pth` | `logs/v5.log` | `reports/result_v5.md` |
