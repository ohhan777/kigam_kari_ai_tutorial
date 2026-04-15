# plan_v3 — 사전학습 규모로의 전환 (DINOv3 ViT-L SAT-493M, RGB @224)

## 1. 목적
v2까지는 "스펙트럼 다양성(6~8ch HLS)"에 의존했다. v3에서는 가설을 뒤집어 **"사전학습 데이터의 규모·품질이 채널 수보다 지배적인가?"** 를 검증한다. DINOv3 ViT-L SAT-493M(4.93억 장 Maxar RGB 위성영상으로 사전학습)을 **RGB 3채널만** 사용해 v2와 비교한다.

## 2. 가설
대규모 자기지도 사전학습(DINO+iBOT+Gram Anchoring) 표현은 RGB 3채널만으로도 다중 스펙트럼 Prithvi 모델에 견줄 수 있을 것이다. 만약 성립하지 않는다면, "사전학습 규모" 단독으로는 부족하며 loss/augmentation 등 학습 레시피와의 결합이 필수라는 결론을 얻는다 (→ v4 설계 근거).

## 3. 이전 버전 대비 추가된 것
| 구분 | v2 | v3 |
|------|----|----|
| Backbone | Prithvi V2 300M (HLS 6ch 사전학습) | **DINOv3 ViT-L SAT-493M** (Maxar RGB 4.93억 장) |
| 입력 채널 | 8ch (HLS+slope+DEM) | **3ch (B4/B3/B2 Red-Green-Blue)** |
| Weight 확장 | Conv3d 6→8 (zero-init) | **확장 없음** — 사전학습 RGB patch embedding 그대로 |
| Decoder | UperNetDecoder (terratorch) | **Multi-scale FPN** (ViT 단일 stream → 4-depth hook) |

## 4. 구현 계획
- 파일: `train_v3.py`, `predict_v3.py`
- 체크포인트: HuggingFace `facebook/dinov3-vitl16-pretrain-sat493m` (auto download)
- RGB 선택: 14ch 입력에서 [B4=3, B3=2, B2=1] 인덱스 추출 후 bilinear 224 업스케일
- Decoder: transformer hidden_states에서 4개 depth hook (depth/4, depth/2, 3·depth/4, depth) → FPN top-down

## 5. 검증 포인트
- RGB 3ch만으로 v2(8ch)와 유의미한 차이가 나는지 — 초과 시 "사전학습 규모 > 채널 수" 확증, 미달 시 추가 레시피(v4) 필요성 확정
- Test F1에서의 generalization 격차 (v2보다 Valid-Test 격차가 커지는지/작아지는지)
- best epoch 위치 — DINOv3 대형 encoder의 조기 과적합 경향 관찰

## 6. 보고
`reports/result_v3.md`: Valid/Test F1 기록, "RGB vs 다중 스펙트럼" 비교, v2 대비 delta 및 원인 분석.
