"""산사태 클래스(라벨 1) 기준 픽셀 단위 Precision / Recall / F1."""

from __future__ import annotations

import torch


def landslide_prf1(pred: torch.Tensor, label: torch.Tensor, eps: float = 1e-14) -> tuple[int, int, int]:
    """pred, label 동형 long 텐서. (tp, fp, fn) 픽셀 개수."""
    p1 = pred == 1
    l1 = label == 1
    tp = int((p1 & l1).sum().item())
    fp = int((p1 & ~l1).sum().item())
    fn = int((~p1 & l1).sum().item())
    return tp, fp, fn


def prf1_from_counts(tp: int, fp: int, fn: int, eps: float = 1e-14) -> tuple[float, float, float]:
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2.0 * p * r / (p + r + eps)
    return float(p), float(r), float(f1)
