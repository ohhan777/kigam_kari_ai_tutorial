"""Custom loss functions for landslide segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for the landslide (positive) class."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        union = probs.sum() + targets_f.sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss."""

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovász extension w.r.t sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    """Lovász-Softmax loss for binary segmentation (L4S competition 1st/2nd place)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        fg_prob = probs[:, 1]  # landslide probability
        losses = []
        for prob, label in zip(fg_prob, targets):
            prob_flat = prob.reshape(-1)
            label_flat = label.reshape(-1)
            errors = (label_flat - prob_flat).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            perm = perm.detach()
            gt_sorted = label_flat[perm]
            grad = _lovasz_grad(gt_sorted)
            losses.append(torch.dot(F.relu(errors_sorted), grad))
        return torch.stack(losses).mean()


class CompetitionLoss(nn.Module):
    """Soft CE + Lovász + Dice (inspired by L4S 1st/2nd place teams)."""

    def __init__(self, ce_weight: float = 0.4, lovasz_weight: float = 0.3, dice_weight: float = 0.3):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.lovasz = LovaszSoftmax()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight * self.ce(logits, targets)
                + self.lovasz_weight * self.lovasz(logits, targets)
                + self.dice_weight * self.dice(logits, targets))
