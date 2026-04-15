"""Shared prediction / evaluation utilities for all experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from l4s.dataset import LandslideH5Dataset
from l4s.metrics import landslide_prf1, prf1_from_counts


@torch.no_grad()
def evaluate_split(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    tp = fp = fn = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        pred = logits.argmax(dim=1)
        for b in range(pred.shape[0]):
            t, f_p, f_n = landslide_prf1(pred[b], labels[b])
            tp += t
            fp += f_p
            fn += f_n
    return prf1_from_counts(tp, fp, fn)


def run_eval(model_fn: Callable[[], nn.Module], model_name: str) -> None:
    p = argparse.ArgumentParser(description=f"Evaluate {model_name}")
    p.add_argument("--data_dir", type=str, default="data/landslides4sense")
    p.add_argument("--checkpoint", type=str, default=f"exp_{model_name}/best.pth")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = model_fn()
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    root = Path(args.data_dir)
    val_ds = LandslideH5Dataset(root, "valid")
    test_ds = LandslideH5Dataset(root, "test")
    kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, **kw)
    test_loader = DataLoader(test_ds, **kw)

    val_p, val_r, val_f1 = evaluate_split(model, val_loader, device)
    test_p, test_r, test_f1 = evaluate_split(model, test_loader, device)
    print(f"\n=== {model_name} Final evaluation ===")
    print(f"[Valid]  Precision: {val_p * 100:.2f}%  Recall: {val_r * 100:.2f}%  F1: {val_f1 * 100:.2f}%")
    print(f"[Test]   Precision: {test_p * 100:.2f}%  Recall: {test_r * 100:.2f}%  F1: {test_f1 * 100:.2f}%")
