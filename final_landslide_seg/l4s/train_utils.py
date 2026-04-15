"""Shared training loop for all experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from l4s.dataset_v2 import LandslideAugDataset
from l4s.losses import DiceCELoss
from l4s.metrics import landslide_prf1, prf1_from_counts
from l4s.transforms import SegmentationAugmentation


def get_args(
    model_name: str,
    default_lr: float = 1e-4,
    default_epochs: int = 30,
    default_bs: int = 64,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train {model_name}")
    p.add_argument("--data_dir", type=str, default="data/landslides4sense")
    p.add_argument("--epochs", type=int, default=default_epochs)
    p.add_argument("--batch_size", type=int, default=default_bs)
    p.add_argument("--lr", type=float, default=default_lr)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default=f"exp_{model_name}")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--no_aug", action="store_true")
    return p.parse_args()


@torch.no_grad()
def validate(
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


def train_model(
    model_fn: Callable[[], nn.Module],
    model_name: str,
    default_lr: float = 1e-4,
    default_epochs: int = 30,
    default_bs: int = 64,
) -> None:
    args = get_args(model_name, default_lr, default_epochs, default_bs)
    data_root = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    aug = None if args.no_aug else SegmentationAugmentation()
    train_ds = LandslideAugDataset(data_root, "train", transform=aug)
    val_ds = LandslideAugDataset(data_root, "valid")
    test_ds = LandslideAugDataset(data_root, "test")

    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kw)

    model = model_fn().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DiceCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_ckpt = save_dir / "best.pth"
    last_ckpt = save_dir / "last.pth"
    history_csv = save_dir / "training_history.csv"
    best_f1 = -1.0
    history: list[tuple[int, float, float, float, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running / max(n_batches, 1)
        p, r, f1 = validate(model, val_loader, device)
        print(
            f"[{model_name}] Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
            f"P={p * 100:.2f}% R={r * 100:.2f}% F1={f1 * 100:.2f}%"
        )

        history.append((epoch, avg_loss, p, r, f1))
        with open(history_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "valid_precision", "valid_recall", "valid_f1"])
            for row in history:
                w.writerow(row)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_ckpt)

    torch.save(model.state_dict(), last_ckpt)

    # Final eval on best checkpoint
    state = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    val_p, val_r, val_f1 = validate(model, val_loader, device)
    test_p, test_r, test_f1 = validate(model, test_loader, device)
    print(f"\n=== {model_name} Final evaluation (best checkpoint) ===")
    print(f"[Valid]  Precision: {val_p * 100:.2f}%  Recall: {val_r * 100:.2f}%  F1: {val_f1 * 100:.2f}%")
    print(f"[Test]   Precision: {test_p * 100:.2f}%  Recall: {test_r * 100:.2f}%  F1: {test_f1 * 100:.2f}%")
