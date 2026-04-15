"""Advanced GFM training with differential LR, warmup, and longer schedules.

Previous Prithvi+slope/DEM experiments failed because:
1. Same LR for pretrained encoder & zero-init new channels
2. Only 30 epochs - insufficient for new params to converge
3. No warmup - unstable early training

This module addresses all three issues.
"""

from __future__ import annotations

import argparse
import csv
import math
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


def _split_params(model: nn.Module) -> tuple[list, list]:
    """Split params into (pretrained_encoder, everything_else)."""
    encoder_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Pretrained encoder backbone (exclude patch_embed.proj which has new channels)
        if "encoder" in name and "patch_embed.proj" not in name and "aux" not in name and "gate" not in name:
            encoder_params.append(p)
        else:
            other_params.append(p)
    return encoder_params, other_params


def _warmup_cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    tp = fp = fn = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        pred = logits.argmax(dim=1)
        for b in range(pred.shape[0]):
            t, f_p, f_n = landslide_prf1(pred[b], labels[b])
            tp += t; fp += f_p; fn += f_n
    return prf1_from_counts(tp, fp, fn)


def train_gfm_advanced(
    model_fn: Callable[[], nn.Module],
    model_name: str,
    encoder_lr: float = 5e-5,
    new_lr: float = 5e-4,
    epochs: int = 80,
    warmup_epochs: int = 5,
    batch_size: int = 64,
    data_dir: str = "data/landslides4sense",
    num_workers: int = 4,
    device_str: str = "",
) -> None:
    # Parse CLI overrides
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--encoder_lr", type=float, default=encoder_lr)
    p.add_argument("--new_lr", type=float, default=new_lr)
    p.add_argument("--warmup_epochs", type=int, default=warmup_epochs)
    p.add_argument("--data_dir", type=str, default=data_dir)
    p.add_argument("--num_workers", type=int, default=num_workers)
    p.add_argument("--device", type=str, default=device_str)
    p.add_argument("--save_dir", type=str, default=f"exp_{model_name}")
    args = p.parse_args()

    data_root = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Data
    aug = SegmentationAugmentation()
    train_ds = LandslideAugDataset(data_root, "train", transform=aug)
    val_ds = LandslideAugDataset(data_root, "valid")
    test_ds = LandslideAugDataset(data_root, "test")
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kw)

    # Model
    model = model_fn().to(device)
    enc_params, new_params = _split_params(model)
    print(f"[{model_name}] encoder params: {sum(p.numel() for p in enc_params)/1e6:.1f}M, "
          f"new/decoder params: {sum(p.numel() for p in new_params)/1e6:.1f}M")

    optimizer = optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": new_params, "lr": args.new_lr},
    ], weight_decay=1e-4)

    criterion = DiceCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    best_ckpt = save_dir / "best.pth"
    last_ckpt = save_dir / "last.pth"
    history_csv = save_dir / "training_history.csv"
    best_f1 = -1.0
    history: list[tuple] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for images, labels in train_loader:
            # Warmup + cosine LR
            enc_lr = _warmup_cosine_lr(global_step, warmup_steps, total_steps, args.encoder_lr)
            new_lr_val = _warmup_cosine_lr(global_step, warmup_steps, total_steps, args.new_lr)
            optimizer.param_groups[0]["lr"] = enc_lr
            optimizer.param_groups[1]["lr"] = new_lr_val
            global_step += 1

            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        vp, vr, vf1 = _validate(model, val_loader, device)
        print(f"[{model_name}] Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
              f"P={vp*100:.2f}% R={vr*100:.2f}% F1={vf1*100:.2f}%  "
              f"enc_lr={enc_lr:.2e} new_lr={new_lr_val:.2e}")

        history.append((epoch, avg_loss, vp, vr, vf1))
        with open(history_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "valid_precision", "valid_recall", "valid_f1"])
            for row in history:
                w.writerow(row)

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), best_ckpt)

    torch.save(model.state_dict(), last_ckpt)

    # Final eval
    state = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    val_p, val_r, val_f1 = _validate(model, val_loader, device)
    test_p, test_r, test_f1 = _validate(model, test_loader, device)
    print(f"\n=== {model_name} Final evaluation (best checkpoint) ===")
    print(f"[Valid]  Precision: {val_p*100:.2f}%  Recall: {val_r*100:.2f}%  F1: {val_f1*100:.2f}%")
    print(f"[Test]   Precision: {test_p*100:.2f}%  Recall: {test_r*100:.2f}%  F1: {test_f1*100:.2f}%")
