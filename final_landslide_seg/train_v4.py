"""v4 — DINOv3 ViT-L SAT @224 + CompetitionLoss (CE+Lovász+Dice) + MixUp.

plan_v4: v3 백본 + L4S 경진대회 상위권 기법(복합 손실 + MixUp).
self-training 없이 라벨링 데이터만 사용. v3 대비 Loss/Aug 고도화 효과만 측정.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from l4s.dinov3_model import dinov3_vitl_sat
from l4s.dataset_v2 import LandslideAugDataset
from l4s.losses import CompetitionLoss
from l4s.metrics import landslide_prf1, prf1_from_counts
from l4s.transforms import SegmentationAugmentation


def _split_params(model: nn.Module):
    enc, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if "encoder" in name else other).append(p)
    return enc, other


def _warmup_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def _validate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images)
        pred = logits.argmax(dim=1)
        for b in range(pred.shape[0]):
            t, f_p, f_n = landslide_prf1(pred[b], labels[b])
            tp += t; fp += f_p; fn += f_n
    return prf1_from_counts(tp, fp, fn)


def mixup_data(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--encoder_lr", type=float, default=5e-5)
    p.add_argument("--new_lr", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--mixup_prob", type=float, default=0.5)
    p.add_argument("--data_dir", type=str, default="data/landslides4sense")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="exp_v4_dinov3_lovasz_mixup")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    data_root = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    aug = SegmentationAugmentation()
    train_ds = LandslideAugDataset(data_root, "train", transform=aug)
    val_ds = LandslideAugDataset(data_root, "valid")
    test_ds = LandslideAugDataset(data_root, "test")
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kw)

    model = dinov3_vitl_sat().to(device)
    enc_params, new_params = _split_params(model)
    print(f"[v4] encoder: {sum(p.numel() for p in enc_params)/1e6:.1f}M, "
          f"new/decoder: {sum(p.numel() for p in new_params)/1e6:.1f}M")

    optimizer = optim.AdamW([
        {"params": enc_params, "lr": args.encoder_lr},
        {"params": new_params, "lr": args.new_lr},
    ], weight_decay=1e-4)
    criterion = CompetitionLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    best_f1 = -1.0
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for images, labels in train_loader:
            enc_lr = _warmup_cosine_lr(global_step, warmup_steps, total_steps, args.encoder_lr)
            new_lr_val = _warmup_cosine_lr(global_step, warmup_steps, total_steps, args.new_lr)
            optimizer.param_groups[0]["lr"] = enc_lr
            optimizer.param_groups[1]["lr"] = new_lr_val
            global_step += 1

            images, labels = images.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            if args.mixup_alpha > 0 and np.random.random() < args.mixup_prob:
                mixed, la, lb, lam = mixup_data(images, labels, args.mixup_alpha)
                with torch.amp.autocast("cuda"):
                    logits = model(mixed)
                    loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            else:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n += 1

        avg_loss = running / max(n, 1)
        vp, vr, vf1 = _validate(model, val_loader, device)
        print(f"[v4] Epoch {epoch}/{args.epochs} loss={avg_loss:.4f} "
              f"P={vp*100:.2f}% R={vr*100:.2f}% F1={vf1*100:.2f}% "
              f"enc_lr={enc_lr:.2e} new_lr={new_lr_val:.2e}")

        history.append((epoch, avg_loss, vp, vr, vf1))
        with open(save_dir / "training_history.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "valid_precision", "valid_recall", "valid_f1"])
            for row in history:
                w.writerow(row)

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), save_dir / "best.pth")
    torch.save(model.state_dict(), save_dir / "last.pth")

    state = torch.load(save_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    val_p, val_r, val_f1 = _validate(model, val_loader, device)
    test_p, test_r, test_f1 = _validate(model, test_loader, device)
    print("\n=== v4 Final ===")
    print(f"[Valid]  P={val_p*100:.2f}% R={val_r*100:.2f}% F1={val_f1*100:.2f}%")
    print(f"[Test]   P={test_p*100:.2f}% R={test_r*100:.2f}% F1={test_f1*100:.2f}%")


if __name__ == "__main__":
    main()
