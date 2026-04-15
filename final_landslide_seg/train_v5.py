"""v5 (최종) — DINOv3 ViT-L SAT @224 + CompetitionLoss + MixUp + SepNorm + Self-training.

plan_v5: v4 + 도메인 적응 2기법.
  (1) Separated Normalization — split별 mean/std 분리 정규화
  (2) Self-Training 2 rounds — pseudo label (confidence ≥ 0.9)

kigam_tutorial/train_dinov3_vitl_advanced.py 의 로직을 그대로 재사용한다.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader

from l4s.dinov3_model import dinov3_vitl_sat
from l4s.losses import CompetitionLoss
from l4s.metrics import landslide_prf1, prf1_from_counts
from l4s.separated_norm_dataset import SepNormDataset, SepNormPseudoDataset
from l4s.transforms import SegmentationAugmentation


def _split_params(model):
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


@torch.no_grad()
def _generate_pseudo(model, data_root, split, device, batch_size=32):
    ds = SepNormDataset(data_root, split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    all_probs, all_preds = [], []
    for images, _ in loader:
        images = images.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_probs.append(conf.cpu().numpy())
        all_preds.append(pred.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    pseudo = {}
    for i, (img_path, _) in enumerate(ds.samples):
        m = re.search(r"image_(\d+)\.h5$", img_path.name)
        if m:
            pseudo[m.group(1)] = (all_probs[i], all_preds[i])
    pos = sum((p[1] > 0).sum() for p in pseudo.values())
    tot = sum(p[1].size for p in pseudo.values())
    avg = np.mean([p[0].mean() for p in pseudo.values()])
    print(f"  pseudo[{split}]: N={len(pseudo)}, landslide={pos/tot*100:.1f}%, conf={avg:.3f}")
    return pseudo


def mixup_data(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epochs_st", type=int, default=30, help="epochs for self-training rounds")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--encoder_lr", type=float, default=5e-5)
    p.add_argument("--new_lr", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--self_train_rounds", type=int, default=2)
    p.add_argument("--confidence_threshold", type=float, default=0.9)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--data_dir", type=str, default="data/landslides4sense")
    p.add_argument("--save_dir", type=str, default="exp_v5_final")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    data_root = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    aug = SegmentationAugmentation()
    val_ds = SepNormDataset(data_root, "valid")
    test_ds = SepNormDataset(data_root, "test")
    kw = dict(num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kw)

    criterion = CompetitionLoss()
    history = []
    best_f1_overall = -1.0

    for st_round in range(args.self_train_rounds + 1):
        print(f"\n{'='*60}\nSelf-training round {st_round}/{args.self_train_rounds}\n{'='*60}")
        model = dinov3_vitl_sat().to(device)

        prev = save_dir / f"best_round{st_round-1}.pth"
        if st_round > 0 and prev.exists():
            model.load_state_dict(torch.load(prev, map_location=device, weights_only=True))
            print(f"Loaded checkpoint from round {st_round-1}")

        if st_round > 0:
            print("Generating pseudo labels...")
            pseudo_val = _generate_pseudo(model, data_root, "valid", device)
            pseudo_test = _generate_pseudo(model, data_root, "test", device)
            train_labeled = SepNormDataset(data_root, "train", transform=aug)
            pseudo_val_ds = SepNormPseudoDataset(
                data_root, pseudo_val, "valid", transform=aug,
                confidence_threshold=args.confidence_threshold)
            pseudo_test_ds = SepNormPseudoDataset(
                data_root, pseudo_test, "test", transform=aug,
                confidence_threshold=args.confidence_threshold)

            class _Wrap(torch.utils.data.Dataset):
                def __init__(self, ds):
                    self.ds = ds
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, i):
                    img, m = self.ds[i]
                    return img, m, torch.ones_like(m, dtype=torch.float32)

            combined = ConcatDataset([_Wrap(train_labeled), pseudo_val_ds, pseudo_test_ds])
            train_loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, drop_last=False, **kw)
            print(f"Combined: {len(combined)} (train={len(train_labeled)} + val_pseudo={len(pseudo_val_ds)} + test_pseudo={len(pseudo_test_ds)})")
        else:
            train_ds = SepNormDataset(data_root, "train", transform=aug)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **kw)
            print(f"Labeled-only: {len(train_ds)}")

        enc_params, new_params = _split_params(model)
        optimizer = optim.AdamW([
            {"params": enc_params, "lr": args.encoder_lr},
            {"params": new_params, "lr": args.new_lr},
        ], weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        epochs = args.epochs if st_round == 0 else args.epochs_st
        total_steps = epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        best_f1 = -1.0
        global_step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            n = 0
            for batch in train_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                enc_lr = _warmup_cosine_lr(global_step, warmup_steps, total_steps, args.encoder_lr)
                new_lr_val = _warmup_cosine_lr(global_step, warmup_steps, total_steps, args.new_lr)
                optimizer.param_groups[0]["lr"] = enc_lr
                optimizer.param_groups[1]["lr"] = new_lr_val
                global_step += 1

                images, labels = images.to(device), labels.to(device).long()
                optimizer.zero_grad(set_to_none=True)
                if st_round == 0 and args.mixup_alpha > 0 and np.random.random() < 0.5:
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
            print(f"[R{st_round}] Epoch {epoch}/{epochs} loss={avg_loss:.4f} "
                  f"P={vp*100:.2f}% R={vr*100:.2f}% F1={vf1*100:.2f}%")
            history.append((f"{st_round}-{epoch}", avg_loss, vp, vr, vf1))
            with open(save_dir / "training_history.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "valid_precision", "valid_recall", "valid_f1"])
                for row in history:
                    w.writerow(row)

            if vf1 > best_f1:
                best_f1 = vf1
                torch.save(model.state_dict(), save_dir / f"best_round{st_round}.pth")
                if vf1 > best_f1_overall:
                    best_f1_overall = vf1
                    torch.save(model.state_dict(), save_dir / "best.pth")

        torch.save(model.state_dict(), save_dir / f"last_round{st_round}.pth")
        print(f"[Round {st_round}] best Valid F1: {best_f1*100:.2f}%")

    print(f"\n{'='*60}\nFinal eval (best overall)\n{'='*60}")
    state = torch.load(save_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    val_p, val_r, val_f1 = _validate(model, val_loader, device)
    test_p, test_r, test_f1 = _validate(model, test_loader, device)
    print(f"[Valid] P={val_p*100:.2f}% R={val_r*100:.2f}% F1={val_f1*100:.2f}%")
    print(f"[Test]  P={test_p*100:.2f}% R={test_r*100:.2f}% F1={test_f1*100:.2f}%")


if __name__ == "__main__":
    main()
