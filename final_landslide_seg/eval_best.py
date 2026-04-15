"""Evaluate best.pth on valid/test for v2~v5."""
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from l4s.metrics import landslide_prf1, prf1_from_counts


@torch.no_grad()
def eval_loader(model, loader, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="exp dir with best.pth")
    ap.add_argument("--model", required=True, choices=["v2", "v3_or_4", "v5"])
    ap.add_argument("--data_dir", default="data/landslides4sense")
    ap.add_argument("--use_sepnorm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda")

    if args.model == "v2":
        from train_v2 import PrithviHiRes8ch
        model = PrithviHiRes8ch()
    else:  # v3, v4, v5 all use dinov3_vitl_sat
        from l4s.dinov3_model import dinov3_vitl_sat
        model = dinov3_vitl_sat()
    state = torch.load(Path(args.exp) / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    if args.use_sepnorm:
        from l4s.separated_norm_dataset import SepNormDataset as DS
        val_ds = DS(args.data_dir, "valid")
        test_ds = DS(args.data_dir, "test")
    else:
        from l4s.dataset_v2 import LandslideAugDataset as DS
        val_ds = DS(args.data_dir, "valid")
        test_ds = DS(args.data_dir, "test")
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    vp, vr, vf1 = eval_loader(model, val_loader, device)
    tp_, tr, tf1 = eval_loader(model, test_loader, device)
    print(f"[Valid] P={vp*100:.2f}% R={vr*100:.2f}% F1={vf1*100:.2f}%")
    print(f"[Test]  P={tp_*100:.2f}% R={tr*100:.2f}% F1={tf1*100:.2f}%")


if __name__ == "__main__":
    main()
