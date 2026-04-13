"""Inference: visualize building segmentation predictions on Inria val images."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.trainers import SemanticSegmentationTask

if torch.cuda.is_available():
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")

# Checkpoint
ckpt_path = "runs/inria/version_1/checkpoints/last.ckpt"
NUM_SAMPLES = 10

# DataModule (training과 동일 설정)
datamodule = InriaAerialImageLabelingDataModule(
    root="./data/Inria",
    batch_size=NUM_SAMPLES,
    num_workers=8,
    patch_size=512,
)

# Predict for validation set
model = SemanticSegmentationTask.load_from_checkpoint(ckpt_path)
model.eval()
model.freeze()
model.cuda()
device = next(model.parameters()).device

datamodule.setup("fit")
val_dataset = datamodule.val_dataset
val_loader = datamodule.val_dataloader()

batch = next(iter(val_loader))
batch = datamodule.aug(batch)                             # CenterCrop(512) + Normalize
images = batch["image"].to(device)

with torch.no_grad():
    logits = model(images)                                # (B, 1, H, W)
    preds = (logits.sigmoid() > 0.5).squeeze(1).long()   # (B, H, W)

out_dir = Path(ckpt_path).parents[1] / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(NUM_SAMPLES, len(images))):
    sample = {key: batch[key][i] for key in batch}
    sample["prediction"] = preds[i].cpu()
    fig = val_dataset.plot(sample)
    fig.savefig(out_dir / f"pred_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"Saved predictions to {out_dir}/")
