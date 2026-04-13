import torch
import terratorch
import lightning.pytorch as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

if torch.cuda.is_available():
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")

# Checkpoint
ckpt_path = "runs/burnscars/version_0/checkpoints/best-epoch=33-val/mIoU=0.8823.ckpt"

# DataModule (test split only — train/val paths still required by the constructor)
dataset_path = Path('data/hls_burn_scars')
DATA_ROOT = dataset_path / 'data/'
SPLITS_ROOT = dataset_path / 'splits/'

datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
    batch_size=4,
    num_workers=2,
    num_classes=2,
    train_data_root=DATA_ROOT,
    train_label_data_root=DATA_ROOT,
    val_data_root=DATA_ROOT,
    val_label_data_root=DATA_ROOT,
    test_data_root=DATA_ROOT,
    test_label_data_root=DATA_ROOT,
    train_split=SPLITS_ROOT / 'train.txt',
    val_split=SPLITS_ROOT / 'val.txt',
    test_split=SPLITS_ROOT / 'test.txt',
    img_grep='*_merged.tif',
    label_grep='*.mask.tif',
    means=[
        0.0333497067415863, 0.0570118552053618, 0.0588974813200132,
        0.2323245113436119, 0.1972854853760658, 0.1194491422518656,
    ],
    stds=[
        0.0226913556882377, 0.0268075602230702, 0.0400410984436278,
        0.0779173242367269, 0.0870873883814014, 0.0724197947743781,
    ],
    no_data_replace=0,
    no_label_replace=-1,
)

# Validation and test metrics via Trainer
model = terratorch.tasks.SemanticSegmentationTask.load_from_checkpoint(ckpt_path)
trainer = pl.Trainer(accelerator="auto", devices=1, precision='bf16-mixed')
trainer.validate(model, datamodule=datamodule) # Validation metrics
trainer.test(model, datamodule=datamodule) # Test metrics

# Predict & plot
model.eval()
model.freeze()
device = next(model.parameters()).device

datamodule.setup("fit")
val_dataset = datamodule.val_dataset
val_loader = datamodule.val_dataloader()

batch = next(iter(val_loader))
images = batch["image"].to(device)

with torch.no_grad():
    outputs = model(images)
    preds = outputs.output.argmax(dim=1)

out_dir = Path(ckpt_path).parents[2] / "predictions"
out_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(4, len(images))):
    sample = {key: batch[key][i] for key in batch}
    sample["prediction"] = preds[i].cpu()
    fig = val_dataset.plot(sample)
    fig.savefig(out_dir / f"pred_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"Saved predictions to {out_dir}/")
