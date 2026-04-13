# Semantic segmentation: UNet on Inria Aerial Image Labeling (binary buildings).
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from torchgeo.datamodules import InriaAerialImageLabelingDataModule
from torchgeo.trainers import SemanticSegmentationTask

# Ampere+: TF32 for conv (new API). Matmul: set_float32_matmul_precision — Lightning still calls
# get_float32_matmul_precision(); mixing cuda.matmul.fp32_precision with that raises RuntimeError.
if torch.cuda.is_available():
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")

pl.seed_everything(0)

datamodule = InriaAerialImageLabelingDataModule(
    root="./data/Inria",
    batch_size=32,
    num_workers=8,
    patch_size=512,
)

task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet34",
    weights=True,           # ImageNet pretrained backbone
    in_channels=3,
    task="binary",
    loss="bce",
    lr=1e-3,
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    mode="min",
    monitor="val_loss",
    filename="best-{epoch:02d}-{val_loss:.4f}",
    save_last=True,
)

trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices=1,
    precision="bf16-mixed",
    logger=TensorBoardLogger(save_dir="runs", name="inria"),
    max_epochs=50,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
    default_root_dir="runs/inria",
)

trainer.fit(model=task, datamodule=datamodule)
