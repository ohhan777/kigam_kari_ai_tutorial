import torch
import terratorch
import albumentations
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

# Ampere+: TF32 for conv (new API). Matmul: set_float32_matmul_precision — Lightning still calls
# get_float32_matmul_precision(); mixing cuda.matmul.fp32_precision with that raises RuntimeError.
if torch.cuda.is_available():
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")

pl.seed_everything(0)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    # dirpath omitted → saves under logger's version_N/checkpoints/
    mode="max",
    monitor="val/mIoU",
    filename="best-{epoch:02d}-{val/mIoU:.4f}",
    save_last=True,
)


dataset_path = Path('data/hls_burn_scars')
DATA_ROOT = dataset_path / 'data/'  # ./data/hls_burn_scars/data/
SPLITS_ROOT = dataset_path / 'splits/'  # ./data/hls_burn_scars/splits/

datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
    batch_size=4,
    num_workers=2,
    num_classes=2,

    # Dataset paths
    train_data_root=DATA_ROOT,
    train_label_data_root=DATA_ROOT,
    val_data_root=DATA_ROOT,
    val_label_data_root=DATA_ROOT,
    test_data_root=DATA_ROOT,
    test_label_data_root=DATA_ROOT,

    # Splits
    train_split=SPLITS_ROOT / 'train.txt',
    val_split=SPLITS_ROOT / 'val.txt',
    test_split=SPLITS_ROOT / 'test.txt',
    
    img_grep='*_merged.tif',
    label_grep='*.mask.tif',
    
    train_transform=[
        albumentations.D4(), # Random flips and rotation
        albumentations.pytorch.transforms.ToTensorV2(),
    ],
    val_transform=None,
    test_transform=None,

    # Standardization values
    means=[
        0.0333497067415863,
        0.0570118552053618,
        0.0588974813200132,
        0.2323245113436119,
        0.1972854853760658,
        0.1194491422518656,
    ],
    stds=[
        0.0226913556882377,
        0.0268075602230702,
        0.0400410984436278,
        0.0779173242367269,
        0.0870873883814014,
        0.0724197947743781,
    ],
    no_data_replace=0,
    no_label_replace=-1,
    # We use all six bands of the data, so we don't need to define dataset_bands and output_bands.
)

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices=1, # Deactivate multi-gpu because it often fails in notebooks
    precision='bf16-mixed',  # Speed up training
    num_nodes=1,
    logger=TensorBoardLogger(save_dir="runs", name="burnscars"),
    max_epochs=50, # For demos
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
    default_root_dir="runs/burnscars",
)

# Model
model = terratorch.tasks.SemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args={
        # Backbone
        "backbone": "prithvi_eo_v2_300", # Model can be either prithvi_eo_v1_100, prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
        "backbone_pretrained": True,
        "backbone_num_frames": 1, # 1 is the default value,
        "backbone_img_size": 512,
        "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        # "backbone_coords_encoding": [], # use ["time", "location"] for time and location metadata
        
        # Necks 
        "necks": [
            {
                "name": "SelectIndices",
                # "indices": [2, 5, 8, 11] # indices for prithvi_eo_v1_100
                "indices": [5, 11, 17, 23] # indices for prithvi_eo_v2_300
                # "indices": [7, 15, 23, 31] # indices for prithvi_eo_v2_600
            },
            {"name": "ReshapeTokensToImage",},
            {"name": "LearnedInterpolateToPyramidal"}            
        ],
        
        # Decoder
        "decoder": "UNetDecoder",
        "decoder_channels": [512, 256, 128, 64],
        
        # Head
        "head_dropout": 0.1,
        "num_classes": 2,
    },
    
    loss="ce",
    optimizer="AdamW",
    lr=1e-4,
    ignore_index=-1,
    freeze_backbone=True, # Only to speed up fine-tuning
    freeze_decoder=False,
    plot_on_val=True,
    class_names=['no burned', 'burned']  # optionally define class names
)


# Training
trainer.fit(model, datamodule=datamodule)