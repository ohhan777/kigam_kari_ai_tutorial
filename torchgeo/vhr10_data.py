import kornia.augmentation as K
from torch.utils.data import DataLoader
from torchgeo.datamodules.utils import collate_fn_detection
from torchgeo.datasets import VHR10

from utils.plots import plot_detection_overlay


# kornia Resize 트랜스폼: 이미지/bbox/mask를 한 번에 동일 크기로 리사이즈.
# data_keys=None + keepdim=True 조합이면 sample dict 전체가 자동 처리됨.
transforms = K.AugmentationSequential(
    K.Resize((512, 512)),
    data_keys=None,
    keepdim=True,
)

dataset = VHR10(root='./data/', download=True, checksum=True, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_detection)

for batch in dataloader:
    images = batch['image']       # (B, 3, 512, 512) float
    labels = batch['label']       # list[(N_i,)]
    bboxes = batch['bbox_xyxy']   # list[(N_i, 4)]
    masks = batch['mask']         # list[(N_i, 512, 512)]

    print(f"Image batch shape: {images.shape}")
    print(f"First sample: {len(labels[0])} objects")

    # train a model, or make predictions using a pre-trained model
    # ...
    
    # Save overlay visualization of the first sample in the batch
    plot_detection_overlay(
        img=images[0],
        boxes=bboxes[0],
        labels=labels[0],
        masks=masks[0],
        save_file="sample_vhr10_overlay.png",
        class_names=VHR10.categories,
    )
    break
