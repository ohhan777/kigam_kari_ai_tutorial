import os
import torch
import numpy as np
import h5py
import re
from pathlib import Path
import albumentations as A

# Normalization constants from Landslide4Sense baseline
MEAN = np.array(
    [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802,
     0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819], dtype=np.float32)
STD = np.array(
    [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491,
     0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913], dtype=np.float32)


class Landslides4SenseDataset(torch.utils.data.Dataset):
    _SPLIT_DIR = {'train': 'TrainData', 'val': 'ValidData', 'test': 'TestData'}

    def __init__(self, root, split='train'):
        if split not in self._SPLIT_DIR:
            raise ValueError(f"split must be one of {list(self._SPLIT_DIR)}, got '{split}'")
        self.root = Path(root)
        self.split = split
        self.train = (split == 'train')
        sub = self._SPLIT_DIR[split]
        self.img_dir = self.root / sub / 'img'
        self.mask_dir = self.root / sub / 'mask'

        # collect image-mask pairs
        self.samples = []
        for p in sorted(self.img_dir.glob('image_*.h5')):
            m = re.search(r'image_(\d+)\.h5$', p.name)
            if not m:
                continue
            mid = m.group(1)
            mp = self.mask_dir / f'mask_{mid}.h5'
            if mp.is_file():
                self.samples.append((p, mp))

        if len(self.samples) == 0:
            raise FileNotFoundError('No image-mask pairs found in %s' % self.img_dir)

        # preload all data into memory to avoid per-sample GPFS I/O
        self.imgs = []
        self.masks = []
        for img_file, mask_file in self.samples:
            with h5py.File(img_file, 'r') as f:
                self.imgs.append(np.array(f['img'][:], dtype=np.float32))
            with h5py.File(mask_file, 'r') as f:
                self.masks.append(np.clip(np.array(f['mask'][:], dtype=np.int64), 0, 1))

        self.transforms = get_transforms(self.train)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img, mask = self.transforms(img, mask)
        return img, mask, str(self.samples[idx][0])

    def __len__(self):
        return len(self.samples)


class ImageAug:
    def __init__(self, train):
        if train:
            self.aug = A.Compose([A.HorizontalFlip(p=0.5),
                                  A.VerticalFlip(p=0.5),
                                  A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625),
                                           rotate=(-45, 45), p=0.5)])
        else:
            self.aug = None

    def __call__(self, img, mask):
        # img: (H, W, C) numpy float32, mask: (H, W) numpy int64
        if self.aug:
            transformed = self.aug(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']
        # normalize: (H, W, C) -> (C, H, W), then channel-wise MEAN/STD
        img = img.transpose(2, 0, 1)  # (C, H, W)
        for i in range(len(MEAN)):
            img[i] = (img[i] - MEAN[i]) / STD[i]
        return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy()).long()


def get_transforms(train):
    transforms = ImageAug(train)
    return transforms
