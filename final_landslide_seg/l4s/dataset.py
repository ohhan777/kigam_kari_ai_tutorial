"""HDF5 패치 로더 — iarai/Landslide4Sense-2022 dataset/landslide_dataset.py 정규화 상수 동일."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 베이스라인 LandslideDataSet 과 동일 (landslide_dataset.py __main__ 주석값)
MEAN = np.array(
    [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819],
    dtype=np.float32,
)
STD = np.array(
    [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913],
    dtype=np.float32,
)


class LandslideH5Dataset(Dataset):
    """TrainData / ValidData / TestData 의 image_*.h5 + mask_*.h5 쌍."""

    def __init__(self, data_root: Path | str, split: str):
        self.data_root = Path(data_root)
        if split == "train":
            sub = "TrainData"
        elif split in ("valid", "val"):
            sub = "ValidData"
        elif split == "test":
            sub = "TestData"
        else:
            raise ValueError(f"split must be train/valid/test, got {split}")

        img_dir = self.data_root / sub / "img"
        mask_dir = self.data_root / sub / "mask"
        self.samples: list[tuple[Path, Path]] = []
        for p in sorted(img_dir.glob("image_*.h5")):
            m = re.search(r"image_(\d+)\.h5$", p.name)
            if not m:
                continue
            mid = m.group(1)
            mp = mask_dir / f"mask_{mid}.h5"
            if mp.is_file():
                self.samples.append((p, mp))

        if not self.samples:
            raise FileNotFoundError(f"No labeled pairs under {img_dir} + {mask_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.samples[index]
        with h5py.File(ip, "r") as hf:
            image = np.asarray(hf["img"][:], dtype=np.float32)
        image = image.transpose(2, 0, 1)
        for i in range(MEAN.shape[0]):
            image[i] = (image[i] - MEAN[i]) / STD[i]

        with h5py.File(mp, "r") as hf:
            label = np.asarray(hf["mask"][:], dtype=np.int64)
        label = np.clip(label, 0, 1)

        return torch.from_numpy(image.copy()), torch.from_numpy(label)
