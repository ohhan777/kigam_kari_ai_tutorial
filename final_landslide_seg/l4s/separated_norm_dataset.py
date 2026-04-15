"""Dataset with separated normalization (per-domain mean/std).

Key insight from L4S competition 1st place: train and val/test have
very different distributions. Using domain-specific normalization
significantly improves generalization.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Per-domain statistics (calculated from full splits)
STATS = {
    "train": {
        "mean": np.array([0.5392, 0.5410, 0.5475, 0.5861, 0.6037, 0.5507,
                          0.5567, 0.5653, 0.6204, 0.7001, 0.6298, 0.7186,
                          0.8362, 1.1303], dtype=np.float32),
        "std":  np.array([0.3599, 0.3789, 0.4442, 0.6404, 0.5536, 0.5647,
                          0.5671, 0.5934, 0.6592, 0.7811, 0.6032, 0.6726,
                          0.7007, 1.0856], dtype=np.float32),
    },
    "valid": {
        "mean": np.array([0.9960, 1.0250, 1.0942, 1.2124, 1.2975, 1.0928,
                          1.0215, 1.0397, 1.5847, 2.3120, 1.2230, 1.3171,
                          1.2336, 1.5350], dtype=np.float32),
        "std":  np.array([0.1166, 0.1705, 0.2483, 0.4994, 0.3968, 0.3022,
                          0.2941, 0.3207, 0.5517, 3.3814, 0.4841, 0.7387,
                          0.4897, 0.7543], dtype=np.float32),
    },
    "test": {
        "mean": np.array([0.9958, 1.0168, 1.0675, 1.1987, 1.2911, 1.0809,
                          1.0172, 1.0348, 1.6393, 2.4263, 1.2366, 1.3421,
                          1.2364, 1.5655], dtype=np.float32),
        "std":  np.array([0.1156, 0.1705, 0.2504, 0.5094, 0.4034, 0.3026,
                          0.2922, 0.3190, 0.5509, 3.0335, 0.4930, 0.7631,
                          0.4849, 0.7607], dtype=np.float32),
    },
}


class SepNormDataset(Dataset):
    """Dataset with separated (per-domain) normalization."""

    def __init__(self, data_root: Path | str, split: str, transform=None):
        self.data_root = Path(data_root)
        sub = {"train": "TrainData", "valid": "ValidData", "val": "ValidData", "test": "TestData"}[split]
        norm_key = {"val": "valid"}.get(split, split)

        self.mean = STATS[norm_key]["mean"]
        self.std = STATS[norm_key]["std"]
        self.transform = transform

        img_dir = self.data_root / sub / "img"
        mask_dir = self.data_root / sub / "mask"
        self.samples: list[tuple[Path, Path | None]] = []
        for p in sorted(img_dir.glob("image_*.h5")):
            m = re.search(r"image_(\d+)\.h5$", p.name)
            if not m:
                continue
            mid = m.group(1)
            mp = mask_dir / f"mask_{mid}.h5"
            self.samples.append((p, mp if mp.is_file() else None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ip, mp = self.samples[index]
        with h5py.File(ip, "r") as hf:
            image = np.asarray(hf["img"][:], dtype=np.float32)
        image = image.transpose(2, 0, 1)  # CHW
        for i in range(14):
            image[i] = (image[i] - self.mean[i]) / (self.std[i] + 1e-8)

        if mp is not None:
            with h5py.File(mp, "r") as hf:
                label = np.asarray(hf["mask"][:], dtype=np.int64)
            label = np.clip(label, 0, 1)
        else:
            label = np.zeros((128, 128), dtype=np.int64)

        image = torch.from_numpy(image.copy())
        label = torch.from_numpy(label)
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label


class SepNormPseudoDataset(Dataset):
    """Combined dataset: labeled train + pseudo-labeled val/test."""

    def __init__(self, data_root: Path | str, pseudo_labels: dict[str, np.ndarray],
                 pseudo_split: str = "valid", transform=None, confidence_threshold: float = 0.9):
        self.transform = transform
        self.items: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        data_root = Path(data_root)
        train_stats = STATS["train"]
        pseudo_stats = STATS[pseudo_split]

        # Load labeled training data
        train_dir = data_root / "TrainData"
        for p in sorted((train_dir / "img").glob("image_*.h5")):
            m = re.search(r"image_(\d+)\.h5$", p.name)
            if not m:
                continue
            mp = train_dir / "mask" / f"mask_{m.group(1)}.h5"
            if not mp.is_file():
                continue
            with h5py.File(p, "r") as hf:
                img = np.asarray(hf["img"][:], dtype=np.float32).transpose(2, 0, 1)
            for i in range(14):
                img[i] = (img[i] - train_stats["mean"][i]) / (train_stats["std"][i] + 1e-8)
            with h5py.File(mp, "r") as hf:
                mask = np.clip(np.asarray(hf["mask"][:], dtype=np.int64), 0, 1)
            self.items.append((img, mask, np.ones_like(mask, dtype=np.float32)))

        # Load pseudo-labeled data
        sub = {"valid": "ValidData", "test": "TestData"}[pseudo_split]
        img_dir = data_root / sub / "img"
        for p in sorted(img_dir.glob("image_*.h5")):
            m = re.search(r"image_(\d+)\.h5$", p.name)
            if not m:
                continue
            fid = m.group(1)
            if fid not in pseudo_labels:
                continue
            with h5py.File(p, "r") as hf:
                img = np.asarray(hf["img"][:], dtype=np.float32).transpose(2, 0, 1)
            for i in range(14):
                img[i] = (img[i] - pseudo_stats["mean"][i]) / (pseudo_stats["std"][i] + 1e-8)
            prob, pred = pseudo_labels[fid]  # (confidence, prediction)
            weight = (prob >= confidence_threshold).astype(np.float32)
            self.items.append((img, pred.astype(np.int64), weight))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, mask, weight = self.items[index]
        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())
        weight = torch.from_numpy(weight.copy())
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            # weight follows mask transforms - simplified: same spatial transforms
        return img, mask, weight
