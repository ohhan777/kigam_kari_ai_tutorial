# Confusion matrix class for semantic segmentation

import numpy as np
import torch
import torch.distributed as dist


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)

    def process_batch(self, preds, targets):
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        mask = (targets >= 0) & (targets < self.num_classes)

        confusion_mtx = np.bincount(
            self.num_classes * targets[mask].astype(int) + preds[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += confusion_mtx.astype(np.float32)

    def print(self):
        for i in range(self.num_classes):
            print(f"Class {i}: {self.confusion_matrix[i, i]} / {self.confusion_matrix[i].sum()}")

    def get_pix_acc(self):
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)

    def get_class_acc(self):
        class_acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return np.mean(class_acc)

    def get_iou(self):
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        return iou

    def get_mean_iou(self):
        iou = self.get_iou()
        return np.mean(iou)

    def get_freq_weighted_iou(self):
        freq = self.confusion_matrix.sum(axis=1) / (self.confusion_matrix.sum() + 1e-10)
        iou = self.get_iou()
        return (freq[freq > 0] * iou[freq > 0]).sum()

    def get_precision(self, cls=1):
        tp = self.confusion_matrix[cls, cls]
        fp = self.confusion_matrix[:, cls].sum() - tp
        return tp / (tp + fp + 1e-10)

    def get_recall(self, cls=1):
        tp = self.confusion_matrix[cls, cls]
        fn = self.confusion_matrix[cls, :].sum() - tp
        return tp / (tp + fn + 1e-10)

    def get_f1(self, cls=1):
        p = self.get_precision(cls)
        r = self.get_recall(cls)
        return 2 * p * r / (p + r + 1e-10)

    def sync(self, device):
        """Sync confusion matrix across all DDP ranks via all_reduce."""
        if not dist.is_initialized():
            return
        cm_tensor = torch.from_numpy(self.confusion_matrix).to(device)
        dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
        self.confusion_matrix = cm_tensor.cpu().numpy()
