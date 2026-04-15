import torch
import torch.nn.functional as F


def ce_loss(preds, targets, ignore_index=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        preds(logits): a tensor of shape [B, C, H, W]
        targets: a tensor of shape [B, H, W].
        ignore_index: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        preds.float(),
        targets.long(),    # [B, H, W]
        ignore_index=ignore_index,
    )
    return ce_loss


def dice_loss(preds, targets, ignore_index=255):
    """Computes the Sorensen-Dice loss.
    Args:
        preds(logits): a tensor of shape [B, C, H, W]
        targets: a tensor of shape [B, H, W].
        ignore_index: target value to exclude from the loss (default: 255).
    Returns:
        dice_loss: the Sorensen-Dice loss.
    """
    num_classes = preds.shape[1]
    targets = targets.long()

    valid = (targets != ignore_index)                                # [B, H, W]
    targets_safe = torch.where(valid, targets, torch.zeros_like(targets))

    true_1_hot = F.one_hot(targets_safe, num_classes=num_classes)    # [B, H, W, C]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2)                      # [B, C, H, W]

    probas = F.softmax(preds, dim=1)
    true_1_hot = true_1_hot.type(preds.type()).contiguous()

    valid_mask = valid.unsqueeze(1).type(preds.type())               # [B, 1, H, W]
    probas = probas * valid_mask
    true_1_hot = true_1_hot * valid_mask

    dims = (0,) + tuple(range(2, targets.ndimension() + 1))    # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)        # intersection w.r.t. the class
    cardinality = torch.sum(probas + true_1_hot, dims)         # cardinality w.r.t. the class

    dice_loss = (2. * intersection / (cardinality + 1e-7)).mean()
    return (1 - dice_loss)


def jaccard_loss(preds, targets, ignore_index=255):
    """Computes the Jaccard loss.
    Args:
        preds(logits): a tensor of shape [B, C, H, W]
        targets: a tensor of shape [B, H, W].
        ignore_index: target value to exclude from the loss (default: 255).
    Returns:
        Jaccard loss
    """
    num_classes = preds.shape[1]
    targets = targets.long()

    valid = (targets != ignore_index)                                # [B, H, W]
    targets_safe = torch.where(valid, targets, torch.zeros_like(targets))

    true_1_hot = F.one_hot(targets_safe, num_classes=num_classes)    # [B, H, W, C]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2)                      # [B, C, H, W]

    probas = F.softmax(preds, dim=1)
    true_1_hot = true_1_hot.type(preds.type()).contiguous()

    valid_mask = valid.unsqueeze(1).type(preds.type())               # [B, 1, H, W]
    probas = probas * valid_mask
    true_1_hot = true_1_hot * valid_mask

    dims = (0,) + tuple(range(2, targets.ndimension() + 1))  # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection

    jacc_loss = (intersection / (union + 1e-7)).mean()
    return (1 - jacc_loss)
