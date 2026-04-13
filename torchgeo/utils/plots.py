"""Visualization utilities for multi-band satellite imagery and segmentation masks."""
import cv2
import numpy as np
import torch


def make_color_mask(mask: np.ndarray, cmap: dict) -> np.ndarray:
    """Colorize an integer class-id mask using an external colormap.

    Args:
        mask: (H, W) integer array of class ids.
        cmap: dict mapping ``class_id -> (R, G, B[, A])`` with values in 0-255.
            Any dataset-specific palette works — e.g. ``CDL.cmap`` for CDL, or
            a custom dict you build yourself. No dataset import is required here.
    """
    h, w = mask.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgba in cmap.items():
        color_label[mask == int(class_id)] = rgba[:3]
    return color_label


def plot_sat_image_and_mask(img, mask, cmap, save_file_prefix='sample',
                            rgb_bands=(2, 1, 0), alpha=0.3):
    """Save a satellite RGB image, a color-coded mask, and their overlay.

    Args:
        img: (C, H, W) tensor or (H, W, C) numpy, multi-band satellite image.
        mask: (H, W) tensor/array of integer class ids.
        cmap: dict mapping ``class_id -> (R, G, B[, A])``. Passed straight through
            to :func:`make_color_mask` — keeps this module independent of any
            specific torchgeo dataset.
        save_file_prefix: files are written as ``{prefix}_image.png``,
            ``{prefix}_mask.png``, ``{prefix}_overlay.png``.
        rgb_bands: indices into the channel axis selecting (R, G, B) bands.
        alpha: blend weight for the mask in the overlay.
    """
    # if img is a tensor, convert to a numpy array
    if torch.is_tensor(img):  # input: (C, H, W) tensor, raw Landsat reflectance values -> (H, W, 3)
        img = img[list(rgb_bands), :, :].cpu().numpy().transpose(1, 2, 0)
    else:
        # (H, W, C) numpy image (RGB) -> (H, W, 3)
        img = img[:, :, list(rgb_bands)]

    img = histogram_stretch(img)  # percentile stretch + conversion to uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

    save_img_file = f"{save_file_prefix}_image.png"
    save_mask_file = f"{save_file_prefix}_mask.png"
    save_overlay_file = f"{save_file_prefix}_overlay.png"

    cv2.imwrite(save_img_file, img)

    # if label_img is tensor, convert to cv2 image
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy().astype(np.uint8)

    color_mask = make_color_mask(mask, cmap)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

    cv2.imwrite(save_mask_file, color_mask)

    overlay = cv2.addWeighted(img, 1.0, color_mask, alpha, 0) # overlays image and label

    cv2.imwrite(save_overlay_file, overlay)


def plot_detection_overlay(img, boxes, labels, masks=None,
                           save_file='sample_overlay.png',
                           class_names=None, alpha=0.5, seed=42):
    """Draw bounding boxes (and optional instance masks) on an RGB image and save as PNG.

    Args:
        img: (3, H, W) tensor (float in [0, 255]) or (H, W, 3) numpy uint8/float, RGB order
        boxes: (N, 4) tensor/array in xyxy format, coords in the img's pixel space
        labels: (N,) tensor/array of class ids
        masks: optional (N, H, W) tensor/array of binary instance masks. If provided,
            each instance's segmentation is rendered as a semi-transparent color fill
            plus a solid contour outline in the same color as its bbox.
        save_file: output PNG path
        class_names: optional list mapping label_id -> str for text annotations
        alpha: opacity for mask fill (0=invisible, 1=solid)
        seed: RNG seed so instance colors are reproducible
    """
    # --- image: -> (H, W, 3) uint8 BGR canvas
    if torch.is_tensor(img):
        img_np = img.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
            img_np = img_np.transpose(1, 2, 0)
    else:
        img_np = np.asarray(img)

    if img_np.dtype != np.uint8:
        max_val = float(img_np.max()) if img_np.size else 0.0
        if max_val <= 1.5:  # values in [0, 1]
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        else:  # values already in [0, 255]
            img_np = img_np.clip(0, 255).astype(np.uint8)

    canvas = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).copy()
    h, w = canvas.shape[:2]

    # --- boxes / labels / masks -> numpy
    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if masks is not None and torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()

    n = len(boxes)
    rng = np.random.default_rng(seed)
    colors = [tuple(int(c) for c in rng.integers(64, 256, size=3)) for _ in range(n)]

    # --- per-instance segmentation overlay (proper alpha compositing + contour)
    if masks is not None and len(masks) > 0:
        for i in range(min(n, len(masks))):
            m = masks[i] > 0.5
            if not m.any():
                continue
            color_arr = np.array(colors[i], dtype=np.float32)
            # Alpha blend: out = canvas*(1-alpha) + color*alpha, only on mask pixels
            canvas[m] = (canvas[m].astype(np.float32) * (1 - alpha)
                         + color_arr * alpha).astype(np.uint8)
            # Draw a solid contour on top so the mask boundary is crisp
            m_u8 = m.astype(np.uint8) * 255
            contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, colors[i], 1)

    # --- bboxes + class labels on top
    for i in range(n):
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
        x1, x2 = max(0, x1), min(w - 1, x2)
        y1, y2 = max(0, y1), min(h - 1, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[i], 1)

        label_id = int(labels[i]) if i < len(labels) else -1
        if class_names is not None and 0 <= label_id < len(class_names):
            text = class_names[label_id]
        else:
            text = str(label_id)
        cv2.putText(canvas, text, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1, cv2.LINE_AA)

    cv2.imwrite(save_file, canvas)


def histogram_stretch(img, lower_percentile=2, upper_percentile=98):
    """Percentile stretch across all channels jointly.

    Computing a single (min, max) over all bands preserves inter-band brightness
    ratios, which is required to keep natural-color balance in RGB composites.
    Per-band stretching would independently remap each band to [0, 255] and
    destroy those ratios (e.g. giving vegetation a blue cast).
    """
    non_zero_pixels = img[img > 0]
    if non_zero_pixels.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    min_val = np.percentile(non_zero_pixels, lower_percentile)
    max_val = np.percentile(non_zero_pixels, upper_percentile)
    if max_val <= min_val:
        return np.zeros_like(img, dtype=np.uint8)

    stretched = np.clip((img - min_val) * 255 / (max_val - min_val), 0, 255)
    return stretched.astype(np.uint8)


