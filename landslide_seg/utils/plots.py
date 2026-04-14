import os
import cv2
import numpy as np
import torch


def make_color_label(label):
    class_colors = [        # RGB color for each class
        [0, 0, 0],          # 0: background (no landslide)
        [255, 0, 0],        # 1: landslide
    ]
    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    for i, class_color in enumerate(class_colors):
        color_label[label == i] = class_color
    return color_label


def plot_image(img, pred=None, gt=None, save_file='image.png', alpha=0.3):
    # if img is a tensor, convert to a numpy array
    if torch.is_tensor(img):  # input: (14, H, W) tensor, MEAN/STD normalized
        img = img.cpu().numpy()

    # select 3 bands for RGB visualization (bands 2,1,0 = Red,Green,Blue from Sentinel-2)
    rgb = img[[2, 1, 0], :, :].transpose(1, 2, 0)  # (H, W, 3)
    rgb = histogram_stretch(rgb)  # histogram stretch for better visualization
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

    if pred is not None:
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy().astype(np.uint8)
        color_pred = make_color_label(pred)
        color_pred = cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(rgb, 1.0, color_pred, alpha, 0)

    if gt is not None and pred is not None:
        # 2x2: [Image, GT] / [Overlay, Pred]
        if torch.is_tensor(gt):
            gt = gt.cpu().numpy().astype(np.uint8)
        color_gt = make_color_label(gt)
        color_gt = cv2.cvtColor(color_gt, cv2.COLOR_RGB2BGR)
        top = np.hstack([rgb, color_gt])
        bottom = np.hstack([overlay, color_pred])
        canvas = np.vstack([top, bottom])
    elif pred is not None:
        # 1x3: [Image, Overlay, Pred]
        canvas = np.hstack([rgb, overlay, color_pred])
    else:
        canvas = rgb

    # save image
    os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
    cv2.imwrite(save_file, canvas)


def histogram_stretch(img, lower_percentile=2, upper_percentile=98):
    stretched_img = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        band = img[:, :, i]
        min_val = np.percentile(band, lower_percentile)
        max_val = np.percentile(band, upper_percentile)
        if max_val > min_val:
            stretched_band = np.clip((band - min_val) * 255 / (max_val - min_val), 0, 255)
            stretched_img[:, :, i] = stretched_band.astype(np.uint8)
    return stretched_img
