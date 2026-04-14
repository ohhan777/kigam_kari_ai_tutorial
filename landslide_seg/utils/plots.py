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


def plot_image(img, label=None, save_file='image.png', alpha=0.3):
    # if img is a tensor, convert to a numpy array
    if torch.is_tensor(img):  # input: (14, H, W) tensor, MEAN/STD normalized
        img = img.cpu().numpy()

    # select 3 bands for RGB visualization (bands 2,1,0 = Red,Green,Blue from Sentinel-2)
    rgb = img[[2, 1, 0], :, :].transpose(1, 2, 0)  # (H, W, 3)
    rgb = histogram_stretch(rgb)  # histogram stretch for better visualization
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

    if label is not None:
        # if label is tensor, convert to numpy
        if torch.is_tensor(label):
            label = label.cpu().numpy().astype(np.uint8)

        color_label = make_color_label(label)
        color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)  # RGB to BGR for cv2

        rgb = cv2.addWeighted(rgb, 1.0, color_label, alpha, 0)  # overlays image and label

    # save image
    os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
    cv2.imwrite(save_file, rgb)


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
