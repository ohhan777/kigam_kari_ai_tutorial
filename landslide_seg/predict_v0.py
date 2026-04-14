import os
import torch
import numpy as np
from tqdm import tqdm
from utils.plots import plot_image
from utils.metrics import ConfusionMatrix
from utils.landsalides4sens_dataset import Landslides4SenseDataset
from models.unet import UNet
import argparse


def predict(opt):
    # Model initialization
    num_classes = 2
    model = UNet(n_classes=num_classes, n_channels=14)

    # GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Load weights
    if not os.path.exists(opt.weight):
        raise FileNotFoundError(f"Model weight file not found: {opt.weight}")
    checkpoint = torch.load(opt.weight, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Validation dataset
    val_dataset = Landslides4SenseDataset(opt.data, train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Create output directory
    os.makedirs(opt.output, exist_ok=True)

    confusion_matrix = ConfusionMatrix(num_classes)

    print('Predicting...')
    model.eval()
    with torch.no_grad():
        for i, (imgs, targets, img_files) in enumerate(tqdm(val_dataloader, desc='Processing')):
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)  # (B, H, W)
            confusion_matrix.process_batch(preds, targets)

            # save sample predictions
            if i < 5:
                for j in range(min(3, preds.size(0))):
                    save_file = os.path.join(opt.output, 'pred_%d_%d.png' % (i, j))
                    plot_image(imgs[j], preds[j], save_file)

    # Print results
    iou = confusion_matrix.get_iou()
    mean_iou = confusion_matrix.get_mean_iou()
    pix_acc = confusion_matrix.get_pix_acc()
    f1 = confusion_matrix.get_f1(cls=1)
    precision = confusion_matrix.get_precision(cls=1)
    recall = confusion_matrix.get_recall(cls=1)

    print('\n=== Results ===')
    print('Pixel Accuracy: %.4f' % pix_acc)
    print('Mean IoU: %.4f' % mean_iou)
    print('Class IoU: [' + ', '.join(['%.4f' % x for x in iou]) + ']')
    print('Landslide F1: %.4f (Precision: %.4f, Recall: %.4f)' % (f1, precision, recall))
    print('Done')

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='./data/landslides4sense', help='Data directory')
    parser.add_argument('--output', '-o', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--weight', '-w', default='weights/landslide_unet_adam_ce_best.pth', help='Weight file path')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    opt = parser.parse_args()

    predict(opt)
