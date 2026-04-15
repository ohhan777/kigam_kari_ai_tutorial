import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils.landslides4sense_dataset import Landslides4SenseDataset
from utils.metrics import ConfusionMatrix
from utils.loss import ce_loss
from utils.plots import plot_image
from models.unet import UNet
import torch.optim as optim
import time
import wandb
from pathlib import Path


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def ddp_mean(value, device):
    """Average a scalar across all DDP ranks (no-op when DDP is not active)."""
    if not dist.is_initialized():
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def setup_ddp():
    """Initialize DDP process group. Called only when torchrun sets RANK env var."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(opt):
    epochs = opt.epochs
    batch_size = opt.batch_size
    name = opt.name

    # DDP setup (torchrun sets RANK env var)
    ddp = 'RANK' in os.environ
    if ddp:
        local_rank = setup_ddp()
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb settings (rank 0 only)
    if is_main_process():
        wandb.init(id=opt.name, resume='allow', project=Path(__file__).parent.stem)
        wandb.config.update(opt, allow_val_change=True)

    # Train dataset
    train_dataset = Landslides4SenseDataset('./data/landslides4sense', split='train')
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    # Train dataloader
    num_workers = min([os.cpu_count(), batch_size, 16])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=(train_sampler is None), sampler=train_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    # Validation dataset
    val_dataset = Landslides4SenseDataset('./data/landslides4sense', split='val')
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    # Network model
    num_classes = 2
    model = UNet(n_classes=num_classes, n_channels=14)
    model.to(device)

    # GPU-support
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    elif torch.cuda.device_count() > 1:   # multi-GPU (DP fallback)
        model = torch.nn.DataParallel(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # AMP mixed-precision (enabled only on CUDA)
    use_amp = (not opt.no_amp) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # loading a weight file (if exists)
    os.makedirs('weights', exist_ok=True)
    weight_file = Path('weights')/(name + '.pth')
    best_f1 = 0.0
    best_epoch = -1
    best_metrics = {}
    start_epoch, end_epoch = (0, epochs)
    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file, map_location=device, weights_only=False)
        # load into unwrapped model
        unwrapped = model.module if hasattr(model, 'module') else model
        unwrapped.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        best_epoch = checkpoint.get('best_epoch', -1)
        best_metrics = checkpoint.get('best_metrics', {})
        if is_main_process():
            print('resumed from epoch %d' % start_epoch)

    confusion_matrix = ConfusionMatrix(num_classes)

    # training/val
    for epoch in range(start_epoch, end_epoch):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if is_main_process():
            print('epoch: %d/%d' % (epoch+1, end_epoch))
        t0 = time.time()
        # training
        epoch_loss = train_one_epoch(train_dataloader, model, optimizer, scaler, use_amp, device)
        epoch_loss = ddp_mean(epoch_loss, device)    # sync across DDP ranks
        t1 = time.time()
        if is_main_process():
            print('loss=%.4f (took %.2f sec)' % (epoch_loss, t1-t0))
        lr_scheduler.step(epoch_loss)
        # val
        val_epoch_loss = val_one_epoch(val_dataloader, model, confusion_matrix, use_amp, device)
        val_epoch_loss = ddp_mean(val_epoch_loss, device)
        confusion_matrix.sync(device)  # sync across DDP ranks (no-op if single GPU)
        val_epoch_iou = confusion_matrix.get_iou()
        val_epoch_mean_iou = confusion_matrix.get_mean_iou()
        val_epoch_f1 = confusion_matrix.get_f1(cls=1)
        val_epoch_precision = confusion_matrix.get_precision(cls=1)
        val_epoch_recall = confusion_matrix.get_recall(cls=1)

        if is_main_process():
            print('[val] loss=%.4f, mean iou=%.4f, F1=%.4f (P=%.4f, R=%.4f)' %
                  (val_epoch_loss, val_epoch_mean_iou, val_epoch_f1, val_epoch_precision, val_epoch_recall))
            print('class IoU: [' + ', '.join([('%.4f' % (x)) for x in val_epoch_iou]) + ']')

            # saving the best status into a weight file
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            if val_epoch_f1 > best_f1:
                 best_weight_file = Path('weights')/(name + '_best.pth')
                 best_f1 = val_epoch_f1
                 best_epoch = epoch
                 best_metrics = {'f1': val_epoch_f1, 'precision': val_epoch_precision,
                                 'recall': val_epoch_recall, 'mean_iou': val_epoch_mean_iou,
                                 'loss': val_epoch_loss}
                 state = {'model': model_state, 'epoch': epoch, 'best_f1': best_f1,
                          'best_epoch': best_epoch, 'best_metrics': best_metrics}
                 torch.save(state, best_weight_file)
                 print('best F1=>saved\n')

            # saving the current status into a weight file
            state = {'model': model_state, 'epoch': epoch, 'best_f1': best_f1,
                     'best_epoch': best_epoch, 'best_metrics': best_metrics}
            torch.save(state, weight_file)
            # wandb logging
            wandb.log({'train_loss': epoch_loss, 'val_loss': val_epoch_loss,
                       'val_mean_iou': val_epoch_mean_iou, 'val_f1': val_epoch_f1,
                       'val_precision': val_epoch_precision, 'val_recall': val_epoch_recall})

    if is_main_process() and best_epoch >= 0:
        print('\n=== Training finished ===')
        print('Best epoch: %d/%d' % (best_epoch+1, end_epoch))
        print('  F1=%.4f, Precision=%.4f, Recall=%.4f, Mean IoU=%.4f, Loss=%.4f' %
              (best_metrics['f1'], best_metrics['precision'], best_metrics['recall'],
               best_metrics['mean_iou'], best_metrics['loss']))

    cleanup_ddp()


def train_one_epoch(train_dataloader, model, optimizer, scaler, use_amp, device):
    model.train()
    losses = []
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    for i, (imgs, targets, _) in enumerate(train_dataloader):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(amp_device, enabled=use_amp):
            preds = model(imgs)                # forward, preds: (B, C, H, W)
            loss = ce_loss(preds, targets)     # calculates the iteration loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_val = loss.item()
        if is_main_process():
            print('\t iteration: %d/%d, loss=%.4f' % (i, len(train_dataloader)-1, loss_val))
        losses.append(loss_val)
    return sum(losses) / max(len(losses), 1)


def val_one_epoch(val_dataloader, model, confusion_matrix, use_amp, device):
    model.eval()
    losses = []
    confusion_matrix.reset()
    amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
    for i, (imgs, targets, _) in enumerate(val_dataloader):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(amp_device, enabled=use_amp):
            preds = model(imgs)                # forward, preds: (B, C, H, W)
            loss = ce_loss(preds, targets)
        losses.append(loss.item())
        preds = torch.argmax(preds, dim=1)  # (B, H, W)
        confusion_matrix.process_batch(preds, targets)
        # sample images (overwrite the previous ones)
        if i == 0 and is_main_process():
            os.makedirs('outputs', exist_ok=True)
            for j in range(min(5, preds.size(0))):
                save_file = os.path.join('outputs', 'val_%d.png' % (j))
                plot_image(imgs[j], pred=preds[j], gt=targets[j], save_file=save_file)

    return sum(losses) / max(len(losses), 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--name', default='landslide_unet_adam_ce', help='name for the run')
    parser.add_argument('--no-amp', action='store_true', default=False, help='disable automatic mixed precision')

    opt = parser.parse_args()

    train(opt)
