"""Training loop (Sec. 3.1.2 - 3.2).

AdamW + warm-up/cosine schedule, categorical cross-entropy (soft-target cross-entropy
when Mixup/CutMix produce mixed labels, with label smoothing 0.1), batch size 32,
model selection and early stopping on validation F1.
"""
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from plant_health.data.transforms import build_mixup
from plant_health.engine.early_stopping import EarlyStopping
from plant_health.engine.scheduler import build_optimizer, build_scheduler
from plant_health.metrics import Estimator
from plant_health.utils.misc import print_msg, save_weights


def build_loaders(cfg, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory,
                              drop_last=True)  # Mixup/CutMix need even-sized batches
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
    return train_loader, val_loader


def build_criterion(cfg, mixup):
    if mixup is not None:
        from timm.loss import SoftTargetCrossEntropy
        return SoftTargetCrossEntropy()  # labels already smoothed/mixed by timm.data.Mixup
    return nn.CrossEntropyLoss(label_smoothing=cfg.augmentation.label_smoothing)


def train(cfg, model, train_dataset, val_dataset, num_classes, logger=None):
    device = next(model.parameters()).device
    train_loader, val_loader = build_loaders(cfg, train_dataset, val_dataset)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    mixup = build_mixup(cfg, num_classes)
    criterion = build_criterion(cfg, mixup)
    stopper = EarlyStopping(patience=cfg.train.early_stopping_patience, mode='max')
    estimator = Estimator(num_classes, cfg.train.average)

    run_dir = cfg.base.run_dir
    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        estimator.reset()
        epoch_loss = 0.0
        bar = tqdm(train_loader, disable=not cfg.base.progress)
        for step, (x, y) in enumerate(bar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            hard_y = y
            if mixup is not None:
                x, y = mixup(x, y)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            estimator.update(logits, hard_y)  # running accuracy on augmented batches (monitoring only)
            bar.set_description('epoch [{}/{}] loss {:.4f} lr {:.2e}'.format(
                epoch, cfg.train.epochs, epoch_loss / (step + 1), optimizer.param_groups[0]['lr']))

        record = {'epoch': epoch, 'loss': epoch_loss / max(1, len(train_loader)),
                  'lr': optimizer.param_groups[0]['lr'],
                  'train_running': estimator.get_scores()}

        if epoch % cfg.train.eval_interval == 0:
            val_scores = evaluate_loader(model, val_loader, num_classes, cfg.train.average)
            record['val'] = val_scores
            print_msg('Epoch {} validation'.format(epoch),
                      ['{}: {}'.format(k, v) for k, v in val_scores.items()])
            if logger is not None:
                for k, v in val_scores.items():
                    logger.add_scalar('val/{}'.format(k), v, epoch)
                logger.add_scalar('train/loss', record['loss'], epoch)
                logger.add_scalar('train/lr', record['lr'], epoch)

            improved, stop = stopper.step(val_scores[cfg.train.indicator])
            if improved:
                save_weights(model, os.path.join(run_dir, 'best_validation_weights.pt'))
                print_msg('Best validation {} = {:.4f}; checkpoint saved.'.format(
                    cfg.train.indicator, val_scores[cfg.train.indicator]))
            history.append(record)
            if stop:
                print_msg('Early stopping at epoch {} (no {} improvement for {} epochs).'.format(
                    epoch, cfg.train.indicator, cfg.train.early_stopping_patience), warning=True)
                break
        else:
            history.append(record)

    save_weights(model, os.path.join(run_dir, 'final_weights.pt'))
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    if logger is not None:
        logger.close()
    return history


@torch.no_grad()
def evaluate_loader(model, loader, num_classes, average='macro'):
    device = next(model.parameters()).device
    model.eval()
    estimator = Estimator(num_classes, average)
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        estimator.update(model(x), y)
    model.train()
    return estimator.get_scores()
