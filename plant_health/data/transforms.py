"""Image pre-processing and augmentation (Sec. 3.1.3).

Per-image augmentations: rotation (+/- 20 degrees), horizontal flip, random crop.
Batch-level augmentations (CutMix, Mixup, label smoothing 0.1) are applied in the
training loop through ``build_mixup`` because they operate on whole mini-batches.
Images are normalised with mean/std and resized to the model's input size
(224 x 224 for VGG-16, ResNet-18 and ViT; 384 x 384 for Swin Transformer).
"""
from torchvision import transforms


def build_transforms(cfg):
    size = int(cfg.data.input_size)
    aug = cfg.augmentation
    normalize = transforms.Normalize(mean=cfg.data.mean, std=cfg.data.std)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=tuple(aug.random_crop_scale)),
        transforms.RandomHorizontalFlip(p=aug.horizontal_flip),
        transforms.RandomRotation(degrees=aug.rotation_degrees),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def build_mixup(cfg, num_classes):
    """timm Mixup/CutMix with label smoothing; returns None if both alphas are 0."""
    aug = cfg.augmentation
    if aug.mixup_alpha <= 0 and aug.cutmix_alpha <= 0:
        return None
    from timm.data import Mixup
    return Mixup(
        mixup_alpha=aug.mixup_alpha,
        cutmix_alpha=aug.cutmix_alpha,
        prob=aug.mixup_prob,
        switch_prob=aug.mixup_switch_prob,
        mode='batch',
        label_smoothing=aug.label_smoothing,
        num_classes=num_classes,
    )
