"""Model construction (Sec. 2.4.1 and 3.1.2).

All five backbones are created from ``timm`` with ImageNet pre-trained weights. The
original classification layer is replaced by a new fully connected layer whose number
of outputs equals the number of retained health classes; SoftMax is applied in the loss
(cross-entropy) and at inference. All layers stay trainable.
"""
import torch
import torch.nn as nn

from plant_health.utils.misc import print_msg


def build_model(cfg, num_classes):
    import timm
    kwargs = dict(pretrained=cfg.model.pretrained, num_classes=num_classes)
    if cfg.model.family == 'transformer':
        # keep the positional-embedding / window resolution consistent with the input size
        kwargs['img_size'] = int(cfg.data.input_size)
    model = timm.create_model(cfg.model.timm_name, **kwargs)

    if cfg.model.freeze_backbone:
        head = set(id(p) for p in model.get_classifier().parameters())
        for p in model.parameters():
            p.requires_grad = id(p) in head
    return model


def generate_model(cfg, num_classes, checkpoint=None):
    model = build_model(cfg, num_classes)
    checkpoint = checkpoint or cfg.model.checkpoint
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print_msg('Missing keys when loading {}'.format(checkpoint), [str(missing)], warning=True)
        if unexpected:
            print_msg('Unused keys when loading {}'.format(checkpoint), [str(unexpected)], warning=True)
        print_msg('Loaded weights from {}'.format(checkpoint))
    device = cfg.base.device if torch.cuda.is_available() or cfg.base.device == 'cpu' else 'cpu'
    return model.to(device)


def extract_features(model, x):
    """Penultimate (pre-logits) representation, used for the t-SNE plots (Figure 6)."""
    feats = model.forward_features(x)
    return model.forward_head(feats, pre_logits=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def gradcam_target_layer(model, model_name):
    """Layer whose activations are used for Grad-CAM (Sec. 3.5).

    CNNs: the final convolutional stage. Transformers: the input of the last attention
    block ("taken before the last attention block"), i.e. the first LayerNorm of the
    last block.
    """
    if model_name == 'resnet18':
        return model.layer4
    if model_name == 'vgg16':
        convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1]
    if model_name == 'convnext_b':
        return model.stages[-1]
    if model_name == 'vit_b':
        return model.blocks[-1].norm1
    if model_name == 'swin_b':
        return model.layers[-1].blocks[-1].norm1
    raise KeyError(model_name)
