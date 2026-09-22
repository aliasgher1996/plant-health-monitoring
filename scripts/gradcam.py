"""Grad-CAM heat maps for a trained model (Sec. 3.5, Figures 7 and 8).

Selects images from a split and overlays the class-activation map of the predicted
class. ``--select misclassified`` collects failure cases such as those in Figure 8.

Usage:
    python scripts/gradcam.py --line 1 --model swin_b --split random \
        --checkpoint runs/line1_swin_b_random/best_validation_weights.pt --num 8
    python scripts/gradcam.py --line 2 --model swin_b --split random \
        --checkpoint runs/line2_swin_b_random/best_validation_weights.pt --select misclassified
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from plant_health.analysis.gradcam import GradCAM, activation_layout, denormalize, num_prefix_tokens
from plant_health.data.builder import build_eval_dataset
from plant_health.engine.evaluate import predict
from plant_health.models.builder import generate_model, gradcam_target_layer
from plant_health.plotting import plot_gradcam_grid
from plant_health.utils.config import build_config, common_parser


def main():
    parser = common_parser('Grad-CAM visualisation (Figures 7-8).')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--select', default='correct', choices=['correct', 'misclassified', 'any'])
    parser.add_argument('--num', type=int, default=8)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = build_config(args.config, args.line, args.model, args.split, args.opts)
    dataset, classes = build_eval_dataset(cfg, args.subset)
    names = classes['class_names']
    model = generate_model(cfg, len(names), checkpoint=args.checkpoint)

    probs, labels = predict(model, dataset, cfg.train.batch_size, cfg.train.num_workers)
    preds = probs.argmax(1)
    if args.select == 'correct':
        pool = np.flatnonzero(preds == labels)
    elif args.select == 'misclassified':
        pool = np.flatnonzero(preds != labels)
    else:
        pool = np.arange(len(labels))
    # take up to --num images spread over the classes
    rng = np.random.RandomState(cfg.base.random_seed)
    chosen = []
    for k in range(len(names)):
        cand = pool[labels[pool] == k]
        rng.shuffle(cand)
        chosen += cand[: max(1, args.num // len(names))].tolist()
    chosen = chosen[: args.num]

    cam_fn = GradCAM(model, gradcam_target_layer(model, cfg.model.name),
                     layout=activation_layout(cfg.model.name),
                     num_prefix_tokens=num_prefix_tokens(model, cfg.model.name))
    device = next(model.parameters()).device
    images, cams, captions = [], [], []
    with torch.enable_grad():
        for i in chosen:
            x, y, _ = dataset[i]
            cam, pred, p = cam_fn(x.unsqueeze(0).to(device))
            images.append(denormalize(x, cfg.data.mean, cfg.data.std))
            cams.append(cam[0])
            captions.append('GT: {} | Pred: {} ({:.2f})'.format(names[y], names[pred[0]], p[0, pred[0]]))
    cam_fn.remove()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), 'eval',
                                   'gradcam_line{}_{}.png'.format(cfg.data.line, args.select))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_gradcam_grid(images, cams, captions, out)
    print('Saved', out)


if __name__ == '__main__':
    main()
