"""t-SNE of the penultimate-layer features of a trained model (Sec. 3.4, Figure 6).

Usage:
    python scripts/tsne.py --line 1 --model swin_b --split random \
        --checkpoint runs/line1_swin_b_random/best_validation_weights.pt --subset val
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader

from plant_health.data.builder import build_eval_dataset
from plant_health.models.builder import extract_features, generate_model
from plant_health.plotting import plot_tsne
from plant_health.utils.config import build_config, common_parser


@torch.no_grad()
def collect_features(model, dataset, batch_size, num_workers):
    device = next(model.parameters()).device
    model.eval()
    feats, labels = [], []
    for x, y, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        feats.append(extract_features(model, x.to(device)).flatten(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    parser = common_parser('t-SNE of learned features (Figure 6).')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--perplexity', type=float, default=30.0)  # [ASSUMPTION] not given in the paper
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = build_config(args.config, args.line, args.model, args.split, args.opts)
    dataset, classes = build_eval_dataset(cfg, args.subset)
    model = generate_model(cfg, len(classes['class_names']), checkpoint=args.checkpoint)
    feats, labels = collect_features(model, dataset, cfg.train.batch_size, cfg.train.num_workers)

    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=args.perplexity, init='pca',
               random_state=cfg.base.random_seed).fit_transform(feats)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), 'eval',
                                   'tsne_line{}_{}.png'.format(cfg.data.line, args.subset))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out.replace('.png', '.npz'), embedding=emb, labels=labels)
    plot_tsne(emb, labels, classes['class_names'], out,
              title='Line {} - {}'.format(cfg.data.line, cfg.model.display_name))
    print('Saved', out)


if __name__ == '__main__':
    main()
