"""Evaluation of a trained checkpoint: overall metrics (Table 6 / Table 8), class-wise
metrics (Table 7), confusion matrix (Figure 5) and per-image predictions (used for the
temporal analysis of Sec. 4.2 and the qualitative figures)."""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from plant_health.metrics import compute_confusion, compute_scores, per_class_scores


@torch.no_grad()
def predict(model, dataset, batch_size=32, num_workers=4):
    """Returns (probabilities [N, C], labels [N]) in dataset order (dataset must return index)."""
    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = np.zeros((len(dataset), model.num_classes), dtype=np.float32)
    labels = np.zeros(len(dataset), dtype=np.int64)
    for x, y, idx in loader:
        p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()  # SoftMax output (Sec. 3.1.2)
        probs[idx.numpy()] = p
        labels[idx.numpy()] = y.numpy()
    return probs, labels


def evaluate_dataset(model, dataset, classes, average='macro', batch_size=32, num_workers=4):
    probs, labels = predict(model, dataset, batch_size, num_workers)
    preds = probs.argmax(1)
    n = len(classes['class_names'])
    report = {
        'n_samples': int(len(labels)),
        'overall': compute_scores(labels, preds, n, average),
        'per_class': per_class_scores(labels, preds, classes['class_names']),
        'confusion_matrix': compute_confusion(labels, preds, n).tolist(),
        'confusion_matrix_row_normalized': np.round(compute_confusion(labels, preds, n, 'true'), 4).tolist(),
        'class_names': classes['class_names'],
    }
    frame = dataset.frame.copy()
    frame['pred_label'] = preds
    index_to_level = {v: int(k) for k, v in classes['level_to_index'].items()}
    frame['pred_level'] = [index_to_level[int(p)] for p in preds]
    for i, name in enumerate(classes['class_names']):
        frame['prob_{}'.format(name.replace(' ', '_').lower())] = probs[:, i]
    return report, frame


def save_report(report, frame, out_dir, subset):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics_{}.json'.format(subset)), 'w') as f:
        json.dump(report, f, indent=2)
    frame.to_csv(os.path.join(out_dir, 'predictions_{}.csv'.format(subset)), index=False)
    pd.DataFrame(report['per_class']).to_csv(os.path.join(out_dir, 'per_class_{}.csv'.format(subset)), index=False)
