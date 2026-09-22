"""Evaluation metrics of Sec. 2.4.2 (Eqs. 2-5).

* accuracy  = (TP + TN) / (TP + FP + FN + TN), computed over all samples
* precision = TP / (TP + FP)
* recall    = TP / (TP + FN)
* F1        = 2 * precision * recall / (precision + recall)

Multi-class precision/recall/F1 are averaged over the retained classes; macro averaging
is the default ([ASSUMPTION], configurable via ``train.average``). Per-class values
correspond to Table 7, and the row-normalised confusion matrix to Figure 5.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


class Estimator:
    """Accumulates predictions over an epoch and computes the paper's metrics."""

    def __init__(self, num_classes, average='macro'):
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self):
        self.y_true, self.y_pred = [], []

    def update(self, logits, targets):
        """``logits``: (B, C) tensor; ``targets``: (B,) integer tensor."""
        self.y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
        self.y_true.extend(targets.detach().cpu().tolist())

    def get_scores(self, digits=4):
        return compute_scores(self.y_true, self.y_pred, self.num_classes, self.average, digits)

    def get_conf_mat(self, normalize=None):
        return compute_confusion(self.y_true, self.y_pred, self.num_classes, normalize)


def compute_scores(y_true, y_pred, num_classes, average='macro', digits=4):
    labels = list(range(num_classes))
    if len(y_true) == 0:
        return {'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0)
    scores = {'acc': accuracy_score(y_true, y_pred), 'precision': p, 'recall': r, 'f1': f}
    return {k: round(float(v), digits) for k, v in scores.items()}


def per_class_scores(y_true, y_pred, class_names, digits=4):
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return [
        {'class': name, 'precision': round(float(p[i]), digits), 'recall': round(float(r[i]), digits),
         'f1': round(float(f[i]), digits), 'support': int(s[i])}
        for i, name in enumerate(class_names)
    ]


def compute_confusion(y_true, y_pred, num_classes, normalize=None):
    """``normalize='true'`` gives row-normalised matrices as plotted in Figure 5."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize=normalize)


def temporal_agreement(y_true, y_pred):
    """Fraction of (plant, week) states where the prediction equals the expert label (Sec. 4.2)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0
