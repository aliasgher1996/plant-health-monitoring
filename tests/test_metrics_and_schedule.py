"""Unit tests for the metrics, view aggregation and learning-rate schedule."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plant_health.analysis.temporal import aggregate_views  # noqa: E402
from plant_health.metrics import compute_confusion, compute_scores, per_class_scores  # noqa: E402


def test_scores_perfect_and_macro():
    assert compute_scores([0, 1, 2], [0, 1, 2], 3)['f1'] == 1.0
    s = compute_scores([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert s['acc'] == 0.75
    # class 0: P=1, R=0.5 ; class 1: P=2/3, R=1  -> macro recall 0.75
    assert s['recall'] == 0.75


def test_row_normalised_confusion():
    m = compute_confusion([0, 0, 1, 1], [0, 1, 1, 1], 2, normalize='true')
    assert np.allclose(m, [[0.5, 0.5], [0.0, 1.0]])


def test_per_class_support():
    rows = per_class_scores([0, 1, 1], [0, 1, 0], ['Class 4', 'Class 5'])
    assert [r['support'] for r in rows] == [1, 2]


def test_view_aggregation_mean_prob():
    frame = pd.DataFrame({
        'plant_id': ['1'] * 3, 'week': [1] * 3, 'health_level': [4] * 3,
        'prob_class_4': [0.8, 0.8, 0.1], 'prob_class_5': [0.2, 0.2, 0.9]})
    out = aggregate_views(frame, {0: 4, 1: 5}, 'mean_prob')
    assert out.loc[0, 'predicted_level'] == 4
    out = aggregate_views(frame, {0: 4, 1: 5}, 'max_prob')
    assert out.loc[0, 'predicted_level'] == 5


def test_warmup_cosine_schedule():
    pytest.importorskip('torch')
    from plant_health.engine.scheduler import warmup_cosine_lambda
    fn = warmup_cosine_lambda(total_steps=100, warmup_steps=10, base_lr=3e-6, min_lr=1e-6, warmup_start_factor=0.01)
    assert math.isclose(fn(0), 0.01)
    assert math.isclose(fn(10), 1.0)                 # peak = initial LR 3e-6 after warm-up
    assert math.isclose(fn(100) * 3e-6, 1e-6)        # ends at minimum LR 1e-6
    assert all(fn(s) >= fn(s + 1) - 1e-12 for s in range(10, 100))
