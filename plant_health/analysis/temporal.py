"""Spatio-temporal plant state estimation (Eq. 1, Figure 4, Sec. 4.2).

    S_t = f( O_{i=1..N} phi(F_t^i) )

F_t^i are the features of the i-th image of a plant at time t (the left/right/top views),
phi is the shared feature extractor + classifier applied to each image, O combines the N
per-image outputs, and f maps the combined output to a health state 1-5.

The paper trains and evaluates per image and does not specify O. This module provides
the combination rules selectable with ``inference.view_aggregation``
([ASSUMPTION], default ``mean_prob``):

* mean_prob     - average the SoftMax probabilities of all views, take the arg-max
* max_prob      - take the single most confident view
* majority_vote - most frequent per-view prediction (ties -> higher mean probability)
"""
import numpy as np
import pandas as pd


def prob_columns(frame):
    return [c for c in frame.columns if c.startswith('prob_')]


def aggregate_views(pred_frame, index_to_level, method='mean_prob'):
    """Collapse per-image predictions to one state per (plant, week).

    ``pred_frame`` is a predictions_<subset>.csv written by the evaluation code.
    Returns a DataFrame with plant_id, week, expert_level, predicted_level, n_views.
    """
    cols = prob_columns(pred_frame)
    rows = []
    for (plant, week), g in pred_frame.groupby(['plant_id', 'week'], sort=True):
        probs = g[cols].to_numpy()
        if method == 'mean_prob':
            k = int(probs.mean(0).argmax())
        elif method == 'max_prob':
            k = int(np.unravel_index(probs.argmax(), probs.shape)[1])
        elif method == 'majority_vote':
            votes = np.bincount(probs.argmax(1), minlength=probs.shape[1])
            best = np.flatnonzero(votes == votes.max())
            k = int(best[probs.mean(0)[best].argmax()])
        else:
            raise ValueError('unknown aggregation: {}'.format(method))
        expert = g['health_level'].mode()
        rows.append({
            'plant_id': plant,
            'week': int(week),
            'expert_level': int(expert.iloc[0]),
            'predicted_level': int(index_to_level[k]),
            'confidence': float(probs.mean(0)[k]),
            'n_views': int(len(g)),
        })
    return pd.DataFrame(rows)


def moving_average(values, window=3):
    s = pd.Series(values, dtype=float)
    return s.rolling(window, min_periods=1, center=True).mean().to_numpy()
