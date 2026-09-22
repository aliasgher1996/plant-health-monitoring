"""Unit tests for class filtering and split protocols on a small synthetic annotation table.
They need no images, no GPU and no model weights."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plant_health.data.annotations import class_distribution, retained_levels  # noqa: E402
from plant_health.data.splits import plant_split, random_split, write_splits, read_classes  # noqa: E402


def synthetic(n_plants=20, weeks=10, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(1, n_plants + 1):
        for w in range(1, weeks + 1):
            level = int(rng.choice([3, 4, 5], p=[0.2, 0.5, 0.3]))
            for v in ('left', 'right', 'top'):
                rows.append({'image_path': 'l1/p{}_w{}_{}.jpg'.format(p, w, v), 'line': 1,
                             'plant_id': str(p), 'week': w, 'view': v, 'health_level': level})
    return pd.DataFrame(rows)


def test_retained_levels_threshold():
    df = pd.DataFrame({'health_level': [1] * 30 + [2] * 31 + [3] * 100})
    assert retained_levels(df, [1, 2, 3, 4, 5], 30, exclude_at_threshold=True) == [2, 3]
    assert retained_levels(df, [1, 2, 3, 4, 5], 30, exclude_at_threshold=False) == [1, 2, 3]


def test_random_split_ratio_and_stratification():
    df = synthetic()
    s = random_split(df, 0.8, seed=1, stratify=True)
    assert len(s['train']) + len(s['val']) == len(df)
    assert abs(len(s['train']) / len(df) - 0.8) < 0.01
    for level in (3, 4, 5):
        frac = (s['train']['health_level'] == level).sum() / (df['health_level'] == level).sum()
        assert abs(frac - 0.8) < 0.02


def test_plant_split_is_disjoint():
    df = synthetic()
    s = plant_split(df, {'train': 0.7, 'val': 0.2, 'test': 0.1}, seed=1)
    ids = [set(s[k]['plant_id']) for k in ('train', 'val', 'test')]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert [len(i) for i in ids] == [14, 4, 2]


def test_write_splits_label_mapping(tmp_path):
    df = synthetic()
    s = random_split(df, 0.8, seed=1)
    info = write_splits(s, str(tmp_path), [3, 4, 5])
    assert info['level_to_index'] == {'3': 0, '4': 1, '5': 2}
    train = pd.read_csv(tmp_path / 'train.csv')
    assert set(train['label']) == {0, 1, 2}
    assert read_classes(str(tmp_path))['class_names'] == ['Class 3', 'Class 4', 'Class 5']


def test_class_distribution_layout():
    table = class_distribution(synthetic())
    assert list(table.columns) == [1, 2, 3, 4, 5, 'Total']
    assert table.loc['Total', 'Total'] == 20 * 10 * 3
