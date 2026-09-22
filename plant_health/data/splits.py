"""The two data-partitioning protocols used in the paper.

* ``random_split``: image-level random 80:20 train/validation split (Sec. 3.1.1, Table 4).
  Used for Tables 6 and 7 and Figures 5-9.
* ``plant_split``: plant-level 70:20:10 train/validation/test split (Sec. 4.1, Table 8), so
  that no plant contributes images to more than one subset.
"""
import json
import os

import numpy as np
import pandas as pd


def random_split(df, train_frac=0.8, seed=42, stratify=True):
    from sklearn.model_selection import train_test_split
    labels = df['health_level'] if stratify else None
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), train_size=train_frac, random_state=seed, shuffle=True, stratify=labels)
    return {
        'train': df.iloc[np.sort(train_idx)].reset_index(drop=True),
        'val': df.iloc[np.sort(val_idx)].reset_index(drop=True),
    }


def plant_split(df, fractions, seed=42):
    """Assign whole plants to train/val/test with the given fractions of plants."""
    plants = np.array(sorted(df['plant_id'].unique(), key=_natural_key))
    rng = np.random.RandomState(seed)
    rng.shuffle(plants)
    n = len(plants)
    n_train = int(round(fractions['train'] * n))
    n_val = int(round(fractions['val'] * n))
    groups = {
        'train': set(plants[:n_train]),
        'val': set(plants[n_train:n_train + n_val]),
        'test': set(plants[n_train + n_val:]),
    }
    out = {name: df[df['plant_id'].isin(ids)].reset_index(drop=True) for name, ids in groups.items()}
    check_plant_disjoint(out)
    return out


def check_plant_disjoint(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]['plant_id']) & set(splits[b]['plant_id'])
            if shared:
                raise AssertionError('plants {} appear in both {} and {}'.format(sorted(shared), a, b))


def write_splits(splits, out_dir, kept_levels, meta=None):
    """Write <out_dir>/{train,val[,test]}.csv and classes.json (level -> class index)."""
    os.makedirs(out_dir, exist_ok=True)
    level_to_index = {int(l): i for i, l in enumerate(sorted(kept_levels))}
    for name, part in splits.items():
        part = part.copy()
        part['label'] = part['health_level'].map(level_to_index).astype(int)
        part.to_csv(os.path.join(out_dir, '{}.csv'.format(name)), index=False)
    info = {
        'health_levels': sorted(int(l) for l in kept_levels),
        'class_names': ['Class {}'.format(l) for l in sorted(kept_levels)],
        'level_to_index': {str(k): v for k, v in level_to_index.items()},
        'sizes': {name: int(len(part)) for name, part in splits.items()},
        'plants': {name: int(part['plant_id'].nunique()) for name, part in splits.items()},
    }
    if meta:
        info.update(meta)
    with open(os.path.join(out_dir, 'classes.json'), 'w') as f:
        json.dump(info, f, indent=2)
    return info


def read_split(split_dir, name):
    path = os.path.join(split_dir, '{}.csv'.format(name))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={'plant_id': str})


def read_classes(split_dir):
    with open(os.path.join(split_dir, 'classes.json')) as f:
        return json.load(f)


def _natural_key(value):
    s = str(value)
    return (0, int(s)) if s.isdigit() else (1, s)
