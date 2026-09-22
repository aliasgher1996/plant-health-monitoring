"""Correctly classified examples with prediction confidence (Sec. 3.6, Figure 9).

Reads the predictions CSV written by main.py / scripts/evaluate.py, so no model is needed.

Usage:
    python scripts/qualitative_examples.py \
        --predictions runs/line1_swin_b_random/eval/predictions_val.csv --per_class 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from plant_health.analysis.temporal import prob_columns
from plant_health.data.dataset import pil_loader
from plant_health.plotting import plot_prediction_grid
from plant_health.utils.config import REPO_ROOT, load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--image_root', default=None, help='default: base.image_root from configs/default.yaml')
    parser.add_argument('--per_class', type=int, default=3)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    root = args.image_root or load_yaml(os.path.join(REPO_ROOT, 'configs', 'default.yaml'))['base']['image_root']
    df = pd.read_csv(args.predictions, dtype={'plant_id': str})
    df['confidence'] = df[prob_columns(df)].max(axis=1)
    correct = df[df['pred_label'] == df['label']]
    picks = (correct.sort_values('confidence', ascending=False)
             .groupby('health_level', group_keys=False).head(args.per_class)
             .sort_values(['health_level', 'confidence'], ascending=[True, False]))

    images = [pil_loader(os.path.join(root, p)) for p in picks['image_path']]
    captions = ['Plant {} wk {} ({})\nClass {} | conf {:.2f}'.format(r.plant_id, r.week, r.view, r.pred_level, r.confidence)
                for r in picks.itertuples()]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.predictions)), 'qualitative_examples.png')
    plot_prediction_grid(images, captions, out, ncols=args.per_class)
    print('Saved', out)


if __name__ == '__main__':
    main()
