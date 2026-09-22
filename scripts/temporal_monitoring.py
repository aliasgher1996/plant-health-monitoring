"""Spatio-temporal health diagrams for unseen plants (Sec. 4.2, Figure 10).

Takes per-image predictions of a model trained with the plant-based split (the test
plants were never seen during training), combines the views of each plant at each week
into one state S_t (Eq. 1; aggregation rule = inference.view_aggregation), reports the
agreement with the expert labels (the paper reports 83% for this analysis) and plots
predicted vs. expert trajectories per plant.

Usage (after training with --split plant):
    python scripts/temporal_monitoring.py \
        --predictions runs/line1_swin_b_plant/eval/predictions_test.csv \
        --classes data/splits/plant/line1/classes.json --plants 5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from plant_health.analysis.temporal import aggregate_views
from plant_health.metrics import temporal_agreement
from plant_health.plotting import plot_prediction_vs_expert
from plant_health.utils.config import REPO_ROOT, load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True, help='predictions_<subset>.csv from evaluation')
    parser.add_argument('--classes', required=True, help='classes.json of the split used')
    parser.add_argument('--aggregation', default=None, choices=['mean_prob', 'max_prob', 'majority_vote'])
    parser.add_argument('--plants', nargs='*', default=None, help='plant ids to plot (default: all)')
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    cfg = load_yaml(os.path.join(REPO_ROOT, 'configs', 'default.yaml'))
    method = args.aggregation or cfg['inference']['view_aggregation']
    with open(args.classes) as f:
        classes = json.load(f)
    index_to_level = {v: int(k) for k, v in classes['level_to_index'].items()}

    preds = pd.read_csv(args.predictions, dtype={'plant_id': str})
    states = aggregate_views(preds, index_to_level, method)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.predictions)), 'temporal')
    os.makedirs(out_dir, exist_ok=True)
    states.to_csv(os.path.join(out_dir, 'plant_week_states.csv'), index=False)

    image_level = temporal_agreement(preds['health_level'], preds['pred_level'])
    state_level = temporal_agreement(states['expert_level'], states['predicted_level'])
    summary = {'aggregation': method, 'n_images': int(len(preds)), 'n_plant_weeks': int(len(states)),
               'n_plants': int(states['plant_id'].nunique()),
               'image_level_agreement': round(image_level, 4),
               'plant_week_agreement': round(state_level, 4)}
    with open(os.path.join(out_dir, 'agreement.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plants = args.plants or sorted(states['plant_id'].unique())
    for p in plants:
        traj = states[states['plant_id'] == str(p)]
        if traj.empty:
            print('plant {} not in predictions, skipped'.format(p))
            continue
        plot_prediction_vs_expert(traj, os.path.join(out_dir, 'plant_{}.png'.format(p)), title='Plant {}'.format(p))
    print('Figures written to', out_dir)


if __name__ == '__main__':
    main()
