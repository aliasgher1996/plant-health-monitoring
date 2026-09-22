"""Gather the summary.json of every run into one table and place it next to the values
reported in the paper, so a re-run can be compared with Table 6 / Table 8.

Output goes to results/reproduced/ (kept separate from results/paper_results/).

Usage:
    python scripts/collect_results.py --split random     # Table 6 layout
    python scripts/collect_results.py --split plant      # Table 8 layout
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from plant_health.utils.config import REPO_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=os.path.join(REPO_ROOT, 'runs'))
    parser.add_argument('--split', choices=['random', 'plant'], required=True)
    parser.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'results', 'reproduced'))
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs, '*', 'summary.json'))):
        with open(path) as f:
            s = json.load(f)
        if s['split'] != args.split:
            continue
        row = {'Line': s['line'], 'Architecture': s['model'], 'run': os.path.basename(os.path.dirname(path)),
               'Training Accuracy': s['train']['acc'], 'Validation Accuracy': s['val']['acc']}
        if args.split == 'random':
            row.update({'Precision': s['val']['precision'], 'Recall': s['val']['recall'], 'F1-Score': s['val']['f1']})
        else:
            row['Test Accuracy'] = s['test']['acc']
        rows.append(row)
    if not rows:
        print('No finished runs with split={} found under {}'.format(args.split, args.runs))
        return

    ours = pd.DataFrame(rows)
    with open(os.path.join(REPO_ROOT, 'results', 'paper_results', 'reported_results.json')) as f:
        reported = json.load(f)
    key = 'table6_performance_random_split' if args.split == 'random' else 'table8_plant_based_partitioning'
    paper = pd.DataFrame(reported[key]['rows']).rename(columns={'line': 'Line', 'architecture': 'Architecture'})
    merged = ours.merge(paper, on=['Line', 'Architecture'], how='left', suffixes=('', ' (paper)'))

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'reproduced_{}_split.csv'.format(args.split))
    merged.to_csv(out, index=False)
    print(merged.to_string(index=False))
    print('Saved', out)


if __name__ == '__main__':
    main()
