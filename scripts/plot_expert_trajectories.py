"""Health-state progression of all plants from the expert annotations (Sec. 4.3, Figure 11).

No model is involved: this plots the expert label of every plant in every week of a
line (state 0 = plant mortality) with a moving average of the weekly mean. When several
views of a plant-week disagree, the most frequent label is used ([ASSUMPTION]).

Usage:
    python scripts/plot_expert_trajectories.py                # all lines
    python scripts/plot_expert_trajectories.py --lines 2 3 --window 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plant_health.data.annotations import load_annotations, select_line
from plant_health.plotting import plot_expert_trajectories
from plant_health.utils.config import REPO_ROOT, load_yaml

VARIETY = {1: 'Nonari-Cherry', 2: 'Amos Coli', 3: 'Amos Coli', 4: 'Dafnis-Hybrid'}  # Table 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lines', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--window', type=int, default=3, help='moving-average window in weeks [ASSUMPTION]')
    parser.add_argument('--out_dir', default=os.path.join(REPO_ROOT, 'results', 'figures', 'generated'))
    args = parser.parse_args()

    cfg = load_yaml(os.path.join(REPO_ROOT, 'configs', 'default.yaml'))
    df = load_annotations(cfg['base']['annotations'])
    os.makedirs(args.out_dir, exist_ok=True)
    for line in args.lines:
        d = select_line(df, line)
        if d.empty:
            continue
        per_week = (d.groupby(['plant_id', 'week'])['health_level']
                    .agg(lambda s: s.mode().iloc[0]).reset_index())
        path = os.path.join(args.out_dir, 'fig11_expert_trajectories_line{}.png'.format(line))
        plot_expert_trajectories(per_week, path, title='Line {} ({})'.format(line, VARIETY[line]), window=args.window)
        print('Saved', path)


if __name__ == '__main__':
    main()
