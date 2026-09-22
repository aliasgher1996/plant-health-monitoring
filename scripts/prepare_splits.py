"""Build the train/validation(/test) split files for every cultivation line.

Steps (Sec. 3.1.1 and 4.1):
  1. read data/annotations.csv and keep images rated 1-5 (level 0 = dead plant is dropped);
  2. per line, drop under-represented health classes (<= 30 images by default);
  3. random protocol: image-level stratified 80:20 train/validation split (Table 4);
     plant protocol:  70% / 20% / 10% of plants for train / validation / test (Table 8);
  4. write data/splits/<protocol>/line<k>/{train,val[,test]}.csv and classes.json.

Also writes data/splits/class_distribution.csv (same layout as Table 3) so the local
data can be compared against the counts reported in the paper.

Usage:
    python scripts/prepare_splits.py                       # all lines, both protocols
    python scripts/prepare_splits.py --lines 1 3 --protocols random
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plant_health.data.annotations import (class_distribution, classification_subset, load_annotations,
                                           retained_levels, select_line)
from plant_health.data.splits import plant_split, random_split, write_splits
from plant_health.utils.config import REPO_ROOT, load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', default=os.path.join(REPO_ROOT, 'configs', 'default.yaml'))
    parser.add_argument('--lines', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--protocols', nargs='+', default=['random', 'plant'], choices=['random', 'plant'])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base, data = cfg['base'], cfg['data']
    seed = base['random_seed']
    df = load_annotations(base['annotations'])

    os.makedirs(base['split_dir'], exist_ok=True)
    dist = class_distribution(df, data['health_levels'])
    dist.to_csv(os.path.join(base['split_dir'], 'class_distribution.csv'))
    print('Class distribution (compare with Table 3 of the paper):')
    print(dist.to_string())

    for line in args.lines:
        df_line = classification_subset(select_line(df, line))
        if df_line.empty:
            print('Line {}: no images found, skipped.'.format(line))
            continue
        kept = retained_levels(df_line, data['health_levels'], data['min_samples_per_class'],
                               data['exclude_at_threshold'])
        dropped = sorted(set(df_line['health_level'].unique()) - set(kept))
        df_kept = df_line[df_line['health_level'].isin(kept)].reset_index(drop=True)
        print('\nLine {}: kept classes {} | dropped {} | {} images, {} plants'.format(
            line, kept, dropped, len(df_kept), df_kept['plant_id'].nunique()))

        for protocol in args.protocols:
            if protocol == 'random':
                splits = random_split(df_kept, data['random_split']['train'], seed, data['stratify_random_split'])
            else:
                splits = plant_split(df_kept, data['plant_split'], seed)
            out_dir = os.path.join(base['split_dir'], protocol, 'line{}'.format(line))
            info = write_splits(splits, out_dir, kept, meta={'line': line, 'protocol': protocol, 'seed': seed,
                                                              'dropped_levels': [int(d) for d in dropped]})
            print('  {:<6} -> {} | images {} | plants {}'.format(protocol, out_dir, info['sizes'], info['plants']))


if __name__ == '__main__':
    main()
