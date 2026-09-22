"""Evaluate a trained checkpoint on one subset of a line's split.

Produces metrics_<subset>.json (overall metrics as in Table 6/8, per-class metrics as in
Table 7, raw and row-normalised confusion matrices as in Figure 5), per_class_<subset>.csv
and predictions_<subset>.csv (one row per image with plant/week/view and class
probabilities).

Usage:
    python scripts/evaluate.py --line 1 --model swin_b --split random \
        --checkpoint runs/line1_swin_b_random/best_validation_weights.pt --subset val
    python scripts/evaluate.py --line 3 --model convnext_b --split plant \
        --checkpoint runs/line3_convnext_b_plant/best_validation_weights.pt --subset test --plot
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from plant_health.data.builder import build_eval_dataset
from plant_health.engine.evaluate import evaluate_dataset, save_report
from plant_health.models.builder import generate_model
from plant_health.utils.config import build_config, common_parser


def main():
    parser = common_parser('Evaluate a checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--out_dir', default=None, help='default: <checkpoint dir>/eval')
    parser.add_argument('--plot', action='store_true', help='also save a Figure-5-style confusion matrix')
    args = parser.parse_args()

    cfg = build_config(args.config, args.line, args.model, args.split, args.opts)
    dataset, classes = build_eval_dataset(cfg, args.subset)
    model = generate_model(cfg, len(classes['class_names']), checkpoint=args.checkpoint)
    report, frame = evaluate_dataset(model, dataset, classes, cfg.train.average,
                                     cfg.train.batch_size, cfg.train.num_workers)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), 'eval')
    save_report(report, frame, out_dir, args.subset)

    print('Overall:', report['overall'])
    for row in report['per_class']:
        print('  {class}: precision {precision} recall {recall} f1 {f1} (n={support})'.format(**row))
    print('Confusion matrix (counts):')
    print(np.array(report['confusion_matrix']))

    if args.plot:
        from plant_health.plotting import plot_confusion_matrix
        path = os.path.join(out_dir, 'confusion_matrix_{}.png'.format(args.subset))
        plot_confusion_matrix(np.array(report['confusion_matrix_row_normalized']), classes['class_names'], path)
        print('Saved', path)
    print('Results written to', out_dir)


if __name__ == '__main__':
    main()
