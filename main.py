"""Train one model on one cultivation line, then evaluate its best-validation checkpoint.

Examples
--------
# Table 6 entry: Swin Transformer-B on Line 1, random 80:20 split
python main.py --line 1 --model swin_b --split random

# Table 8 entry: ConvNeXt-B on Line 3, plant-based 70:20:10 split
python main.py --line 3 --model convnext_b --split plant

# Override any config value
python main.py --line 2 --model vgg16 --opts train.epochs=50 train.num_workers=4
"""
import json
import os

from plant_health.data.builder import build_eval_dataset, generate_dataset
from plant_health.engine.evaluate import evaluate_dataset, save_report
from plant_health.engine.train import train
from plant_health.models.builder import generate_model
from plant_health.utils.config import build_config, common_parser, prepare_save_path, print_config
from plant_health.utils.misc import print_msg, set_random_seed


def main():
    args = common_parser('Train a plant health classifier (Fuentes, Asgher et al., 2025).').parse_args()
    cfg = build_config(args.config, args.line, args.model, args.split, args.opts)
    if args.print_config:
        print_config(cfg)

    run_dir = prepare_save_path(cfg, args.config)
    print_msg('Experiment: {}'.format(cfg.experiment_name), ['Output: {}'.format(run_dir)])
    set_random_seed(cfg.base.random_seed, cfg.base.cudnn_deterministic)

    train_set, val_set, test_set, classes = generate_dataset(cfg)
    num_classes = len(classes['class_names'])
    model = generate_model(cfg, num_classes)

    logger = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(os.path.join(run_dir, 'log'))
    except ImportError:
        pass

    train(cfg, model, train_set, val_set, num_classes, logger)

    # ---- evaluate the best validation checkpoint --------------------------------
    best = os.path.join(run_dir, 'best_validation_weights.pt')
    model = generate_model(cfg, num_classes, checkpoint=best)
    summary = {'line': cfg.data.line, 'model': cfg.model.display_name, 'split': cfg.data.split}
    subsets = ['train', 'val'] + (['test'] if test_set is not None else [])
    for subset in subsets:
        # the training subset is re-evaluated without augmentation to obtain "Training Accuracy"
        dataset, _ = build_eval_dataset(cfg, subset)
        report, frame = evaluate_dataset(model, dataset, classes, cfg.train.average,
                                         cfg.train.batch_size, cfg.train.num_workers)
        save_report(report, frame, os.path.join(run_dir, 'eval'), subset)
        summary[subset] = report['overall']
        print_msg('{} ({} images)'.format(subset, report['n_samples']),
                  ['{}: {}'.format(k, v) for k, v in report['overall'].items()])

    with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Summary written to {}'.format(os.path.join(run_dir, 'summary.json')))


if __name__ == '__main__':
    main()
