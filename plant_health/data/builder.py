import os

from plant_health.data.dataset import PlantHealthDataset
from plant_health.data.splits import read_classes, read_split
from plant_health.data.transforms import build_transforms
from plant_health.utils.config import split_dir_for


def generate_dataset(cfg):
    """Return (train, val, test_or_None, classes_info) for the configured line and split protocol."""
    split_dir = split_dir_for(cfg)
    if not os.path.exists(os.path.join(split_dir, 'classes.json')):
        raise FileNotFoundError(
            'No split found in {}. Run scripts/prepare_splits.py first.'.format(split_dir))
    classes = read_classes(split_dir)
    train_tf, eval_tf = build_transforms(cfg)
    root = cfg.base.image_root

    train = PlantHealthDataset(read_split(split_dir, 'train'), root, train_tf)
    val = PlantHealthDataset(read_split(split_dir, 'val'), root, eval_tf)
    test_frame = read_split(split_dir, 'test')
    test = PlantHealthDataset(test_frame, root, eval_tf) if test_frame is not None else None

    print('=========================')
    print('Line {} | split: {} | classes: {}'.format(cfg.data.line, cfg.data.split, classes['class_names']))
    print('Training:   {}'.format(len(train)))
    print('Validation: {}'.format(len(val)))
    print('Test:       {}'.format(len(test) if test is not None else '-'))
    print('=========================')
    return train, val, test, classes


def build_eval_dataset(cfg, subset, return_index=True):
    split_dir = split_dir_for(cfg)
    frame = read_split(split_dir, subset)
    if frame is None:
        raise FileNotFoundError('{}/{}.csv not found'.format(split_dir, subset))
    _, eval_tf = build_transforms(cfg)
    return PlantHealthDataset(frame, cfg.base.image_root, eval_tf, return_index=return_index), read_classes(split_dir)
