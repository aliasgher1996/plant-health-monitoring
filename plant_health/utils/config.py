"""Configuration loading (YAML + command-line overrides), in the style of the
reference repository's ``utils/func.py``."""
import argparse
import copy
import os
import shutil

import yaml
from munch import munchify

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODELS_YAML = os.path.join(REPO_ROOT, 'configs', 'models.yaml')


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _parse_value(text):
    """Parse 'key=value' override values with YAML semantics (numbers, lists, null...)."""
    return yaml.safe_load(text)


def apply_overrides(cfg, overrides):
    """Apply overrides of the form ``section.key=value``."""
    for item in overrides or []:
        if '=' not in item:
            raise ValueError('Override must look like section.key=value, got: {}'.format(item))
        key, value = item.split('=', 1)
        node = cfg
        parts = key.split('.')
        for p in parts[:-1]:
            if p not in node:
                raise KeyError('Unknown config section: {}'.format(key))
            node = node[p]
        if parts[-1] not in node:
            raise KeyError('Unknown config key: {}'.format(key))
        node[parts[-1]] = _parse_value(value)
    return cfg


def model_spec(name):
    specs = load_yaml(MODELS_YAML)
    if name not in specs:
        raise KeyError('Unknown model "{}". Available: {}'.format(name, ', '.join(specs)))
    return specs[name]


def build_config(config_path, line=None, model=None, split=None, overrides=None):
    """Load the default YAML, apply CLI shortcuts and overrides, and resolve derived values."""
    cfg = copy.deepcopy(load_yaml(config_path))
    if line is not None:
        cfg['data']['line'] = int(line)
    if model is not None:
        cfg['model']['name'] = model
    if split is not None:
        cfg['data']['split'] = split
    apply_overrides(cfg, overrides)

    spec = model_spec(cfg['model']['name'])
    cfg['model']['timm_name'] = spec['timm_name']
    cfg['model']['display_name'] = spec['display_name']
    cfg['model']['family'] = spec['family']
    if cfg['data']['input_size'] in (None, 'auto'):
        cfg['data']['input_size'] = spec['input_size']

    if cfg['data']['split'] not in ('random', 'plant'):
        raise ValueError('data.split must be "random" or "plant"')

    cfg['experiment_name'] = 'line{}_{}_{}'.format(cfg['data']['line'], cfg['model']['name'], cfg['data']['split'])
    return munchify(cfg)


def split_dir_for(cfg):
    return os.path.join(cfg.base.split_dir, cfg.data.split, 'line{}'.format(cfg.data.line))


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', default=os.path.join(REPO_ROOT, 'configs', 'default.yaml'))
    parser.add_argument('--line', type=int, choices=[1, 2, 3, 4], help='cultivation line (dataset)')
    parser.add_argument('--model', choices=['vgg16', 'resnet18', 'vit_b', 'swin_b', 'convnext_b'])
    parser.add_argument('--split', choices=['random', 'plant'], help='random 80:20 or plant-based 70:20:10')
    parser.add_argument('--opts', nargs='*', default=[], help='overrides, e.g. train.epochs=50 solver.learning_rate=1e-5')
    parser.add_argument('-p', '--print_config', action='store_true')
    return parser


def prepare_save_path(cfg, config_path):
    path = os.path.join(cfg.base.save_path, cfg.experiment_name)
    if os.path.exists(path) and not cfg.base.overwrite:
        suffix = 1
        while os.path.exists('{}_{}'.format(path, suffix)):
            suffix += 1
        path = '{}_{}'.format(path, suffix)
    os.makedirs(path, exist_ok=True)
    cfg.base.run_dir = path
    save_yaml(to_dict(cfg), os.path.join(path, 'resolved_config.yaml'))
    shutil.copy(config_path, os.path.join(path, os.path.basename(config_path)))
    return path


def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def print_config(cfg, indent=''):
    for key, value in cfg.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indent, key))
            print_config(value, indent + '    ')
        else:
            print('{}{}: {}'.format(indent, key, value))
