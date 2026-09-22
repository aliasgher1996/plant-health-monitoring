import os
import random

import numpy as np


def set_random_seed(seed, deterministic=False):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def save_weights(model, path):
    import torch
    module = model.module if hasattr(model, 'module') else model
    torch.save(module.state_dict(), path)


def print_msg(msg, appendixs=(), warning=False):
    color, end = '\033[93m', '\033[0m'
    fn = (lambda x: print(color + x + end)) if warning else print
    width = min(max(len(m) for m in (msg, *appendixs)), 100)
    fn('=' * width)
    fn(msg)
    for a in appendixs:
        fn(a)
    fn('=' * width)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
