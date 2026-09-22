"""Warm-up + cosine-annealing learning-rate schedule (Sec. 3.1.2, Table 5).

The learning rate rises linearly during the warm-up phase to the initial rate (3e-6)
and then follows a cosine curve down to the minimum rate (1e-6). The schedule is
stepped once per iteration.
"""
import math

from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_lambda(total_steps, warmup_steps, base_lr, min_lr, warmup_start_factor=0.01):
    min_factor = min_lr / base_lr

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return warmup_start_factor + (1.0 - warmup_start_factor) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def build_scheduler(cfg, optimizer, steps_per_epoch):
    total = cfg.train.epochs * steps_per_epoch
    warmup = cfg.train.warmup_epochs * steps_per_epoch
    fn = warmup_cosine_lambda(total, warmup, cfg.solver.learning_rate, cfg.solver.min_learning_rate,
                              cfg.solver.warmup_start_factor)
    return LambdaLR(optimizer, lr_lambda=fn)


def build_optimizer(cfg, model):
    import torch
    if cfg.solver.optimizer.lower() != 'adamw':
        raise NotImplementedError('The paper uses AdamW (Table 5).')
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.solver.learning_rate, betas=tuple(cfg.solver.betas),
                             weight_decay=cfg.solver.weight_decay)
