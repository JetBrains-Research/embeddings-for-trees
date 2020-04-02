from typing import Dict, Callable

import numpy
import torch


def get_scheduler(scheduler_info: Dict, optimizer, total_steps: int):
    name = scheduler_info.get('name', '')
    if name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_info['step_size'], gamma=scheduler_info['step_gamma']
        )
    # LambdaLR multiply lr by function result
    # setting initial lr to 1.0 correspond to regularize lr throw lambda
    elif name == 'vaswani':
        for g in optimizer.param_groups:
            g['lr'] = 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_vaswani_lambda(scheduler_info['warm_up_cf'], scheduler_info['d_model'], total_steps)
        )
    elif name == 'cosine_warm_up':
        for g in optimizer.param_groups:
            g['lr'] = 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_cosine_warm_up_lambda(
                scheduler_info['warm_up_cf'], scheduler_info['max_lr'], scheduler_info['min_lr'], total_steps
            )
        )
    else:
        raise ValueError(f"unknown scheduler name: {name}")

    return scheduler


def _vaswani_lambda(warm_up_cf: float, d_model: int, total_steps: int) -> Callable[[int], float]:
    warm_up_steps = int(warm_up_cf * total_steps)
    d_model_cf = numpy.power(d_model, -0.5)
    warm_start_cf = numpy.power(warm_up_steps, -1.5)
    # add 1 for avoiding dividing by zero
    return lambda cur_step: d_model_cf * min(numpy.power(cur_step + 1, -0.5), warm_start_cf * (cur_step + 1))


def _cosine_warm_up_lambda(warm_up_cf: float, max_lr: float, min_lr: float, total_steps: int) -> Callable[[int], float]:
    warm_up_steps = int(warm_up_cf * total_steps)
    return lambda cur_step: \
        max_lr * cur_step / warm_up_steps \
        if cur_step < warm_up_steps else \
        min_lr + 0.5 * (max_lr - min_lr) * (1 + numpy.cos(
                numpy.pi * (cur_step - warm_up_steps) / (total_steps - warm_up_steps)
            ))
