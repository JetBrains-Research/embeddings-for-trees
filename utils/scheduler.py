from typing import Dict, Callable

import numpy
import torch


def get_scheduler(params: Dict, optimizer, total_steps: int):
    name = params.get('name', '')
    if name == 'const':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    elif name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=params['step'], gamma=params['gamma']
        )
    elif name in ['vaswani', 'cosine']:
        # LambdaLR multiply original lr to lambda function result
        # If original lr is equal to 1, it's correspond to control it by function
        for g in optimizer.param_groups:
            g['lr'] = 1
        scheduler_lambda = \
            _vaswani_lambda(params['warm_up_cf'], params['d_model'], total_steps) \
            if name == 'vaswani' else \
            _cosine_lambda(params['warm_up_cf'], params['max_lr'], params['min_lr'], total_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)
    else:
        raise ValueError(f"unknown scheduler name: {name}")

    return scheduler


def _vaswani_lambda(warm_up_cf: float, d_model: int, total_steps: int) -> Callable[[int], float]:
    warm_up_steps = int(warm_up_cf * total_steps)
    d_model_cf = numpy.power(d_model, -0.5)
    warm_start_cf = numpy.power(warm_up_steps, -1.5)
    # add 1 for avoiding dividing by zero
    return lambda cur_step: d_model_cf * min(numpy.power(cur_step + 1, -0.5), warm_start_cf * (cur_step + 1))


def _cosine_lambda(warm_up_cf: float, max_lr: float, min_lr: float, total_steps: int) -> Callable[[int], float]:
    warm_up_steps = int(warm_up_cf * total_steps)
    return lambda cur_step: \
        min_lr + (max_lr - min_lr) * cur_step / warm_up_steps \
        if cur_step < warm_up_steps else \
        min_lr + 0.5 * (max_lr - min_lr) * (1 + numpy.cos(
                numpy.pi * (cur_step - warm_up_steps) / (total_steps - warm_up_steps)
            ))
