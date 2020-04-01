from typing import Dict, Callable

import numpy
import torch


def get_scheduler(scheduler_info: Dict, optimizer):
    name = scheduler_info.get('name', '')
    if name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_info['step_size'], gamma=scheduler_info['step_gamma']
        )
    elif name == 'vaswani':
        warm_start = scheduler_info['warm_start_step']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=_vaswani_lambda(warm_start, scheduler_info['d_model'])
        )
    else:
        raise ValueError(f"unknown scheduler name: {name}")

    return scheduler


def _vaswani_lambda(warm_start: int, d_model: int) -> Callable[[int], float]:
    d_model_cf = numpy.power(d_model, -0.5)
    warm_start_cf = numpy.power(warm_start, -1.5)
    return lambda cur_step: d_model_cf * min(numpy.power(cur_step, -0.5), warm_start_cf * cur_step)
