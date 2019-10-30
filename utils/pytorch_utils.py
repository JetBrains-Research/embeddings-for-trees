import torch
import numpy as np


def get_device() -> torch.device:
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def fix_seed(seed: int = 7) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
