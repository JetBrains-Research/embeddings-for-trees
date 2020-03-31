from typing import Union, Tuple

import dgl
import torch
from torch import nn


class _IEncoder(nn.Module):
    """Encode given batched graph."""
    def __init__(self, h_emb: int, h_enc: int):
        super().__init__()
        self.h_emb = h_emb
        self.h_enc = h_enc

    def forward(
            self, graph: dgl.DGLGraph, device: torch.device
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Produce new states for each node in given graph"""
        raise NotImplementedError
