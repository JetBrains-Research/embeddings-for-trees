import dgl
import torch
from torch import nn


class _IEncoder(nn.Module):
    """Encode given batched graph.
    Forward method takes batched graph and return encoded vector for each node
    """
    def __init__(self, h_emb: int, h_enc: int):
        super().__init__()
        self.h_emb = h_emb
        self.h_enc = h_enc

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> torch.Tensor:
        raise NotImplementedError
