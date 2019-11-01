import dgl
import torch
from torch import nn


class _IEncoder(nn.Module):
    """Encode given batched graph.
    Forward method takes batched graph and return encoded vector for each node
    """
    def __init__(self):
        super().__init__()

    def forward(self, graph: dgl.BatchedDGLGraph) -> torch.Tensor:
        pass
