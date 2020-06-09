import dgl
import torch
from torch import nn


class ITreeEncoder(nn.Module):

    name = "Tree encoder interface"

    def __init__(self, h_emb: int, h_enc: int):
        super().__init__()
        self.h_emb = h_emb
        self.h_enc = h_enc

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
        raise NotImplementedError
