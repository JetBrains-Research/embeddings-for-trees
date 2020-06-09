from typing import Union, Tuple, Dict

import dgl
import torch
from torch import nn

from model.encoder.transformer_encoder import TransformerEncoder
from model.encoder.treelstm import TreeLSTM, DfsLSTM, TwoOrderLSTM


class Encoder(nn.Module):
    """Encode given batched graph."""

    _encoders = {
        TreeLSTM.__name__: TreeLSTM,
        TransformerEncoder.__name__: TransformerEncoder,
        DfsLSTM.__name__: DfsLSTM,
        TwoOrderLSTM.__name__: TwoOrderLSTM
    }

    def __init__(self, h_emb: int, h_enc: int, name: str, params: Dict):
        super().__init__()
        self.h_emb = h_emb
        self.h_enc = h_enc
        self.encoder_name = name

        if self.encoder_name not in self._encoders:
            raise ValueError(f"unknown encoder: {self.encoder_name}")
        self.encoder = self._encoders[self.encoder_name](self.h_emb, self.h_enc, **params)

    def forward(
            self, graph: dgl.DGLGraph, device: torch.device
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Produce new states for each node in given graph"""
        return self.encoder(graph, device)
