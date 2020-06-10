from typing import Union, Tuple, Dict, List

import dgl
import torch
from torch import nn


class ITreeEncoder(nn.Module):

    name = "Tree encoder interface"

    def __init__(self, h_emb: int, h_enc: int):
        super().__init__()
        self.h_emb = h_emb
        self.h_enc = h_enc

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        raise NotImplementedError


class Encoder(nn.Module):
    """Encode given batched graph."""

    _known_tree_encoders = {}

    def __init__(self, h_emb: int, h_enc: int, name: str, params: Dict):
        super().__init__()
        self.h_emb = h_emb
        self.h_enc = h_enc
        self.encoder_name = name

        if self.encoder_name not in self._known_tree_encoders:
            raise ValueError(f"unknown encoder: {self.encoder_name}")
        self.encoder = self._known_tree_encoders[self.encoder_name](self.h_emb, self.h_enc, **params)

    def forward(self, graph: dgl.DGLGraph) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Produce new states for each node in given graph"""
        return self.encoder(graph)

    @staticmethod
    def register_tree_encoder(tree_encoder: ITreeEncoder):
        if not issubclass(tree_encoder, ITreeEncoder):
            raise ValueError(f"Attempt to register not a Tree Encoder class "
                             f"({tree_encoder.__name__} not a subclass of {ITreeEncoder.__name__})")
        Encoder._known_tree_encoders[tree_encoder.name] = tree_encoder

    @staticmethod
    def get_known_tree_encoders() -> List[str]:
        return list(Encoder._known_tree_encoders.keys())
