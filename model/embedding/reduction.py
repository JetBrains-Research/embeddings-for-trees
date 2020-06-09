from typing import List

import torch
import torch.nn as nn

from embedding import IReduction


class SumReduction(IReduction):

    name = "sum"

    def __init__(self, n_embeds: int, h_emb: int) -> int:
        super().__init__(n_embeds, h_emb)

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        return sum(embeds)


class LinearReduction(IReduction):

    name = "linear"

    _activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU
    }

    def __init__(self, n_embeds: int, h_emb: int, activation: str):
        super().__init__(n_embeds, h_emb)
        self.linear = nn.Linear(h_emb * n_embeds, h_emb)
        if activation not in self._activations:
            raise ValueError(f"unknown activation {activation}")
        self.activation = self._activations[activation]()

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        cat_embeds = torch.cat(embeds, dim=1)
        return self.activation(self.linear(cat_embeds))


class ConcatenationReduction(IReduction):

    name = "concat"

    def __init__(self, n_embeds: int, h_emb: int):
        super().__init__(n_embeds, h_emb)
        if h_emb % n_embeds != 0:
            raise ValueError(f"h_emb must be divided into {n_embeds} equal parts ({h_emb} % {n_embeds} != 0)")

    @property
    def embedding_size(self) -> int:
        return self.h_emb // self.n_embed

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(embeds, dim=1)
