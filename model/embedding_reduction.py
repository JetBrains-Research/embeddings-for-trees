from typing import List

import torch
import torch.nn as nn


class _IReduction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class SumReduction(_IReduction):
    def __init__(self):
        super().__init__()

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        return sum(embeds)


class LinearReduction(_IReduction):
    _activations = {
        'tanh': nn.Tanh,
        'relu': nn.ReLU
    }

    def __init__(self, n_embeds: int, h_emb: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(h_emb * n_embeds, h_emb)
        if activation not in self._activations:
            raise ValueError(f"unknown activation {activation}")
        self.activation = self._activations[activation]()

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        cat_embeds = torch.cat(embeds, dim=1)
        return self.activation(self.linear(cat_embeds))


class ConcatenationReduction(_IReduction):
    def __init__(self):
        super().__init__()

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        print(len(embeds))
        return torch.cat(embeds, dim=1)
