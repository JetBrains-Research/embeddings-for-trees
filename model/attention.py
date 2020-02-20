from math import sqrt
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as f

from utils.common import segment_sizes_to_slices


class _IAttention(nn.Module):
    def __init__(self, h_enc: int, h_dec: int) -> None:
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec

    def forward(self, prev_hidden_states: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List)\
            -> torch.Tensor:
        """ Compute attention weights based on previous decoder state and encoder output

        :param prev_hidden_states: [batch size, hidden size]
        :param encoder_output: [number of nodes in batch, hidden size]
        :param tree_sizes: [batch size]
        :return: attention weights [number of nodes in batch, 1]
        """
        raise NotImplementedError


class LuongConcatAttention(_IAttention):
    def __init__(self, h_enc: int, h_dec: int) -> None:
        super().__init__(h_enc, h_dec)
        self.linear = nn.Linear(self.h_dec + self.h_enc, self.h_enc)
        self.v = nn.Parameter(torch.rand(self.h_enc, 1), requires_grad=True)

    def forward(self, prev_hidden_states: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List)\
            -> torch.Tensor:
        # [number of nodes in batch, h_dec]
        repeated_hidden_states = torch.cat(
            [prev_hidden_state.expand(tree_size, -1)
             for prev_hidden_state, tree_size in zip(prev_hidden_states, tree_sizes)],
            dim=0
        )

        # [number of nodes in batch, h_enc]
        energy = torch.tanh(self.linear(
            torch.cat((repeated_hidden_states, encoder_output), dim=1)
        ))

        # [number of nodes in batch, 1]
        scores = torch.matmul(energy, self.v)

        # [number of nodes in batch, 1]
        attentions = torch.cat(
            [nn.functional.softmax(scores[tree_slice], dim=0)
             for tree_slice in segment_sizes_to_slices(tree_sizes)],
            dim=0
        )

        return attentions


def scaled_dot_product_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        mask: torch.Tensor = None, dropout: nn.Dropout = None
) -> torch.Tensor:
    scores = query.matmul(key.transpose(-2, -1)) / sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = f.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, value)
    return output

