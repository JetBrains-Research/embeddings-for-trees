from typing import List

import torch
import torch.nn as nn

from utils.common import segment_sizes_to_slices


class _IAttention(nn.Module):
    def __init__(self):
        super().__init__()

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
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
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
