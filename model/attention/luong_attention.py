from typing import List

import torch
from torch import nn

from model.attention import ISubtreeAttention
from utils.common import segment_sizes_to_slices


class LuongAttention(ISubtreeAttention):

    name = "Luong"

    def __init__(self, h_enc: int, h_dec: int, score: str = 'concat') -> None:
        super().__init__(h_enc, h_dec)
        self.score = score
        if self.score == 'concat':
            self.linear = nn.Linear(self.h_dec + self.h_enc, self.h_enc, bias=False)
            self.v = nn.Parameter(torch.rand(self.h_enc, 1), requires_grad=True)
        elif self.score == 'general':
            self.linear = nn.Linear(self.h_enc, h_dec, bias=False)
        else:
            raise ValueError(f"Unknown score function: {score}")

    def forward(self, hidden_states: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List) -> torch.Tensor:
        # [number of nodes in batch, h_dec]
        repeated_hidden_states = torch.cat(
            [prev_hidden_state.expand(tree_size, -1)
             for prev_hidden_state, tree_size in zip(hidden_states, tree_sizes)],
            dim=0
        )

        if self.score == 'concat':
            # [number of nodes in batch, h_enc]
            energy = torch.tanh(self.linear(
                torch.cat((repeated_hidden_states, encoder_output), dim=1)
            ))
            # [number of nodes in batch, 1]
            scores = torch.matmul(energy, self.v)
        elif self.score == 'general':
            # [number of nodes in batch; h dec]
            linear_encoder = self.linear(encoder_output)
            # [number of nodes in batch; 1]
            scores = torch.bmm(linear_encoder.unsqueeze(1), repeated_hidden_states.unsqueeze(2)).squeeze(2)
        else:
            raise RuntimeError("Oh, this is strange")

        # [number of nodes in batch, 1]
        attentions = torch.cat(
            [nn.functional.softmax(scores[tree_slice], dim=0)
             for tree_slice in segment_sizes_to_slices(tree_sizes)],
            dim=0
        )

        return attentions
