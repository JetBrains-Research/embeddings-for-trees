from typing import Tuple, List

import torch
import torch.nn as nn

from model.attention import _IAttention
from utils.common import segment_sizes_to_slices


class _IAttentionDecoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make decoder step with attention mechanism for given token id and previous states

        :param input_token_id: [batch size]
        :param root_hidden_states: [1, batch size, hidden size]
        :param root_memory_cells: [1, batch size, hidden size]
        :param encoder_output: [number of nodes in batch, hidden size]
        :param tree_sizes: [batch size]
        :return: Tuple[
            output: [batch size, number of classes]
            new hidden state, new memory cell
        ]
        """
        raise NotImplementedError


class LSTMAttentionDecoder(_IAttentionDecoder):
    def __init__(self, h_enc: int, h_dec: int, out_size: int,
                 padding_index: int, attention: _IAttention, dropout_prob: float = 0.):
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.out_size = out_size

        self.embedding = nn.Embedding(self.out_size, self.h_dec, padding_idx=padding_index)
        self.lstm = nn.LSTM(self.h_dec + self.h_enc, self.h_dec)
        self.linear = nn.Linear(self.h_dec, self.out_size)
        self.attention = attention
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [1, batch size, embedding size]
        embedded = self.embedding(input_token_id.unsqueeze(0))

        # [number of nodes in batch, 1]
        attention = self.attention(root_hidden_states.squeeze(0), encoder_output, tree_sizes)

        # [number of nodes in batch, hidden_size]
        weighted_hidden_states = encoder_output * attention

        # [1, batch size, hidden_size]
        attended_hidden_states = torch.cat(
            [torch.sum(weighted_hidden_states[tree_slice], dim=0, keepdim=True)
             for tree_slice in segment_sizes_to_slices(tree_sizes)],
            dim=0
        ).unsqueeze(0)

        # [1, batch size, hidden size + embedding size]
        lstm_input = torch.cat((embedded, attended_hidden_states), dim=2)

        # output: [1, batch size, hidden size]
        output, (new_hidden_states, new_memory_cells) = self.lstm(lstm_input, (root_hidden_states, root_memory_cells))

        # [batch size, number of classes]
        logits = self.linear(output.squeeze(0))

        return logits, new_hidden_states, new_memory_cells
