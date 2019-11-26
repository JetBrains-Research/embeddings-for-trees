from typing import Tuple

import torch
import torch.nn as nn

from model.attention import _IAttention


class _IAttentionDecoder(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor, encoder_hidden_states: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make decoder step with attention mechanism for given token id and previous states

        :param input_token_id: [batch size]
        :param root_hidden_states: [1, batch size, hidden size]
        :param root_memory_cells: [1, batch size, hidden size]
        :param encoder_hidden_states: [batch size, max number of nodes, hidden size]
        :return: Tuple[
            output: [batch size, number of classes]
            new hidden state, new memory cell
        ]
        """
        raise NotImplementedError


class LSTMAttentionDecoder(_IAttentionDecoder):
    def __init__(self, embedding_size: int, hidden_size: int, out_size: int,
                 padding_index: int, attention: _IAttention, dropout_prob: float = 0.):
        super().__init__()
        # because of concatenating weighted hidden states and embedding of input token
        assert embedding_size == hidden_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.embedding = nn.Embedding(self.out_size, self.embedding_size, padding_idx=padding_index)
        self.lstm = nn.LSTM(self.hidden_size + self.embedding_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
        self.attention = attention
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor, encoder_hidden_states: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [1, batch size, embedding size]
        embedded = self.embedding(input_token_id.unsqueeze(0))

        # [batch size, max number of nodes]
        attention = self.attention(root_hidden_states.squeeze(0), encoder_hidden_states)
        # [batch size, 1, max number of nodes]
        attention = attention.unsqueeze(1)

        # [1, batch size, hidden size]
        weighted_hidden_states = torch.bmm(attention, encoder_hidden_states).permute(1, 0, 2)

        # [1, batch size, hidden size + embedding size]
        lstm_input = torch.cat((embedded, weighted_hidden_states), dim=2)

        # output: [1, batch size, hidden size]
        output, (new_hidden_states, new_memory_cells) = self.lstm(lstm_input, (root_hidden_states, root_memory_cells))

        # [batch size, number of classes]
        logits = self.linear(output.squeeze(0))

        return logits, new_hidden_states, new_memory_cells
