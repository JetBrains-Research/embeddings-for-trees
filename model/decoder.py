from typing import Dict, Tuple

import torch
import torch.nn as nn


class _IDecoder(nn.Module):
    """Interface for decoder block in Tree2Seq model
    """

    h_dec = None

    def __init__(self):
        super().__init__()

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make decoder step for given token id and previous states

        :param input_token_id: [batch size]
        :param root_hidden_states: [1, batch size, hidden state]
        :param root_memory_cells: [1, batch size, hidden state]
        :return: Tuple[
            output: [batch size, number of classes]
            new hidden state, new memory cell
        ]
        """
        raise NotImplementedError


class LinearDecoder(_IDecoder):

    def __init__(self, h_enc: int, h_dec: int) -> None:
        super().__init__()
        self.h_dec = h_dec
        self.linear = nn.Linear(h_enc, h_dec, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, h_dec)), requires_grad=True)

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(root_hidden_states) + self.bias
        return logits, root_hidden_states, root_memory_cells


class LSTMDecoder(_IDecoder):

    def __init__(self, h_enc: int, h_dec: int, out_size: int,
                 padding_index: int):
        """

        :param h_enc: size of hidden state of encoder, so it's equal to size of hidden state of lstm cell
        :param h_dec: size of lstm input/output, so labels embedding size equal to it
        :param out_size: size of label vocabulary
        :param padding_index:
        """
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.out_size = out_size

        self.embedding = nn.Embedding(self.out_size, self.h_dec, padding_idx=padding_index)
        self.lstm = nn.LSTM(self.h_dec, self.h_enc)
        self.linear = nn.Linear(self.h_dec, self.out_size)

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [1, batch size, embedding size]
        embedded = self.embedding(
            input_token_id.unsqueeze(0)
        )

        output, (root_hidden_states, root_memory_cells) = self.lstm(embedded, (root_hidden_states, root_memory_cells))

        # [batch size, number of classes]
        logits = self.linear(output.squeeze(0))

        return logits, root_hidden_states, root_memory_cells

