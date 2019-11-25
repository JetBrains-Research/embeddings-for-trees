from typing import Dict, Tuple

import torch
import torch.nn as nn


class _IDecoder(nn.Module):
    """Interface for decoder block in Tree2Seq model
    """

    out_size = None

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

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, out_size)), requires_grad=True)

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(root_hidden_states) + self.bias
        return logits, root_hidden_states, root_memory_cells


class LSTMDecoder(_IDecoder):

    def __init__(self, embedding_size: int, hidden_size: int, out_size: int,
                 padding_index: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.embedding = nn.Embedding(self.out_size, self.embedding_size, padding_idx=padding_index)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.out_size)

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

