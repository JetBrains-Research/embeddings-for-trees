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
        """Predict next token based on previous state and previous token

        :param input_token_id: [Batch size] -- previous token's ids
        :param root_hidden_states: [Batch size, hidden state] -- hidden states from an encoder or previous steps
        :param root_memory_cells: [Batch size, hidden state] -- memory cells from an encoder or previous steps
        :return: [Batch size, number of classes] -- logits for each token in batch
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
