from typing import Dict, Union, Tuple

import torch
from torch import nn

from model.decoder import ITreeDecoder


class LinearDecoder(ITreeDecoder):

    name = "Linear"

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict, dropout: float = 0.) -> None:
        super().__init__(h_enc, h_dec, label_to_id)
        self.linear = nn.Linear(self.h_enc, self.out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self, encoded_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
            root_indexes: torch.LongTensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # [number of nodes, hidden state]
        if isinstance(encoded_data, tuple):
            node_hidden_states = encoded_data[0]
        else:
            node_hidden_states = encoded_data

        # [batch size, hidden state]
        root_hidden_states = node_hidden_states[root_indexes]
        root_hidden_states = self.dropout(root_hidden_states)

        # [1, batch size, vocab size]
        logits = self.linear(root_hidden_states).unsqueeze(0)

        return logits
