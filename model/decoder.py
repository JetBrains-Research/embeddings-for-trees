from typing import Dict, Tuple, List

import torch
import torch.nn as nn

from utils.common import PAD, UNK
from utils.token_processing import convert_label_to_sublabels, get_dict_of_subtokens


class _IDecoder(nn.Module):
    """Interface for decoder block in Tree2Seq model
    """

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict) -> None:
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.label_to_id = label_to_id

        if UNK not in self.label_to_id:
            self.label_to_id[UNK] = len(self.label_to_id)

        self.out_size = len(self.label_to_id)
        self.pad_index = self.label_to_id[PAD] if PAD in self.label_to_id else -1

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

    def convert_labels(self, labels: List[str], device: torch.device) -> torch.Tensor:
        """Convert labels with respect to decoder

        :param device: torch device
        :param labels: [batch_size] string names of each label
        :return [batch_size, ...]
        """
        raise NotImplementedError


class LinearDecoder(_IDecoder):

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict) -> None:
        super().__init__(h_enc, h_dec, label_to_id)
        self.h_dec = h_dec
        self.linear = nn.Linear(h_enc, h_dec, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, h_dec)), requires_grad=True)

    def forward(self, input_token_id: torch.Tensor, root_hidden_states: torch.Tensor,
                root_memory_cells: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.linear(root_hidden_states)
        return logits, root_hidden_states, root_memory_cells

    def convert_labels(self, labels: List[str], device: torch.device) -> torch.Tensor:
        return torch.tensor([self.label_to_id[label] for label in labels], device=device)


class LSTMDecoder(_IDecoder):

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict):
        """Convert label to consequence of sublabels and use lstm cell to predict next

        :param h_enc: size of hidden state of encoder, so it's equal to size of hidden state of lstm cell
        :param h_dec: size of lstm input/output, so labels embedding size equal to it
        """
        self.sublabel_to_id = get_dict_of_subtokens(label_to_id)
        super().__init__(h_enc, h_dec, self.sublabel_to_id)

        self.embedding = nn.Embedding(self.out_size, self.h_dec, padding_idx=self.pad_index)
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

    def convert_labels(self, labels: List[str], device: torch.device) -> torch.Tensor:
        return convert_label_to_sublabels(labels, self.sublabel_to_id).to(device)
