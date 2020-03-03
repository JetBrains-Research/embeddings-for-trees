from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn

from model.attention import get_attention
from utils.common import PAD, UNK, segment_sizes_to_slices
from utils.token_processing import get_dict_of_subtokens


class _IDecoder(nn.Module):
    """Decode sequence given hidden states of nodes"""

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict) -> None:
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.label_to_id = label_to_id

        if UNK not in self.label_to_id:
            self.label_to_id[UNK] = len(self.label_to_id)

        self.out_size = len(self.label_to_id)
        self.pad_index = self.label_to_id[PAD] if PAD in self.label_to_id else -1

    def forward(
            self, encoded_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]], labels: List[str],
            root_indexes: torch.LongTensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode given encoded vectors of nodes

        :param encoded_data: tensor or tuple of tensors with encoded data
        :param labels: list of string labels
        :param root_indexes: indexes of roots in encoded data
        :param device: torch device object
        :return: Tuple[
          logits [sequence len, batch size, labels vocab size],
          ground truth [sequence len, batch size]
        ]
        """
        raise NotImplementedError


class LinearDecoder(_IDecoder):

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict) -> None:
        super().__init__(h_enc, h_dec, label_to_id)
        self.linear = nn.Linear(h_enc, h_dec)

    def forward(
            self, encoded_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]], labels: List[str],
            root_indexes: torch.LongTensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [number of nodes, hidden state]
        if isinstance(encoded_data, tuple):
            node_hidden_states = encoded_data[0]
        else:
            node_hidden_states = encoded_data

        # [batch size, hidden state]
        root_hidden_states = node_hidden_states[root_indexes]

        # [1, batch size, vocab size]
        logits = self.linear(root_hidden_states).unsqueeze(0)

        # [1, batch size]
        ground_truth = torch.tensor([self.label_to_id[l] for l in labels], device=device).view(1, -1)
        return logits, ground_truth


class LSTMDecoder(_IDecoder):

    def __init__(
            self, h_enc: int, h_dec: int, label_to_id: Dict, dropout: float = 0.,
            teacher_force: float = 0., attention: Dict = None
    ):
        """Convert label to consequence of sublabels and use lstm cell to predict next

        :param h_enc: encoder hidden state, correspond to hidden state of LSTM cell
        :param h_dec: size of LSTM cell input/output
        :param label_to_id: dict for converting labels to ids
        :param dropout: probability of dropout
        :param teacher_force: probability of teacher forcing, 0 corresponds to always use previous predicted value
        :param attention: if passed, init attention with given args
        """
        self.sublabel_to_id, self.label_to_sublabels = get_dict_of_subtokens(label_to_id, add_sos_eos=True)
        super().__init__(h_enc, h_dec, self.sublabel_to_id)
        self.teacher_force = teacher_force

        self.embedding = nn.Embedding(self.out_size, self.h_dec, padding_idx=self.pad_index)
        self.linear = nn.Linear(self.h_enc, self.out_size)
        self.dropout = nn.Dropout(dropout)

        if attention is not None:
            self.use_attention = True
            attention_class = get_attention(attention['name'])
            self.attention = attention_class(h_enc=self.h_enc, h_dec=self.h_dec, **attention['params'])
        else:
            self.use_attention = False

        lstm_input_size = self.h_enc + self.h_dec if self.use_attention else self.h_dec
        self.lstm_cell = nn.LSTMCell(input_size=lstm_input_size, hidden_size=self.h_enc)

    def forward(
            self, encoded_data: Tuple[torch.Tensor, torch.Tensor], labels: List[str],
            root_indexes: torch.LongTensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(encoded_data) == 2, f"For LSTM decoder, encoder should produce hidden and memory states"
        # [number of nodes, encoder hidden state]
        node_hidden_states, node_memory_states = encoded_data

        # [batch size, encoder hidden state]
        root_hidden_states = node_hidden_states[root_indexes]
        root_memory_states = node_memory_states[root_indexes]

        sublabels = [self.label_to_sublabels[label] for label in labels]
        sublabels_len = [len(sl) for sl in sublabels]
        max_length = max(sublabels_len)
        batch_size = len(labels)
        # [the longest sequence, batch size]
        ground_truth = torch.full((max_length, batch_size), self.label_to_id[PAD], dtype=torch.long, device=device)
        for i, (sl, sl_len) in enumerate(zip(sublabels, sublabels_len)):
            ground_truth[:sl_len, i] = torch.tensor(sl, dtype=torch.long, device=device)

        # [the longest sequence, batch size, vocab size]
        outputs = torch.zeros(max_length, batch_size, self.out_size, device=device)

        tree_sizes = [(root_indexes[i] - root_indexes[i - 1]).item() for i in range(1, batch_size)]
        tree_sizes.append(node_hidden_states.shape[0] - root_indexes[-1].item())

        # ground_truth[0] correspond to batch of <SOS> tokens
        # [batch size]
        current_input = ground_truth[0]
        for step in range(1, max_length):
            # [batch size, decoder hidden state]
            embedded = self.embedding(current_input)

            if self.use_attention:
                # [number of nodes]
                attention = self.attention(root_hidden_states, node_hidden_states, tree_sizes)

                # [number of nodes, encoder hidden size]
                weighted_hidden_states = node_hidden_states * attention

                # [batch size, encoder hidden size]
                attended_hidden_states = torch.cat(
                    [torch.sum(weighted_hidden_states[tree_slice], dim=0, keepdim=True)
                     for tree_slice in segment_sizes_to_slices(tree_sizes)],
                    dim=0
                )

                # [batch size, decoder hidden size + encoder hidden size]
                lstm_cell_input = torch.cat((embedded, attended_hidden_states), dim=1)
            else:
                # [batch size, decoder hidden size]
                lstm_cell_input = embedded

            lstm_cell_input = self.dropout(lstm_cell_input)

            # [batch size, encoder hidden state]
            root_hidden_states, root_memory_states = \
                self.lstm_cell(lstm_cell_input, (root_hidden_states, root_memory_states))

            # [batch size, vocab size]
            current_output = self.linear(root_hidden_states)
            outputs[step] = current_output

            if self.training:
                current_input = \
                    ground_truth[step] if torch.rand(1) < self.teacher_force else current_output.argmax(dim=-1)
            else:
                current_input = current_output.argmax(dim=-1)

        return outputs, ground_truth
