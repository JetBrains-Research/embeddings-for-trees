from typing import Union, Tuple

import torch
from omegaconf import DictConfig
from torch import nn

from models.parts.attention import LuongAttention
from utils.training import cut_encoded_data
from utils.common import PAD, SOS
from utils.vocabulary import Vocabulary


class LSTMDecoder(nn.Module):

    _negative_value = -1e9

    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self._out_size = len(vocabulary.label_to_id)
        self._sos_token = vocabulary.label_to_id[SOS]
        self._decoder_num_layers = config.decoder_num_layers
        self._teacher_forcing = config.teacher_forcing

        self._target_embedding = nn.Embedding(
            len(vocabulary.label_to_id), config.embedding_size, padding_idx=vocabulary.label_to_id[PAD]
        )

        self._attention = LuongAttention(config.decoder_size)

        self._decoder_lstm = nn.LSTM(
            config.embedding_size,
            config.decoder_size,
            num_layers=config.decoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )
        self._dropout_rnn = nn.Dropout(config.rnn_dropout)

        self._concat_layer = nn.Linear(config.decoder_size * 2, config.decoder_size, bias=False)
        self._norm = nn.LayerNorm(config.decoder_size)
        self._projection_layer = nn.Linear(config.decoder_size, self._out_size, bias=False)

    def forward(
        self,
        encoded_trees: torch.Tensor,
        tree_sizes: torch.LongTensor,
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = tree_sizes.shape[0]
        # [batch size; max tree size; decoder size], [batch size; max tree size]
        batched_encoded_trees, attention_mask = cut_encoded_data(encoded_trees, tree_sizes, self._negative_value)

        # [n layers; batch size; decoder size]
        initial_state = (
            torch.cat([ctx_batch.mean(0).unsqueeze(0) for ctx_batch in encoded_trees.split(tree_sizes.tolist())])
            .unsqueeze(0)
            .repeat(self._decoder_num_layers, 1, 1)
        )
        h_prev, c_prev = initial_state, initial_state

        # [target len; batch size; vocab size]
        output = encoded_trees.new_zeros((output_length, batch_size, self._out_size))
        # [batch size]
        current_input = encoded_trees.new_full((batch_size,), self._sos_token, dtype=torch.long)
        for step in range(output_length):
            current_output, (h_prev, c_prev) = self.decoder_step(
                current_input, h_prev, c_prev, batched_encoded_trees, attention_mask
            )
            output[step] = current_output
            if target_sequence is not None and torch.rand(1) < self._teacher_forcing:
                current_input = target_sequence[step]
            else:
                current_input = output[step].argmax(dim=-1)

        return output

    def decoder_step(
        self,
        input_tokens: torch.Tensor,  # [batch size]
        h_prev: torch.Tensor,  # [n layers; batch size; decoder size]
        c_prev: torch.Tensor,  # [n layers; batch size; decoder size]
        batched_context: torch.Tensor,  # [batch size; context size; decoder size]
        attention_mask: torch.Tensor,  # [batch size; context size]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # [batch size; 1; embedding size]
        embedded = self._target_embedding(input_tokens).unsqueeze(1)

        # hidden -- [n layers; batch size; decoder size]
        # output -- [batch size; 1; decoder size]
        rnn_output, (h_prev, c_prev) = self._decoder_lstm(embedded, (h_prev, c_prev))
        rnn_output = self._dropout_rnn(rnn_output)

        # [batch size; context size]
        attn_weights = self._attention(h_prev[-1], batched_context, attention_mask)

        # [batch size; 1; decoder size]
        context = torch.bmm(attn_weights.unsqueeze(1), batched_context)

        # [batch size; 2 * decoder size]
        concat_input = torch.cat([rnn_output, context], dim=2).squeeze(1)

        # [batch size; decoder size]
        concat = self._concat_layer(concat_input)
        concat = self._norm(concat)
        concat = torch.tanh(concat)

        # [batch size; vocab size]
        output = self._projection_layer(concat)

        return output, (h_prev, c_prev)
