from typing import Dict, Tuple, Optional

import dgl
import torch
from commode_utils.modules import LuongAttention
from commode_utils.training import cut_into_segments
from omegaconf import DictConfig
from torch import nn, Tensor
from torch_scatter import scatter

from embeddings_for_trees.data.vocabulary import Vocabulary
from embeddings_for_trees.utils.common import TOKEN


class PointerDecoder(nn.Module):
    LSTM_STATE = Tuple[Tensor, Tensor]

    _threshold_cf = 1.0

    def __init__(
        self,
        config: DictConfig,
        token_to_id: Dict[str, int],
        token_embeddings: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self._n_tokens = len(token_to_id)
        self._sos_token = token_to_id[Vocabulary.SOS]
        self._pad_token = token_to_id[Vocabulary.PAD]
        self._eos_token = token_to_id[Vocabulary.EOS]
        self._token_embeddings = token_embeddings or nn.Embedding(
            self._n_tokens, config.embedding_size, padding_idx=self._pad_token
        )
        self._attention = LuongAttention(config.decoder_size)

        self._decoder_num_layers = config.decoder_num_layers
        self._decoder_lstm = nn.LSTM(
            config.embedding_size,
            config.decoder_size,
            num_layers=config.decoder_num_layers,
            dropout=config.rnn_dropout if config.decoder_num_layers > 1 else 0,
            batch_first=True,
        )
        self._dropout = nn.Dropout(config.rnn_dropout)

        ignore__idx = [
            token_to_id[it] for it in ["EMPTY", "<E>", "<UNK>", "METHOD_NAME", "<STR>", "<INT>"] if it in token_to_id
        ]
        self._ignore_pointer_idx = nn.Parameter(torch.tensor(ignore__idx), requires_grad=False)

        self._id_to_token = {v: k for k, v in token_to_id.items()}

    def _init_lstm_state(self, encoder_output: Tensor, attention_mask: Tensor) -> LSTM_STATE:
        lstm_initial_state: torch.Tensor = encoder_output.sum(dim=1)  # [batch size; encoder size]
        segment_sizes: torch.Tensor = (attention_mask == 0).sum(dim=1, keepdim=True)  # [batch size; 1]
        lstm_initial_state /= segment_sizes  # [batch size; encoder size]
        lstm_initial_state = lstm_initial_state.unsqueeze(0).repeat(self._decoder_num_layers, 1, 1)
        return lstm_initial_state, lstm_initial_state

    def _single_step(
        self, input_token: Tensor, encoder_output: Tensor, attention_mask: Tensor, decoder_state: LSTM_STATE
    ) -> Tuple[torch.Tensor, LSTM_STATE]:
        h_prev, c_prev = decoder_state

        # [batch size; 1; embedding size]
        embedded = self._token_embeddings(input_token).unsqueeze(1)

        # hidden -- [n layers; batch size; decoder size]
        # output -- [batch size; 1; decoder size]
        output, (h_new, c_new) = self._decoder_lstm(embedded, (h_prev, c_prev))

        # [batch size; decoder size]
        output = self._dropout(output).squeeze(1)

        # [batch size; max nodes]
        encoder_tokens_probability = self._attention(output, encoder_output, attention_mask)
        return encoder_tokens_probability, (h_new, c_new)

    def forward(
        self,
        batched_trees: dgl.DGLGraph,
        encoder_states: Tensor,
        output_len: int,
        target_sequence: Optional[Tensor] = None,
    ):
        trees = dgl.unbatch(batched_trees)
        # encoder output -- [batch size; max nodes; encoder dim]
        # attention mask -- [batch size; max nodes]
        batched_states, mask = cut_into_segments(encoder_states, batched_trees.batch_num_nodes(), -1e9)

        h, c = self._init_lstm_state(batched_states, mask)

        # Nodes with 0 in-degree are leaves (since topological sort traverse from leaves to roots).
        batch_data = []
        for i, tree in enumerate(trees):
            leaves = tree.in_degrees() == 0
            token_ids = tree.ndata[TOKEN][:, 0]
            bad_ids = torch.isin(token_ids, self._ignore_pointer_idx)

            pointer_mask = leaves & (~bad_ids)
            mask[i, : leaves.shape[0]][~pointer_mask] = -1e9

            # print([self._id_to_token[it.item()] for it in tree.ndata[TOKEN][pointer_mask, 0]])

            batch_data.append((i, tree, pointer_mask))

        # [output len; batch size; n tokens]
        output = encoder_states.new_zeros((output_len, len(trees), self._n_tokens))
        output[0, :, self._sos_token] = 1

        # [batch size]
        current_input = encoder_states.new_full((len(trees),), self._sos_token, dtype=torch.long)

        for step in range(1, output_len):
            # [batch size; max nodes]
            current_pointer, (h, c) = self._single_step(current_input, batched_states, mask, (h, c))

            for i, tree, pointer_mask in batch_data:
                token_ids = tree.ndata[TOKEN][pointer_mask, 0]
                probabilities = current_pointer[i, : pointer_mask.shape[0]][pointer_mask]

                probability_threshold = 1.0 / (self._threshold_cf * pointer_mask.sum())
                probabilities[probabilities < probability_threshold] = 0

                scatter(probabilities, token_ids, reduce="sum", out=output[step, i])

                output[step, i, self._eos_token] = 1 - probabilities.sum()

            if target_sequence is not None:
                current_input = target_sequence[step]
            else:
                current_input = output[step].argmax(dim=-1)
        return output
