from typing import Dict

import torch
import torch.nn as nn
from dgl import DGLGraph

from model.decoder import Decoder
from model.embedding import Embedding
from model.encoder import Encoder


class Tree2Seq(nn.Module):
    def __init__(self, embedding_info: Dict, encoder_info: Dict, decoder_info: Dict,
                 hidden_states: Dict, token_to_id: Dict, type_to_id: Dict, label_to_id: Dict):
        super().__init__()

        self.embedding_info = embedding_info
        self.encoder_info = encoder_info
        self.decoder_info = decoder_info

        self.hidden_states = hidden_states

        self.token_to_id = token_to_id
        self.type_to_id = type_to_id
        self.label_to_id = label_to_id

        self.embedding = Embedding(
            h_emb=self.hidden_states['embedding'], token_to_id=self.token_to_id,
            type_to_id=self.type_to_id, **self.embedding_info)
        self.encoder = Encoder(
            h_emb=self.hidden_states['embedding'], h_enc=self.hidden_states['encoder'],
            **self.encoder_info
        )
        self.decoder = Decoder(
            h_enc=self.hidden_states['encoder'], h_dec=self.hidden_states['decoder'],
            label_to_id=self.label_to_id, **self.decoder_info
        )

    def forward(self, graph: DGLGraph, labels: torch.Tensor) -> torch.Tensor:
        """Predict sequence of tokens for given batched graph

        :param graph: the batched graph
        :param labels: [batch size] string labels of each example
        :return: logits [the longest sequence, batch size, vocab size]
        """
        return self.decoder(
            *self.encoder(self.embedding(graph)), labels
        )

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        """Predict token for each step by given logits

        :param logits: [max length, batch size, number of classes] logits for each position in sequence
        :return: [max length, batch size] token's ids for each position in sequence
        """
        return logits.argmax(dim=-1)

    def get_configuration(self) -> Dict:
        return {
            'embedding_info': self.embedding_info,
            'encoder_info': self.encoder_info,
            'decoder_info': self.decoder_info,
            'hidden_states': self.hidden_states,
            'token_to_id': self.token_to_id,
            'type_to_id': self.type_to_id,
            'label_to_id': self.label_to_id
        }
