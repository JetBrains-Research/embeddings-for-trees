from typing import Dict

import torch.nn as nn
from dgl import BatchedDGLGraph
from torch import Tensor, device, LongTensor

from model.decoder import _IDecoder, LinearDecoder
from model.embedding import _IEmbedding, TokenEmbedding
from model.encoder import _IEncoder
from model.treelstm import TreeLSTM


class Model(nn.Module):
    def __init__(self, embedding: _IEmbedding, encoder: _IEncoder, decoder: _IDecoder) -> None:
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, graph: BatchedDGLGraph, root_indexes: LongTensor) -> Tensor:
        embedded_graph = self.embedding(graph)
        node_hidden_states = self.encoder(embedded_graph)
        root_hidden_states = node_hidden_states[root_indexes]
        logits = self.decoder(root_hidden_states)
        return logits

    def predict(self, batch: Tensor) -> Tensor:
        return self.decoder.predict(batch)


class ModelFactory:

    _embeddings = {
        'TokenEmbedding': TokenEmbedding
    }
    _encoders = {
        'TreeLSTM': TreeLSTM
    }
    _decoders = {
        'LinearDecoder': LinearDecoder
    }

    def __init__(self, embedding_info: Dict, encoder_info: Dict, decoder_info: Dict, using_device: device):
        self.embedding_info = embedding_info
        self.encoder_info = encoder_info
        self.decoder_info = decoder_info
        self.device = using_device

        self.embedding = self._get_module(self.embedding_info['name'], self._embeddings)
        self.encoder = self._get_module(self.encoder_info['name'], self._encoders)
        self.decoder = self._get_module(self.decoder_info['name'], self._decoders)

    def _get_module(self, module_name: str, modules_dict: Dict) -> nn.Module:
        if module_name not in modules_dict:
            raise ModuleNotFoundError(f"Unknown module {module_name}, try one of {', '.join(modules_dict.keys())}")
        return modules_dict[module_name]

    def construct_model(self) -> Model:
        return Model(
            self.embedding(**self.embedding_info['params'], using_device=self.device),
            self.encoder(**self.encoder_info['params'], using_device=self.device),
            self.decoder(**self.decoder_info['params']),
        ).to(self.device)
