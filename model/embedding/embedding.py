from typing import Dict

import dgl
import torch
import torch.nn as nn

from embedding.node_embedding import TokenNodeEmbedding, TypeNodeEmbedding, SubTokenNodeEmbedding
from embedding.positional_embedding import PositionalEmbedding
from model.embedding.reduction import SumReduction, LinearReduction, ConcatenationReduction


class Embedding(nn.Module):
    _embeddings = {
        'token': TokenNodeEmbedding,
        'type': TypeNodeEmbedding,
        'subtoken': SubTokenNodeEmbedding,
        'positional': PositionalEmbedding
    }
    _reductions = {
        'sum': SumReduction,
        'linear': LinearReduction,
        'cat': ConcatenationReduction
    }

    def _init_embedding_layer(self, embedding_name: str, embedding_params: Dict) -> nn.Module:
        if embedding_name not in self._embeddings:
            raise ValueError(f"unknown embedding function: {embedding_name}")
        embedding_module = self._embeddings[embedding_name]
        return embedding_module(
            token_to_id=self.token_to_id, type_to_id=self.type_to_id, h_emb=self.embedding_size, **embedding_params
        )

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, embeddings: Dict, reduction: Dict):
        super().__init__()
        self.token_to_id = token_to_id
        self.type_to_id = type_to_id

        if reduction['name'] not in self._reductions:
            raise ValueError(f"unknown embedding reduction: {reduction['name']}")
        self.reduction = self._reductions[reduction['name']](len(embeddings), h_emb, **reduction["params"])
        self.embedding_size = self.reduction.embedding_size

        self.node_embeddings = nn.ModuleList([
            self._init_embedding_layer(name, params) for name, params in embeddings.items()
        ])

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
        embeds = [embedding(graph, device) for embedding in self.node_embeddings]
        graph.ndata['x'] = self.reduction(embeds)
        return graph
