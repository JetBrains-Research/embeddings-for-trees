from typing import Dict, List

import dgl
import torch
import torch.nn as nn

from model.embedding.node_embedding import INodeEmbedding
from model.embedding.reduction import IReduction


class Embedding(nn.Module):

    _known_node_embeddings = {}
    _known_reductions = {}

    def _init_embedding_layer(self, embedding_name: str, embedding_params: Dict) -> nn.Module:
        if embedding_name not in self._known_node_embeddings:
            raise ValueError(f"unknown embedding function: {embedding_name}")
        embedding_module = self._known_node_embeddings[embedding_name]
        return embedding_module(
            token_to_id=self.token_to_id, type_to_id=self.type_to_id, h_emb=self.embedding_size, **embedding_params
        )

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, embeddings: Dict, reduction: Dict):
        super().__init__()
        self.token_to_id = token_to_id
        self.type_to_id = type_to_id

        if reduction['name'] not in self._known_reductions:
            raise ValueError(f"unknown embedding reduction: {reduction['name']}")
        self.reduction = self._known_reductions[reduction['name']](len(embeddings), h_emb, **reduction["params"])
        self.embedding_size = self.reduction.embedding_size

        self.node_embeddings = nn.ModuleList([
            self._init_embedding_layer(name, params) for name, params in embeddings.items()
        ])

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
        embeds = [embedding(graph, device) for embedding in self.node_embeddings]
        graph.ndata['x'] = self.reduction(embeds)
        return graph

    @staticmethod
    def register_node_embedding(node_embedding: INodeEmbedding):
        if not issubclass(node_embedding, INodeEmbedding):
            raise ValueError(f"Attempt to register not a Node Embedding class "
                             f"({node_embedding.__name__} not a subclass of {INodeEmbedding.__name__})")
        Embedding._known_node_embeddings[node_embedding.name] = node_embedding

    @staticmethod
    def register_reduction(reduction: IReduction):
        if not issubclass(reduction, IReduction):
            raise ValueError(f"Attempt to register not a Reduction class "
                             f"({reduction.__name__} not a subclass of {IReduction.__name__})")
        Embedding._known_reductions[reduction.name] = reduction

    @staticmethod
    def get_known_node_embeddings() -> List[str]:
        return list(Embedding._known_node_embeddings.keys())

    @staticmethod
    def get_known_reductions() -> List[str]:
        return list(Embedding._known_reductions.keys())
