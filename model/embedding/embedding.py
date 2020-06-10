from typing import Dict, List

import dgl
import torch
import torch.nn as nn

from utils.common import UNK, PAD


class INodeEmbedding(nn.Module):
    """Interface of embedding module.
    Forward method takes batched graph and applies embedding to its features.
    """

    name = "Node embedding interface"

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int) -> None:
        super().__init__()
        self.token_to_id = token_to_id
        self.type_to_id = type_to_id
        self.h_emb = h_emb

        if UNK not in self.token_to_id:
            self.token_to_id[UNK] = len(token_to_id)
        if UNK not in self.type_to_id:
            self.type_to_id[UNK] = len(type_to_id)

        self.token_vocab_size = len(self.token_to_id)
        self.type_vocab_size = len(self.type_to_id)
        self.token_pad_index = self.token_to_id[PAD] if PAD in self.token_to_id else -1
        self.type_pad_index = self.type_to_id[PAD] if PAD in self.type_to_id else -1

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        raise NotImplementedError


class IReduction(nn.Module):

    name = "reduction interface"

    def __init__(self, n_embeds: int, h_emb: int):
        super().__init__()
        self.n_embeds = n_embeds
        self.h_emb = h_emb

    @property
    def embedding_size(self) -> int:
        return self.h_emb

    def forward(self, embeds: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class Embedding(nn.Module):

    _known_node_embeddings = {}
    _known_reductions = {}

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, embeddings: Dict, reduction: Dict):
        super().__init__()
        self.token_to_id = token_to_id
        self.type_to_id = type_to_id

        if reduction['name'] not in self._known_reductions:
            raise ValueError(f"unknown embedding reduction: {reduction['name']}")
        self.reduction = self._known_reductions[reduction['name']](
            len(embeddings), h_emb, **reduction["params"]
        )
        self.embedding_size = self.reduction.embedding_size

        self.node_embeddings = nn.ModuleList([
            self._init_embedding_layer(name, params) for name, params in embeddings.items()
        ])

    def forward(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        embeds = [embedding(graph) for embedding in self.node_embeddings]
        graph.ndata['x'] = self.reduction(embeds)
        return graph

    def _init_embedding_layer(self, embedding_name: str, embedding_params: Dict) -> nn.Module:
        if embedding_name not in self._known_node_embeddings:
            raise ValueError(f"unknown embedding function: {embedding_name}")
        embedding_module = self._known_node_embeddings[embedding_name]
        return embedding_module(
            token_to_id=self.token_to_id, type_to_id=self.type_to_id, h_emb=self.embedding_size, **embedding_params
        )

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
