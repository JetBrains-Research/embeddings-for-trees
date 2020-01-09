from typing import Dict

import torch
import torch.nn as nn
from dgl import BatchedDGLGraph


class _IEmbedding(nn.Module):
    """Interface of embedding module.
    Forward method takes batched graph and applies embedding to its features.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        return graph


class FullTokenEmbedding(_IEmbedding):
    def __init__(self, token_vocab_size: int, out_size: int, padding_index: int, **kwargs) -> None:
        super().__init__()
        self.padding_index = padding_index
        self.token_embedding = nn.Embedding(token_vocab_size, out_size, padding_idx=self.padding_index)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph.ndata['token_embeds'] = self.token_embedding(graph.ndata['token_id'])
        return graph


class SubTokenEmbedding(_IEmbedding):
    def __init__(self, token_vocab_size: int, out_size: int,
                 token_to_subtoken: Dict, token_padding_index: int) -> None:
        super().__init__()
        self.vocab_size = token_vocab_size
        self.out_size = out_size
        self.token_to_subtoken = token_to_subtoken
        self.padding_index = token_padding_index
        self.subtoken_embedding = nn.Embedding(self.vocab_size, self.out_size, padding_idx=self.padding_index)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        number_of_nodes = graph.number_of_nodes()
        subtoken_lengths = torch.tensor([
            self.token_to_subtoken[token.item()].shape[0] for token in graph.ndata['token_id']
        ])
        max_subtoken_length = subtoken_lengths.max()

        subtokens = torch.full(
            (number_of_nodes, max_subtoken_length.item()),
            self.padding_index, dtype=torch.long, device=device
        )
        for node in range(number_of_nodes):
            token_id = graph.nodes[node].data['token_id'].item()
            subtokens[node, :subtoken_lengths[node]] = self.token_to_subtoken[token_id]
        graph.ndata['token_embeds'] = torch.sum(
            self.subtoken_embedding(subtokens), dim=1
        )
        return graph


class SubTokenTypeEmbedding(SubTokenEmbedding):

    def __init__(self,
                 token_vocab_size: int, out_size: int, token_to_subtoken: Dict, token_padding_index: int,
                 type_vocab_size: int, type_padding_index: int) -> None:
        super().__init__(token_vocab_size, out_size, token_to_subtoken, token_padding_index)
        self.type_vocab_size = type_vocab_size
        self.type_padding_index = type_padding_index
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.out_size,
                                           padding_idx=self.type_padding_index)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph = super().forward(graph, device)
        graph.ndata['type_embeds'] = self.type_embedding(graph.ndata['type_id'])
        return graph
