from typing import Dict

import torch
import torch.nn as nn
from dgl import BatchedDGLGraph

from utils.common import split_tokens_to_subtokens, UNK, PAD, NAN, METHOD_NAME


class _IEmbedding(nn.Module):
    """Interface of embedding module.
    Forward method takes batched graph and applies embedding to its features.
    """

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, device: torch.device) -> None:
        super().__init__()
        self.token_to_id = token_to_id
        self.type_to_id = type_to_id
        self.h_emb = h_emb
        self.device = device

        self.token_vocab_size = len(self.token_to_id)
        self.type_vocab_size = len(self.type_to_id)
        self.token_pad_index = self.token_to_id[PAD] if PAD in self.token_to_id else -1
        self.type_pad_index = self.type_to_id[PAD] if PAD in self.type_to_id else -1

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        return graph


class FullTokenEmbedding(_IEmbedding):
    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, device: torch.device) -> None:
        if UNK not in token_to_id:
            token_to_id[UNK] = len(token_to_id)
        super().__init__(token_to_id, type_to_id, h_emb, device)
        self.token_embedding = nn.Embedding(self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index)

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        graph.ndata['token_embeds'] = self.token_embedding(graph.ndata['token_id'])
        return graph


class SubTokenEmbedding(_IEmbedding):
    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, device: torch.device) -> None:
        self.subtoken_to_id, self.token_to_subtoken = split_tokens_to_subtokens(
            token_to_id, required_tokens=[UNK, PAD, METHOD_NAME, NAN], return_ids=True, device=device
        )
        super().__init__(self.subtoken_to_id, type_to_id, h_emb, device)
        self.subtoken_embedding = nn.Embedding(
            self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index
        )

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        number_of_nodes = graph.number_of_nodes()
        subtoken_lengths = torch.tensor([
            self.token_to_subtoken[token.item()].shape[0] for token in graph.ndata['token_id']
        ])
        max_subtoken_length = subtoken_lengths.max()

        subtokens = torch.full(
            (number_of_nodes, max_subtoken_length.item()),
            self.token_pad_index, dtype=torch.long, device=self.device
        )
        for node in range(number_of_nodes):
            token_id = graph.nodes[node].data['token_id'].item()
            subtokens[node, :subtoken_lengths[node]] = self.token_to_subtoken[token_id]
        graph.ndata['token_embeds'] = torch.sum(
            self.subtoken_embedding(subtokens), dim=1
        )
        return graph


class SubTokenTypeEmbedding(SubTokenEmbedding):

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, device: torch.device) -> None:
        if UNK not in type_to_id:
            type_to_id[UNK] = len(type_to_id)
        super().__init__(token_to_id, type_to_id, h_emb, device)
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.h_emb, padding_idx=self.type_pad_index)

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        graph = super().forward(graph)
        graph.ndata['type_embeds'] = self.type_embedding(graph.ndata['type_id'])
        return graph
