from typing import Dict

import torch
import torch.nn as nn
from dgl import BatchedDGLGraph

from utils.common import UNK, PAD, NAN, METHOD_NAME
from utils.token_processing import get_dict_of_subtokens, get_token_id_to_subtoken_dict


class _IEmbedding(nn.Module):
    """Interface of embedding module.
    Forward method takes batched graph and applies embedding to its features.
    """

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

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        raise NotImplementedError


class FullTokenEmbedding(_IEmbedding):
    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int) -> None:
        super().__init__(token_to_id, type_to_id, h_emb)
        self.token_embedding = nn.Embedding(self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph.ndata['token_embeds'] = self.token_embedding(graph.ndata['token_id'])
        return graph


class SubTokenEmbedding(_IEmbedding):
    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int) -> None:
        self.subtoken_to_id = get_dict_of_subtokens(token_to_id, required_tokens=[UNK, PAD, METHOD_NAME, NAN])
        super().__init__(self.subtoken_to_id, type_to_id, h_emb)
        self.subtoken_embedding = nn.Embedding(
            self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index
        )
        self.token_id_to_full_token = {v: k for k, v in token_to_id.items()}

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        number_of_nodes = graph.number_of_nodes()

        # since in __init__ token_to_id replaced by subtoken_to_id
        token_id_to_subtoken = get_token_id_to_subtoken_dict(
            graph.ndata['token_id'].tolist(), self.token_id_to_full_token, self.token_to_id, device
        )

        subtoken_lengths = torch.tensor([
            token_id_to_subtoken[token.item()].shape[0] for token in graph.ndata['token_id']
        ])
        max_subtoken_length = subtoken_lengths.max()

        subtokens = torch.full(
            (number_of_nodes, max_subtoken_length.item()),
            self.token_pad_index, dtype=torch.long, device=device
        )
        for node in range(number_of_nodes):
            token_id = graph.nodes[node].data['token_id'].item()
            subtokens[node, :subtoken_lengths[node]] = token_id_to_subtoken[token_id]
        graph.ndata['token_embeds'] = torch.sum(
            self.subtoken_embedding(subtokens), dim=1
        )
        return graph


class SubTokenTypeEmbedding(SubTokenEmbedding):

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int) -> None:
        super().__init__(token_to_id, type_to_id, h_emb)
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.h_emb, padding_idx=self.type_pad_index)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph = super().forward(graph, device)
        graph.ndata['type_embeds'] = self.type_embedding(graph.ndata['type_id'])
        return graph
