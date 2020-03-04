from typing import Dict

import dgl
import torch
import torch.nn as nn
from dgl import BatchedDGLGraph
from numpy import sqrt

from utils.common import UNK, PAD, NAN, METHOD_NAME
from utils.token_processing import get_dict_of_subtokens


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


class FullTypeEmbedding(_IEmbedding):
    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int) -> None:
        super().__init__(token_to_id, type_to_id, h_emb)
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.h_emb, padding_idx=self.type_pad_index)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph.ndata['type_embeds'] = self.type_embedding(graph.ndata['type_id'])
        return graph


class SubTokenEmbedding(_IEmbedding):
    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, delimiter: str = '|') -> None:
        self.delimiter = delimiter
        self.subtoken_to_id, self.token_to_subtokens =\
            get_dict_of_subtokens(token_to_id, required_tokens=[UNK, PAD, METHOD_NAME, NAN], delimiter=delimiter)
        # subtoken_to_id saved to token_to_id via super class init
        super().__init__(self.subtoken_to_id, type_to_id, h_emb)
        self.subtoken_embedding = nn.Embedding(
            self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index
        )
        self.full_token_id_to_token = {v: k for k, v in token_to_id.items()}
        self.full_token_id_to_subtokens = {
            _id: self.token_to_subtokens[token] for token, _id in token_to_id.items()
        }

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        start_index = 0
        subtoken_ids = []
        node_slices = []
        for node in graph.ndata['token_id']:
            node_id = node.item()

            if node_id in self.full_token_id_to_subtokens:
                cur_subtokens = self.full_token_id_to_subtokens[node_id]
            else:
                unk_id = self.subtoken_to_id[UNK]
                cur_subtokens = [
                    self.subtoken_to_id.get(st, unk_id)
                    for st in self.full_token_id_to_token[node_id].split(self.delimiter)
                ]

            subtoken_ids += cur_subtokens
            node_slices.append(slice(start_index, start_index + len(cur_subtokens)))
            start_index += len(cur_subtokens)

        full_subtokens_embeds = self.subtoken_embedding(torch.tensor(subtoken_ids, device=device))

        token_embeds = torch.zeros((graph.number_of_nodes(), self.h_emb), device=device)
        for node in range(graph.number_of_nodes()):
            token_embeds[node] = full_subtokens_embeds[node_slices[node]].sum(0)

        graph.ndata['token_embeds'] = token_embeds
        return graph


class SubTokenTypeEmbedding(_IEmbedding):

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int) -> None:
        super().__init__(token_to_id, type_to_id, h_emb)
        self.subtoken_embedding = SubTokenEmbedding(self.token_to_id, self.type_to_id, self.h_emb)
        self.type_embedding = FullTypeEmbedding(self.token_to_id, self.type_to_id, self.h_emb)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph = self.subtoken_embedding(graph, device)
        graph = self.type_embedding(graph, device)
        return graph


class PositionalEmbedding(nn.Module):
    """Implement positional embedding from
    https://papers.nips.cc/paper/9376-novel-positional-encodings-to-enable-tree-based-transformers.pdf
    """

    def __init__(self, n: int, k: int, p: float = 1.) -> None:
        super().__init__()
        self.n, self.k, self.p = n, k, p
        self.h_emb = self.n * self.k
        self.p_emb = torch.tensor([self.p ** i for i in range(self.h_emb)])

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        """Forward pass for positional embedding

        @param graph: a batched graph with oriented edges from leaves to roots
        @param device: torch device
        @return: a batched graph with "pos_embeds" field in each node
        """
        graph.ndata['pos_embeds'] = torch.zeros(graph.number_of_nodes(), self.h_emb, device=device)
        for layer in dgl.topological_nodes_generator(graph, reverse=True):
            for node in layer:
                children = graph.in_edges(node, form='uv')[0]
                graph.ndata['pos_embeds'][children, self.n:] = graph.ndata['pos_embeds'][node, :-self.n]
                graph.ndata['pos_embeds'][children, :self.n] = torch.eye(children.shape[0], self.n, device=device)
        # TODO: implement parametrized positional embedding with using p
        return graph


class PositionalSubTokenTypeEmbedding(_IEmbedding):

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, n: int, k: int, p: float = 1.):
        assert h_emb == n * k
        super().__init__(token_to_id, type_to_id, h_emb)
        self.subtoken_embedding = SubTokenEmbedding(self.token_to_id, self.type_to_id, self.h_emb)
        self.type_embedding = FullTypeEmbedding(self.token_to_id, self.type_to_id, self.h_emb)
        self.positional_embedding = PositionalEmbedding(n, k, p)

    def forward(self, graph: BatchedDGLGraph, device: torch.device) -> BatchedDGLGraph:
        graph = self.subtoken_embedding(graph, device)
        graph = self.type_embedding(graph, device)
        graph = self.positional_embedding(graph, device)
        graph.ndata['token_embeds'] *= sqrt(self.h_emb)
        graph.ndata['type_embeds'] *= sqrt(self.h_emb)
        return graph
