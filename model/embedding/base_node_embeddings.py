from typing import Dict

import dgl
import torch
from numpy.ma import sqrt
from torch import nn

from model.embedding import INodeEmbedding
from utils.common import UNK, PAD, METHOD_NAME, NAN, SELF
from utils.token_processing import get_dict_of_subtokens


class TokenNodeEmbedding(INodeEmbedding):

    name = "token"

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, normalize: bool = False) -> None:
        super().__init__(token_to_id, type_to_id, h_emb)
        self.token_embedding = nn.Embedding(self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index)
        self.normalize = normalize

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        token_embeds = self.token_embedding(graph.ndata['token']).squeeze(1)
        if self.normalize:
            return token_embeds * sqrt(self.h_emb)
        return token_embeds


class TypeNodeEmbedding(INodeEmbedding):

    name = "type"

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, normalize: bool = False) -> None:
        super().__init__(token_to_id, type_to_id, h_emb)
        self.type_embedding = nn.Embedding(self.type_vocab_size, self.h_emb, padding_idx=self.type_pad_index)
        self.normalize = normalize

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        type_embeds = self.type_embedding(graph.ndata['type'])
        if self.normalize:
            return type_embeds * sqrt(self.h_emb)
        return type_embeds


class SubTokenNodeEmbedding(INodeEmbedding):

    name = "subtoken"

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, normalize: bool = False,
                 delimiter: str = '|') -> None:
        self.delimiter = delimiter
        self.normalize = normalize
        self.subtoken_to_id, self.token_to_subtokens = \
            get_dict_of_subtokens(token_to_id, required_tokens=[UNK, PAD, METHOD_NAME, NAN, SELF], delimiter=delimiter)
        # subtoken_to_id saved to token_to_id via super class init
        super().__init__(self.subtoken_to_id, type_to_id, h_emb)
        self.subtoken_embedding = nn.Embedding(
            self.token_vocab_size, self.h_emb, padding_idx=self.token_pad_index
        )
        self.full_token_id_to_token = {v: k for k, v in token_to_id.items()}
        self.full_token_id_to_subtokens = {
            _id: self.token_to_subtokens[token] for token, _id in token_to_id.items()
        }

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
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

        full_subtokens_embeds = self.subtoken_embedding(
            graph.ndata['token_id'].new_tensor(subtoken_ids)
        )

        token_embeds = graph.ndata['token_id'].new_empty((graph.number_of_nodes(), self.h_emb))
        for node in range(graph.number_of_nodes()):
            token_embeds[node] = full_subtokens_embeds[node_slices[node]].sum(0)
        if self.normalize:
            return token_embeds * sqrt(self.h_emb)
        return token_embeds
