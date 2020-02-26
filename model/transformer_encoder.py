from typing import Tuple

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder
from utils.common import segment_sizes_to_slices


class Transformer(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, n_head: int, n_layers: int,
                 h_ffd: int = 2048, dropout: float = 0.1, activation: str = "relu") -> None:
        """init transformer encoder

        :param h_emb: size of embedding
        :param h_enc: size of encoder
        :param n_head: number of heads in multi-head attention
        :param n_layers: number of transformer cell layers
        :param h_ffd: size of hidden layer in feedforward part
        :param dropout: probability to be zeroed
        :param activation: type of activation
        """
        assert h_emb == h_enc, f"for transformer encoder size of embedding should be equal to size of encoder"
        super().__init__(h_emb, h_enc)
        self.n_head = n_head
        self.h_ffd = h_ffd
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        encoder_layers = nn.TransformerEncoderLayer(
            self.h_emb, self.n_head, dim_feedforward=self.h_ffd, dropout=self.dropout, activation=self.activation
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.n_layers)

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transformer encoder

        :param graph: batched dgl graph
        :param device: torch device
        :return: tensor of shape [max tree size, batch size, h_enc]
        """
        embeds = [key for key in graph.ndata if 'embeds' in key]
        graphs = dgl.unbatch(graph)
        tree_sizes = [g.number_of_nodes() for g in graphs]
        max_tree_size = max(tree_sizes)
        tree_slices = segment_sizes_to_slices(tree_sizes)

        # [number of nodes, h_emb]
        embeds_sum = sum([graph.ndata[e] for e in embeds])
        # [max_tree_size, batch_size, h_enc]
        features = torch.zeros(max_tree_size, len(graphs), self.h_enc, device=device)

        for i, tree_slice in enumerate(tree_slices):
            features[:tree_sizes[i], i, :] = embeds_sum[tree_slice]

        # [max_tree_size, batch_size, h_enc]
        encoded = self.transformer_encoder(features)

        # [number_of_nodes, h_enc]
        output = torch.cat([
            encoded[:tree_sizes[i], i, :] for i in range(len(tree_sizes))
        ], dim=0)
        return output, output
