from typing import Dict

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder


class TransformerEncoder(_IEncoder):

    def __init__(
            self, h_emb: int, h_enc: int, n_head: int, h_ffd: int = 2048, dropout: float = 0.1, n_layers: int = 1
    ) -> None:
        """init transformer encoder

        :param h_emb: size of embedding
        :param h_enc: size of encoder
        :param n_head: number of heads in multi-head attention
        :param h_ffd: size of hidden layer in feedforward part
        :param dropout: probability to be zeroed
        """
        super().__init__(h_emb, h_enc)
        self.transformer_layer = nn.TransformerEncoderLayer(h_emb, n_head, h_ffd, dropout)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, n_layers)

        self.linear_h = nn.Linear(self.h_emb, self.h_enc)
        self.linear_c = nn.Linear(self.h_emb, self.h_enc)
        self.tanh = nn.Tanh()

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        """reduce a batch of incoming nodes

        :param nodes: 'x' is a tensor of shape [batch size, number of children, hidden state]
        :return: Dict with reduced features
        """
        # [n_child, bs, h]
        x_children = nodes.mailbox['x'].transpose(0, 1)
        # [1, bs, h]
        x_cur = nodes.data['x'].unsqueeze(0)

        # [n_child + 1, bs, h]
        x = torch.cat([x_cur, x_children], dim=0)
        # root attend on children
        mask = torch.full((x.shape[0], x.shape[0]), -1e5)
        mask[0, 1:] = 0
        # [n_child + 1, bs, h]
        x_trans = self.transformer(x, mask=mask)
        return {
            'x': x_trans[0]
        }

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> torch.Tensor:
        """Apply transformer encoder

        :param graph: batched dgl graph
        :param device: torch device
        :return: encoded nodes [max tree size, batch size, hidden state]
        """
        # graph.ndata['x_trans'] = torch.zeros(graph.number_of_nodes(), self.h_emb, device=device)
        dgl.prop_nodes_topo(
            graph, message_func=[dgl.function.copy_u('x', 'x')],
            reduce_func=self.reduce_func
        )
        # print(graph.ndata['x'])

        # [n_nodes, h_emb]
        h = self.tanh(self.linear_h(graph.ndata['x']))
        c = self.tanh(self.linear_c(graph.ndata['x']))

        return h, c
