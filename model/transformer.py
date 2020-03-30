from typing import Dict

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder


class TransformerEncoder(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, n_head: int, h_ffd: int = 2048, dropout: float = 0.1) -> None:
        """init transformer encoder

        :param h_emb: size of embedding
        :param h_enc: size of encoder
        :param n_head: number of heads in multi-head attention
        :param h_ffd: size of hidden layer in feedforward part
        :param dropout: probability to be zeroed
        """
        super().__init__(h_emb, h_enc)
        self.multihead_attention = nn.MultiheadAttention(self.h_emb, n_head, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.h_emb)

        self.linear1 = nn.Linear(self.h_emb, h_ffd)
        self.relu = nn.ReLU()
        self.dropout_ffd = nn.Dropout(dropout)
        self.linear2 = nn.Linear(h_ffd, self.h_emb)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(self.h_emb)

        self.linear_h = nn.Linear(self.h_emb, self.h_enc)
        self.linear_c = nn.Linear(self.h_emb, self.h_enc)

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        """reduce a batch of incoming nodes

        :param nodes: 'x' is a tensor of shape [batch size, number of children, hidden state]
        :return: Dict with reduced features
        """
        # [n_child, bs, h]
        x_in_nodes = nodes.mailbox['x_trans'].transpose(0, 1)
        # [1, bs, h]
        x_cur = nodes.data['x'].unsqueeze(0)

        # [1, bs, h]
        x_attn = self.multihead_attention(x_cur, x_in_nodes, x_in_nodes)[0]

        x_add_norm = self.norm1(x_cur + self.dropout1(x_attn))
        return {
            'x': x_add_norm.squeeze(0)
        }

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        # [1, bs, h]
        x_cur = nodes.data['x'].unsqueeze(0)

        # [1, bs, h]
        x_linear = self.linear2(self.dropout_ffd(self.relu(self.linear1(x_cur))))

        x_trans = x_cur + self.dropout2(x_linear)
        x_trans = self.norm2(x_trans)
        return {'x_trans': x_trans.squeeze(0)}

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> torch.Tensor:
        """Apply transformer encoder

        :param graph: batched dgl graph
        :param device: torch device
        :return: encoded nodes [max tree size, batch size, hidden state]
        """
        graph.ndata['x_trans'] = torch.zeros(graph.number_of_nodes(), self.h_emb, device=device)
        dgl.prop_nodes_topo(
            graph, message_func=[dgl.function.copy_u('x_trans', 'x_trans')],
            reduce_func=self.reduce_func, apply_node_func=self.apply_node_func
        )

        # [n_nodes, h_emb]
        h = self.linear_h(graph.ndata['x_trans'])
        c = self.linear_c(graph.ndata['x_trans'])

        return h, c
