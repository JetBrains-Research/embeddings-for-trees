from typing import Dict

import dgl
import torch
import torch.nn as nn

from model.encoder import ITreeEncoder


class TransformerEncoder(ITreeEncoder):

    name = "Transformer"

    def __init__(
            self, h_emb: int, h_enc: int, n_heads: int, h_ffd: int = 2048, dropout: float = 0.1, n_layers: int = 1
    ) -> None:
        """init transformer encoder

        :param h_emb: size of embedding
        :param h_enc: size of encoder
        :param n_heads: number of heads in multi-head attention
        :param h_ffd: size of hidden layer in feedforward part
        :param dropout: probability to be zeroed
        """
        super().__init__(h_emb, h_enc)
        self.transformer_layer = nn.TransformerEncoderLayer(h_emb, n_heads, h_ffd, dropout)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, n_layers)
        self.norm = nn.LayerNorm(h_enc)

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        """reduce a batch of incoming nodes

        :param nodes:
            mailbox -- tensor [batch size, number of children, hidden state]
            data -- tensor [batch size, hidden state]
        :return: Dict with reduced features
        """
        # [n_children, bs, h]
        h_children = nodes.mailbox['h'].transpose(0, 1)
        h_trans = self.transformer(h_children).transpose(0, 1)
        return {
            'h': h_trans.sum(1)
        }

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        h = self.norm(nodes.data['x'] + nodes.data['h'])
        return {'h': h}

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Apply transformer encoder

        :param graph: batched dgl graph
        :return: encoded nodes [number of nodes, hidden size]
        """
        graph.ndata['h'] = graph.ndata['x'].new_zeros((graph.number_of_nodes(), self.h_enc))
        dgl.prop_nodes_topo(
            graph, message_func=[dgl.function.copy_u('h', 'h')],
            reduce_func=self.reduce_func,
            apply_node_func=self.apply_node_func
        )
        return graph.ndata.pop('h')
