from typing import Tuple, Dict

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder
from model.treeLSTM_cell import ChildSumTreeLSTMCell, LuongAttentionTreeLSTMCell, MultiHeadAttentionTreeLSTMCell


class TreeLSTM(_IEncoder):

    _tree_lstm_cells = {
        ChildSumTreeLSTMCell.__name__: ChildSumTreeLSTMCell,
        LuongAttentionTreeLSTMCell.__name__: LuongAttentionTreeLSTMCell,
        MultiHeadAttentionTreeLSTMCell.__name__: MultiHeadAttentionTreeLSTMCell
    }

    def __init__(self, h_emb: int, h_enc: int, cell: Dict, dropout: float = 0., n_layers: int = 1):
        super().__init__(h_emb, h_enc)
        if cell['name'] not in self._tree_lstm_cells:
            raise ValueError(f"unknown TreeLSTM cell: {cell['name']}")
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        self.norm = [nn.LayerNorm(h_enc) for _ in range(self.n_layers)]
        self.cell = [
            self._tree_lstm_cells[cell['name']](self.h_emb, self.h_enc, **cell['params']) for _ in range(self.n_layers)
        ]

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(graph.ndata['x'])

        for layer in range(self.n_layers):
            graph.ndata['x'] = x

            graph = self.cell[layer].init_matrices(graph, device)
            dgl.prop_nodes_topo(
                graph,
                reduce_func=self.cell[layer].get_reduce_func(),
                message_func=self.cell[layer].get_message_func(),
                apply_node_func=self.cell[layer].get_apply_node_func()
            )

            x = self.norm[layer](graph.ndata.pop('h') + x)

        c = graph.ndata.pop('c')
        return x, c
