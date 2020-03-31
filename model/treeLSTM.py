from typing import Tuple, Dict

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder
from model.treeLSTM_cell import EdgeChildSumTreeLSTMCell, NodeChildSumTreeLSTMCell, \
    EdgeSpecificTreeLSTMCell, TypeSpecificTreeLSTMCell, TypeAttentionTreeLSTMCell, FullMultiHeadAttentionTreeLSTMCell


class TreeLSTM(_IEncoder):

    _tree_lstm_cells = {
        EdgeChildSumTreeLSTMCell.__name__: EdgeChildSumTreeLSTMCell,
        NodeChildSumTreeLSTMCell.__name__: NodeChildSumTreeLSTMCell,
        EdgeSpecificTreeLSTMCell.__name__: EdgeSpecificTreeLSTMCell,
        TypeSpecificTreeLSTMCell.__name__: TypeSpecificTreeLSTMCell,
        TypeAttentionTreeLSTMCell.__name__: TypeAttentionTreeLSTMCell,
        FullMultiHeadAttentionTreeLSTMCell.__name__: FullMultiHeadAttentionTreeLSTMCell
    }

    def __init__(self, h_emb: int, h_enc: int, cell: Dict, dropout: float = 0.):
        super().__init__(h_emb, h_enc)
        if cell['name'] not in self._tree_lstm_cells:
            raise ValueError(f"unknown TreeLSTM cell: {cell['name']}")
        self.cell = self._tree_lstm_cells[cell['name']](self.h_emb, self.h_enc, **cell['params'])
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        graph.ndata['x'] = self.dropout(graph.ndata['x'])
        return self.cell(graph, device)
