from typing import Tuple, Dict, Union

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder
from model.treeLSTM_cell import ChildSumTreeLSTMCell, LuongAttentionTreeLSTMCell, SelfAttentionTreeLSTMCell, \
    SequenceTreeLSTMCell
from utils.common import get_root_indexes


class TreeLSTM(_IEncoder):

    _tree_lstm_cells = {
        ChildSumTreeLSTMCell.__name__: ChildSumTreeLSTMCell,
        LuongAttentionTreeLSTMCell.__name__: LuongAttentionTreeLSTMCell,
        SelfAttentionTreeLSTMCell.__name__: SelfAttentionTreeLSTMCell,
        SequenceTreeLSTMCell.__name__: SequenceTreeLSTMCell
    }

    def __init__(self, h_emb: int, h_enc: int, cell: Dict, dropout: float = 0., n_layers: int = 1):
        super().__init__(h_emb, h_enc)
        if cell['name'] not in self._tree_lstm_cells:
            raise ValueError(f"unknown TreeLSTM cell: {cell['name']}")
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        self.norm = nn.ModuleList([nn.LayerNorm(h_enc) for _ in range(self.n_layers)])
        self.cell = nn.ModuleList([
            self._tree_lstm_cells[cell['name']](self.h_emb, self.h_enc, **cell['params']) for _ in range(self.n_layers)
        ])

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


class DfsLSTM(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, dropout: float):
        super().__init__(h_emb, h_enc)
        self.lstm = nn.LSTMCell(self.h_emb, self.h_enc)
        self.dropout = nn.Dropout(dropout)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        # [bs; h size]
        h = edges.src['h']
        c = edges.src['c']
        # [bs; x size]
        x = edges.dst['x']
        h_cur, c_cur = self.lstm(x, (h, c))
        return {
            'h': h_cur,
            'c': c_cur
        }

    @staticmethod
    def reduce_func(nodes: dgl.NodeBatch) -> Dict:
        return {
            'h': nodes.mailbox['h'].squeeze(1),
            'c': nodes.mailbox['c'].squeeze(1)
        }

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        graph.ndata['x'] = self.dropout(graph.ndata['x'])
        root_indexes = torch.tensor(
            get_root_indexes(graph.batch_num_nodes), dtype=torch.long, device=device, requires_grad=False
        )
        graph.ndata['h'] = torch.zeros((graph.number_of_nodes(), self.h_enc), device=device)
        graph.ndata['c'] = torch.zeros((graph.number_of_nodes(), self.h_enc), device=device)
        graph.ndata['h'][root_indexes], graph.ndata['c'][root_indexes] = \
            self.lstm(graph.ndata['x'][root_indexes])

        dgl.prop_edges_dfs(
            graph, root_indexes, True,
            message_func=self.message_func,
            reduce_func=self.reduce_func
        )

        return graph.ndata.pop('h'), graph.ndata.pop('c')


class TwoOrderLSTM(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, dropout: float):
        super().__init__(h_emb, h_enc)
        cell_info = {
            'name': ChildSumTreeLSTMCell.__name__,
            'params': {}
        }
        self.tree_lstm = TreeLSTM(self.h_emb, self.h_enc, cell_info, dropout=dropout)
        self.dfs_lstm = DfsLSTM(self.h_emb, self.h_enc, dropout=dropout)

        self.blend_alpha = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)
        self.linear_h = nn.Linear(self.h_enc, self.h_enc)
        self.linear_c = nn.Linear(self.h_enc, self.h_enc)

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        h_tree_lstm, c_tree_lstm = self.tree_lstm(graph, device)
        h_dfs_lstm, c_dfs_lstm = self.dfs_lstm(graph, device)

        h = self.blend_alpha[0] * h_tree_lstm + (1 - self.blend_alpha[0]) * h_dfs_lstm
        c = self.blend_alpha[1] * c_tree_lstm + (1 - self.blend_alpha[1]) * c_dfs_lstm

        return nn.functional.relu(self.linear_h(h)), nn.functional.relu(self.linear_c(c))
