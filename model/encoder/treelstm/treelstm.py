from typing import Tuple, Dict, Union, List

import dgl
import torch
import torch.nn as nn

from model.encoder import ITreeEncoder
from utils.common import get_root_indexes


class ITreeLSTMCell(nn.Module):

    name = "TreeLSTM cell interface"

    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size

        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)

        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def get_message_func(self):
        raise NotImplementedError

    def get_reduce_func(self):
        raise NotImplementedError

    @staticmethod
    def get_apply_node_func():
        def apply_node_func(nodes: dgl.NodeBatch) -> Dict:
            iou = nodes.data['x_iou'] + nodes.data['Uh_sum']
            i, o, u = torch.chunk(iou, 3, 1)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

            c = i * u + nodes.data['fc_sum']
            h = o * torch.tanh(c)

            return {'h': h, 'c': c}
        return apply_node_func

    def init_matrices(self, graph: dgl.DGLGraph, device: torch.device) -> dgl.DGLGraph:
        number_of_nodes = graph.number_of_nodes()
        graph.ndata['x_iou'] = self.W_iou(graph.ndata['x']) + self.b_iou
        graph.ndata['x_f'] = self.W_f(graph.ndata['x']) + self.b_f
        graph.ndata['h'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['c'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['Uh_sum'] = torch.zeros((number_of_nodes, 3 * self.h_size), device=device)
        graph.ndata['fc_sum'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        return graph


class TreeLSTM(ITreeEncoder):

    name = "TreeLSTM"
    _known_tree_lstm_cells = {}

    def __init__(
            self, h_emb: int, h_enc: int, cell: Dict,
            dropout: float = 0., n_layers: int = 1, residual: bool = False
    ):
        super().__init__(h_emb, h_enc)
        if cell['name'] not in self._known_tree_lstm_cells:
            raise ValueError(f"unknown TreeLSTM cell: {cell['name']}")
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.residual = residual

        self.norm = nn.ModuleList([nn.LayerNorm(h_enc) for _ in range(self.n_layers)])
        self.cell = nn.ModuleList([
            self._known_tree_lstm_cells[cell['name']](self.h_emb, self.h_enc, **cell['params']) for _ in range(self.n_layers)
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

            if self.residual:
                x = self.norm[layer](graph.ndata.pop('h') + x)
            else:
                x = graph.ndata.pop('h')

        c = graph.ndata.pop('c')
        return x, c

    @staticmethod
    def register_cell(tree_lstm_cell: ITreeLSTMCell):
        if not issubclass(tree_lstm_cell, ITreeLSTMCell):
            raise ValueError(f"Attempt to register not a Tree Encoder class "
                             f"({tree_lstm_cell.__name__} not a subclass of {ITreeLSTMCell.__name__})")
        TreeLSTM._known_tree_lstm_cells[tree_lstm_cell.name] = tree_lstm_cell

    @staticmethod
    def get_known_cells() -> List[str]:
        return list(TreeLSTM._known_tree_lstm_cells.keys())


class ChildSumTreeLSTMCell(ITreeLSTMCell):
    """All calculations are happening in message function,
    reduce function only sum children features
    """

    name = "ChildSum"

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)

    def get_message_func(self):
        def message_func(edges: dgl.EdgeBatch) -> Dict:
            h_f = self.U_f(edges.src['h'])
            x_f = edges.dst['x_f']
            f = torch.sigmoid(x_f + h_f)
            return {
                'Uh': self.U_iou(edges.src['h']),
                'fc': edges.src['c'] * f
            }
        return message_func

    def get_reduce_func(self):
        return [dgl.function.sum('Uh', 'Uh_sum'), dgl.function.sum('fc', 'fc_sum')]


class DfsLSTM(ITreeEncoder):

    name = "DfsLSTM"

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


class TwoOrderLSTM(ITreeEncoder):

    name = "TwoOrderLSTM"

    def __init__(self, h_emb: int, h_enc: int, dropout: float):
        super().__init__(h_emb, h_enc)
        cell_info = {
            'name': ChildSumTreeLSTMCell.name,
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
