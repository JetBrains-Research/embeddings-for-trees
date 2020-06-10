from typing import Tuple, Dict, List

import dgl
import torch
import torch.nn as nn

from model.encoder import ITreeEncoder


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

    def init_matrices(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        number_of_nodes = graph.number_of_nodes()
        graph.ndata['x_iou'] = self.W_iou(graph.ndata['x']) + self.b_iou
        graph.ndata['x_f'] = self.W_f(graph.ndata['x']) + self.b_f
        graph.ndata['h'] = graph.ndata['x'].new_zeros((number_of_nodes, self.h_size))
        graph.ndata['c'] = graph.ndata['x'].new_zeros((number_of_nodes, self.h_size))
        graph.ndata['Uh_sum'] = graph.ndata['x'].new_zeros((number_of_nodes, 3 * self.h_size))
        graph.ndata['fc_sum'] = graph.ndata['x'].new_zeros((number_of_nodes, self.h_size))
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

    def forward(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.dropout(graph.ndata['x'])

        for layer in range(self.n_layers):
            graph.ndata['x'] = x

            graph = self.cell[layer].init_matrices(graph)
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
