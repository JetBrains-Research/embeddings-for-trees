from typing import Dict, Tuple

import dgl
import torch
import torch.nn as nn


class _ITreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        raise NotImplementedError

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        raise NotImplementedError

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        raise NotImplementedError

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagate nodes by defined order,
        assuming graph.ndata['x'] contain features
        """
        raise NotImplementedError


class ChildSumTreeLSTMCell(_ITreeLSTMCell):
    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)
        self.U_f = nn.Linear(self.h_size, self.h_size)
        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, self.h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        return {
            'h': edges.src['h'],
            'f_dot_c': edges.src['f_dot_c'],
        }

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        h_sum = torch.sum(nodes.mailbox['h'], 1)
        fc_sum = torch.sum(nodes.mailbox['f_dot_c'], 1)
        return {'Uh_sum': self.U_iou(h_sum), 'fc_sum': fc_sum}

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        iou = nodes.data['x_iou'] + nodes.data['Uh_sum']
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + nodes.data['fc_sum']
        h = o * torch.tanh(c)

        f = torch.sigmoid(nodes.data['x_f'] + self.U_f(h))
        f_dot_c = f * c
        return {'h': h, 'f_dot_c': f_dot_c}

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        number_of_nodes = graph.number_of_nodes()
        graph.ndata['x_iou'] = self.W_iou(graph.ndata['x']) + self.b_iou
        graph.ndata['x_f'] = self.W_f(graph.ndata['x']) + self.b_f
        graph.ndata['h'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['c'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['Uh_sum'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['fc_sum'] = torch.zeros((number_of_nodes, self.h_size), device=device)

        graph.register_message_func(self.message_func)
        graph.register_reduce_func(self.reduce_func)
        graph.register_apply_node_func(self.apply_node_func)

        dgl.prop_nodes_topo(graph)

        h = graph.ndata.pop('h')
        c = graph.ndata.pop('c')
        return h, c

