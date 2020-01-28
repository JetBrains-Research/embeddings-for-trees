import math
from typing import Dict, Tuple

import dgl
import dgl.function as fn
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

    def get_params(self) -> Dict:
        raise NotImplementedError


class EdgeChildSumTreeLSTMCell(_ITreeLSTMCell):
    """All calculations are happening in message function,
    reduce function only sum children features
    """

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)

        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        h_f = self.U_f(edges.src['h'])
        x_f = edges.dst['x_f']
        f = torch.sigmoid(x_f + h_f)
        return {
            'Uh': self.U_iou(edges.src['h']),
            'fc': edges.src['c'] * f
        }

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        """Using builtin functions"""
        raise NotImplementedError

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        iou = nodes.data['x_iou'] + nodes.data['Uh_sum']
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + nodes.data['fc_sum']
        h = o * torch.tanh(c)

        return {'h': h, 'c': c, 'i': i}

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        number_of_nodes = graph.number_of_nodes()
        graph.ndata['x_iou'] = self.W_iou(graph.ndata['x']) + self.b_iou
        graph.ndata['x_f'] = self.W_f(graph.ndata['x']) + self.b_f
        graph.ndata['h'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['c'] = torch.zeros((number_of_nodes, self.h_size), device=device)
        graph.ndata['Uh_sum'] = torch.zeros((number_of_nodes, 3 * self.h_size), device=device)
        graph.ndata['fc_sum'] = torch.zeros((number_of_nodes, self.h_size), device=device)

        graph.register_message_func(self.message_func)
        graph.register_apply_node_func(self.apply_node_func)

        dgl.prop_nodes_topo(graph, reduce_func=[fn.sum('Uh', 'Uh_sum'), fn.sum('fc', 'fc_sum')])

        h = graph.ndata.pop('h')
        c = graph.ndata.pop('c')
        return h, c

    def get_params(self):
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.weight.t(), 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.weight.t(), 'b_f': self.b_f.data
        }


class NodeChildSumTreeLSTMCell(_ITreeLSTMCell):
    """All calculations are happening in reduce function
    message function only pass features from source to destination node
    """

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)

        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        """Using builtin functions"""
        raise NotImplementedError

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        h_tilda = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']) + nodes.data['node_f'].unsqueeze(1))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'Uh_tilda': self.U_iou(h_tilda), 'c': c}

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        iou = nodes.data['node_iou'] + nodes.data['Uh_tilda']
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # register function for message passing
        graph.register_reduce_func(self.reduce_func)
        graph.register_apply_node_func(self.apply_node_func)

        features = graph.ndata['x']
        nodes_in_batch = graph.number_of_nodes()
        graph.ndata['node_iou'] = self.W_iou(features) + self.b_iou
        graph.ndata['node_f'] = self.W_f(features) + self.b_f
        graph.ndata['h'] = torch.zeros(nodes_in_batch, self.h_size).to(device)
        graph.ndata['c'] = torch.zeros(nodes_in_batch, self.h_size).to(device)
        graph.ndata['Uh_tilda'] = torch.zeros(nodes_in_batch, 3 * self.h_size).to(device)
        # propagate
        dgl.prop_nodes_topo(graph, message_func=[fn.copy_u('h', 'h'), fn.copy_u('c', 'c')])
        # get encoded output
        h = graph.ndata.pop('h')
        c = graph.ndata.pop('c')
        return h, c

    def get_params(self):
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.weight.t(), 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.weight.t(), 'b_f': self.b_f.data
        }
