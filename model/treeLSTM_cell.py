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


class ChildSumTreeLSTMCell(_ITreeLSTMCell):
    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.U_iou = nn.Parameter(torch.zeros(self.h_size, 3 * self.h_size), requires_grad=True)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)

        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.U_f = nn.Parameter(torch.zeros(self.h_size, self.h_size), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros((1, self.h_size)), requires_grad=True)

        nn.init.kaiming_uniform_(self.U_iou, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U_f, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b_iou, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b_f, a=math.sqrt(5))

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        h = edges.src['h'].unsqueeze(1)
        h_f = torch.bmm(h, edges.data['U_f']).squeeze(1)
        x_f = edges.dst['x_f']
        f = torch.sigmoid(x_f + h_f)
        return {
            'h_iou': torch.bmm(h, edges.data['U_iou']).squeeze(1),
            'fc': edges.src['c'] * f
        }

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        # x_f = nodes.data['x_f'].unsqueeze(1).expand(nodes.mailbox['h_f'].shape)
        # f = torch.sigmoid(
        #     x_f + nodes.mailbox['h_f']
        # )
        # fc_sum = torch.sum(nodes.mailbox['c'] * f, 1)
        return {
            'Uh_sum': torch.sum(nodes.mailbox['h_iou'], 1),
            # 'fc_sum': fc_sum,
            'fc_sum': torch.sum(nodes.mailbox['fc'], 1)
        }

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

        graph.edata['U_iou'] = self.U_iou.unsqueeze(0).expand(graph.number_of_edges(), -1, -1)
        graph.edata['U_f'] = self.U_f.unsqueeze(0).expand(graph.number_of_edges(), -1, -1)

        graph.register_message_func(self.message_func)
        # graph.register_reduce_func(self.reduce_func)
        graph.register_apply_node_func(self.apply_node_func)

        dgl.prop_nodes_topo(graph, reduce_func=[fn.sum('h_iou', 'Uh_sum'), fn.sum('fc', 'fc_sum')])

        h = graph.ndata.pop('h')
        c = graph.ndata.pop('c')
        return h, c

    def get_params(self):
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.data, 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.data, 'b_f': self.b_f.data
        }
