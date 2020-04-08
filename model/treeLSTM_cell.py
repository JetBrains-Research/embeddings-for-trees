from typing import Dict, Tuple

import dgl
import torch
import torch.nn as nn


class _ITreeLSTMCell(nn.Module):
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


class ChildSumTreeLSTMCell(_ITreeLSTMCell):
    """All calculations are happening in message function,
    reduce function only sum children features
    """

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


class LuongAttentionTreeLSTMCell(_ITreeLSTMCell):

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)

        self.W_a = nn.Linear(self.x_size + self.h_size, self.h_size, bias=False)
        self.v_a = nn.Linear(self.h_size, 1, bias=False)

    def get_message_func(self):
        return [dgl.function.copy_u('h', 'h'), dgl.function.copy_u('c', 'c')]

    def _reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        # [bs; n children; h size]
        h_children = nodes.mailbox['h']

        # [bs; n children; x size]
        x = nodes.data['x'].unsqueeze(1).expand(-1, h_children.shape[1], -1)

        # [bs; n children; h size]
        energy = torch.tanh(self.W_a(
            # [bs; n children; x size + h size]
            torch.cat([x, h_children], dim=2)
        ))

        # [bs; n children]
        scores = self.v_a(energy).squeeze(2)

        # [bs; n children]
        align = nn.functional.softmax(scores, dim=1)

        # [bs; h size]
        h_attn = torch.bmm(align.unsqueeze(1), h_children).squeeze(1)

        # [bs; n children; h size]
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']) + nodes.data['x_f'].unsqueeze(1))
        # [bs; h size]
        fc_sum = torch.sum(f * nodes.mailbox['c'], 1)

        return {
            'Uh_sum': self.U_iou(h_attn),  # name for using with super functions
            'fc_sum': fc_sum
        }

    def get_reduce_func(self):
        return self._reduce_func


class MultiHeadAttentionTreeLSTMCell(_ITreeLSTMCell):

    def __init__(self, x_size: int, h_size: int, n_heads: int, dropout: float = 0):
        super().__init__(x_size, h_size)
        assert x_size == h_size
        self.multihead_attention = nn.MultiheadAttention(x_size, n_heads, dropout)

        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size)
        self.U_f = nn.Linear(self.h_size, self.h_size)

    def get_message_func(self):
        return [dgl.function.copy_u('h', 'h'), dgl.function.copy_u('c', 'c')]

    def _reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        # [1, bs, x size]
        query = nodes.data['x'].unsqueeze(0)
        # [n children, bs, h size]
        key_value = nodes.mailbox['h']

        # [bs, h size]
        h_attn = self.multihead_attention(query, key_value, key_value)[0].squeeze(0)

        # [bs; 3 * h_size]
        h_iou = self.U_iou(h_attn)
        # [bs; h_size]
        h_f = self.U_f(h_attn)

        f = torch.sigmoid(nodes.data['x_f'] + h_f).unsqueeze(1)
        fc = nodes.mailbox['c'] * f
        return {
            'Uh_sum': h_iou,  # name for using with super functions
            'fc_sum': torch.sum(fc, 1)
        }

    def get_reduce_func(self):
        return self._reduce_func
