from typing import Dict

import dgl
import torch
from torch import nn

from model.encoder.treelstm import ITreeLSTMCell


class LuongAttentionTreeLSTMCell(ITreeLSTMCell):
    """
    align = softmax(hWx)
    [bs; h] * [h; x] * [x; 1] = [bs; 1]
    """

    name = "LuongAttention"

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)

        self.W = nn.Linear(self.h_size, self.x_size, bias=False)

    def get_message_func(self):
        def message_func(edges: dgl.EdgeBatch) -> Dict:
            # [bs; 1; x size]
            scores = self.W(edges.src['h']).unsqueeze(1)
            # [bs; 1]
            scores = torch.bmm(scores, edges.dst['x'].unsqueeze(2)).view(-1)

            return {
                'c': edges.src['c'],
                'scores': scores,
                'h': edges.src['h']
            }
        return message_func

    def _reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        # [bs; n_children]
        scores = nodes.mailbox['scores']
        # [bs * n children; 1]
        align = nn.functional.softmax(scores, dim=-1).view(-1, 1)

        h_shape = nodes.mailbox['h'].shape
        # [bs * n children; h size]
        h_attn = nodes.mailbox['h'].view(-1, h_shape[-1])
        h_attn = h_attn * align
        h_attn = h_attn.view(h_shape)

        # [bs; h size]
        h_iou = torch.sum(h_attn, 1)
        # [bs; n children, h size]
        h_f = self.U_f(h_attn)

        # [bs; n children; h size]
        fc = torch.sigmoid(h_f + nodes.data['x_f'].unsqueeze(1))
        fc = fc * nodes.mailbox['c']

        return {
            'Uh_sum': self.U_iou(h_iou),
            'fc_sum': torch.sum(fc, dim=1)
        }

    def get_reduce_func(self):
        return self._reduce_func


class SelfAttentionTreeLSTMCell(ITreeLSTMCell):

    name = "SelfAttention"

    def __init__(self, x_size: int, h_size: int, n_heads: int):
        super().__init__(x_size, h_size)
        self.mha = torch.nn.MultiheadAttention(self.h_size, n_heads)

        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size)
        self.U_f = nn.Linear(self.h_size, self.h_size)

    def get_message_func(self):
        def message_func(edges: dgl.EdgeBatch) -> Dict:
            f = torch.sigmoid(edges.dst['x_f'] + self.U_f(edges.src['h']))
            return {
                'fc': edges.src['c'] * f,
                'h': edges.src['h']
            }
        return message_func

    def _reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        # [1, bs, x size]
        query = nodes.data['x'].unsqueeze(0)
        # [n children, bs, h size]
        key_value = nodes.mailbox['h'].transpose(0, 1)

        # [bs, h size]
        h_attn = self.mha(query, key_value, key_value)[0].squeeze(0)
        fc_sum = torch.sum(nodes.mailbox['fc'], 1)

        return {
            'Uh_sum': self.U_iou(h_attn),  # name for using with super functions
            'fc_sum': fc_sum
        }

    def get_reduce_func(self):
        return self._reduce_func
