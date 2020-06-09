from typing import Dict

import dgl
import torch
from torch import nn

from model.encoder.treelstm import ITreeLSTMCell


class ConvolutionalTreeLSTMCell(ITreeLSTMCell):

    name = "Convolutional"

    def __init__(self, x_size: int, h_size: int, kernel: int):
        assert kernel % 2 == 1, f"Kernel size should be odd for keeping the same size after convolutional"
        super().__init__(x_size, h_size)
        padding = (kernel - 1) // 2
        self.convolution = nn.Conv2d(1, 1, kernel, padding=padding)

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
        # [bs; 1; n children; h size]
        h = nodes.mailbox['h'].unsqueeze(1)
        # [bs; h size]
        h = self.convolution(h).max(2)[0].squeeze(1)

        return {
            'Uh_sum': self.U_iou(h),
            'fc_sum': nodes.mailbox['fc'].sum(1)
        }

    def get_reduce_func(self):
        return self._reduce_func
