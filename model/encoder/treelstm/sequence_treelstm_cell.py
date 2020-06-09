from typing import Dict

import dgl
import torch
from torch import nn

from model.encoder.treelstm import ITreeLSTMCell


class SequenceTreeLSTMCell(ITreeLSTMCell):

    name = "Sequence"

    def __init__(self, x_size: int, h_size: int, bidirectional: bool = True):
        super().__init__(x_size, h_size)
        self.lstm = nn.LSTM(self.h_size, self.h_size, num_layers=1, bidirectional=bidirectional)

        if bidirectional:
            self.U_iou = nn.Linear(2 * self.h_size, 3 * self.h_size, bias=False)
            self.U_f = nn.Linear(2 * self.h_size, self.h_size, bias=False)
        else:
            self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
            self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)

    def get_message_func(self):
        return [dgl.function.copy_u('h', 'h'), dgl.function.copy_u('c', 'c')]

    def get_reduce_func(self):
        def reduce_func(nodes: dgl.NodeBatch) -> Dict:
            # [n children; bs; h size]
            h = nodes.mailbox['h'].transpose(0, 1)

            # [n children; bs; h size * 1 or 2]
            h_lstm = self.lstm(h)[0]
            # [bs; n children; h size]
            h_f = self.U_f(h_lstm).transpose(0, 1)
            # [bs; 3 * h size]
            h_iou = self.U_iou(h_lstm[-1])

            # [bs; n children; h size]
            fc = torch.sigmoid(h_f + nodes.data['x_f'].unsqueeze(1))
            fc = fc * nodes.mailbox['c']

            return {
                'Uh_sum': h_iou,
                'fc_sum': torch.sum(fc, dim=1)
            }
        return reduce_func
