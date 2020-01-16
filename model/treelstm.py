import torch
import torch.nn as nn
import dgl
from typing import Dict, Tuple

from model.encoder import _IEncoder


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = nn.Linear(h_size, h_size)
        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        return {'h': edges.src['h'], 'c': edges.src['c']}

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

    def process_batch(self, batch: dgl.BatchedDGLGraph, features: torch.Tensor, device: torch.device)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        # register function for message passing
        batch.register_message_func(self.message_func)
        batch.register_reduce_func(self.reduce_func)
        batch.register_apply_node_func(self.apply_node_func)

        nodes_in_batch = batch.number_of_nodes()
        batch.ndata['node_iou'] = self.W_iou(features) + self.b_iou
        batch.ndata['node_f'] = self.W_f(features) + self.b_f
        batch.ndata['h'] = torch.zeros(nodes_in_batch, self.h_size).to(device)
        batch.ndata['c'] = torch.zeros(nodes_in_batch, self.h_size).to(device)
        batch.ndata['Uh_tilda'] = torch.zeros(nodes_in_batch, 3 * self.h_size).to(device)
        # propagate
        dgl.prop_nodes_topo(batch)
        # get encoded output
        h = batch.ndata.pop('h')
        c = batch.ndata.pop('c')
        return h, c


class TokenTreeLSTM(_IEncoder):

    def __init__(self, x_size: int, h_size: int, dropout_prob: float = 0.) -> None:
        super().__init__()
        self.h_size = h_size
        self.cell = TreeLSTMCell(x_size, h_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, batch: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dropout_tokens = self.dropout(batch.ndata['token_embeds'])
        return self.cell.process_batch(batch, dropout_tokens, device)


class TokenTypeTreeLSTM(_IEncoder):

    def __init__(self, x_size: int, h_size: int, dropout_prob: float = 0.) -> None:
        super().__init__()
        self.h_size = h_size
        self.cell = TreeLSTMCell(h_size, h_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(2 * x_size, h_size)
        self.tanh = nn.Tanh()

    def forward(self, batch: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.tanh(
            self.linear(
                torch.cat([batch.ndata['token_embeds'], batch.ndata['type_embeds']], 1)
            )
        )
        dropout_tokens = self.dropout(features)
        return self.cell.process_batch(batch, dropout_tokens, device)
