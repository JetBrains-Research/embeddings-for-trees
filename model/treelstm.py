import torch
import torch.nn as nn
import dgl
from typing import Dict


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = nn.Linear(h_size, h_size)
        self.W_f = nn.Linear(x_size, h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        h_sum = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(
            self.W_f(nodes.data['token_embeds']) + self.U_f(nodes.data['h']) + self.b_f
        )
        c = torch.sum(f * nodes.mailbox['c'], 1)
        iou = self.W_iou(nodes.data['token_embeds']) + self.U_iou(h_sum) + self.b_iou
        return {'iou': iou, 'c': c}

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        iou = nodes.data['iou']
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):

    def __init__(self, x_size: int, h_size: int) -> None:
        super().__init__()
        self.h_size = h_size
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, batch: dgl.BatchedDGLGraph) -> torch.Tensor:
        # register function for message passing
        batch.register_message_func(self.cell.message_func)
        batch.register_reduce_func(self.cell.reduce_func)
        batch.register_apply_node_func(self.cell.apply_node_func)
        # set hidden and memory state
        nodes_in_batch = batch.number_of_nodes()
        batch.ndata['iou'] = self.cell.W_iou(batch.ndata['token_embeds']) + self.cell.b_iou
        batch.ndata['h'] = torch.zeros(nodes_in_batch, self.h_size)
        batch.ndata['c'] = torch.zeros(nodes_in_batch, self.h_size)
        # propagate
        dgl.prop_nodes_topo(batch)
        # compute hidden state of roots
        h = batch.ndata.pop('h')
        return h
