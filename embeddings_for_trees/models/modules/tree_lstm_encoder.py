from typing import Dict

import dgl
import torch
from dgl.udf import EdgeBatch, NodeBatch
from omegaconf import DictConfig
from torch import nn


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges: EdgeBatch):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes: NodeBatch):
        h_tild = torch.sum(nodes.mailbox["h"], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes: NodeBatch):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self._encoder_size = config.encoder_size
        self._dropout = nn.Dropout(config.encoder_dropout)
        self._linear = nn.Linear(config.encoder_size, config.decoder_size)
        self._cell = ChildSumTreeLSTMCell(config.embedding_size, config.encoder_size)

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        x = self._dropout(graph.ndata["x"])

        # init matrices for message propagation
        number_of_nodes = graph.number_of_nodes()
        graph.ndata["iou"] = self._cell.W_iou(x)
        graph.ndata["h"] = graph.ndata["x"].new_zeros((number_of_nodes, self._encoder_size))
        graph.ndata["c"] = graph.ndata["x"].new_zeros((number_of_nodes, self._encoder_size))

        # propagate nodes
        dgl.prop_nodes_topo(
            graph, self._cell.message_func, self._cell.reduce_func, apply_node_func=self._cell.apply_node_func
        )

        # [n nodes; encoder size]
        h = self._dropout(graph.ndata.pop("h"))
        # [n nodes; decoder size]
        out = self._linear(h)
        return out
