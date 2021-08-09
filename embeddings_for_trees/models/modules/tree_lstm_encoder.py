from typing import Dict

import dgl
import torch
from omegaconf import DictConfig
from torch import nn


class TreeLSTM(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self._encoder_size = config.encoder_size

        self._dropout = nn.Dropout(config.encoder_dropout)

        self._W_iou = nn.Linear(config.embedding_size, 3 * config.encoder_size)
        self._U_iou = nn.Linear(config.encoder_size, 3 * config.encoder_size, bias=False)

        self._W_f = nn.Linear(config.embedding_size, config.encoder_size)
        self._U_f = nn.Linear(config.encoder_size, config.encoder_size, bias=False)

        self._out_linear = nn.Linear(config.encoder_size, config.decoder_size)
        self._norm = nn.LayerNorm(config.decoder_size)
        self._tanh = nn.Tanh()

    def message_func(self, edges: dgl.udf.EdgeBatch) -> Dict:
        h_f = self._U_f(edges.src["h"])
        x_f = edges.dst["x_f"]
        f = torch.sigmoid(x_f + h_f)
        return {"Uh": self._U_iou(edges.src["h"]), "fc": edges.src["c"] * f}

    @staticmethod
    def reduce_func(nodes: dgl.udf.NodeBatch) -> Dict:
        return {"Uh_sum": torch.sum(nodes.mailbox["Uh"], dim=1), "fc_sum": torch.sum(nodes.mailbox["fc"], dim=1)}

    @staticmethod
    def apply_node_func(nodes: dgl.udf.NodeBatch) -> Dict:
        iou = nodes.data["x_iou"] + nodes.data["Uh_sum"]
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = i * u + nodes.data["fc_sum"]
        h = o * torch.tanh(c)

        return {"h": h, "c": c}

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        x = self._dropout(graph.ndata["x"])

        # init matrices for message propagation
        number_of_nodes = graph.number_of_nodes()
        graph.ndata["x_iou"] = self._W_iou(x)
        graph.ndata["x_f"] = self._W_f(x)
        graph.ndata["h"] = graph.ndata["x"].new_zeros((number_of_nodes, self._encoder_size))
        graph.ndata["c"] = graph.ndata["x"].new_zeros((number_of_nodes, self._encoder_size))
        graph.ndata["Uh_sum"] = graph.ndata["x"].new_zeros((number_of_nodes, 3 * self._encoder_size))
        graph.ndata["fc_sum"] = graph.ndata["x"].new_zeros((number_of_nodes, self._encoder_size))

        # propagate nodes
        dgl.prop_nodes_topo(
            graph,
            message_func=self.message_func,
            reduce_func=self.reduce_func,
            apply_node_func=self.apply_node_func,
        )

        # [n nodes; encoder size]
        h = graph.ndata.pop("h")
        # [n nodes; decoder size]
        out = self._tanh(self._norm(self._out_linear(h)))
        return out
