import dgl
import torch
from omegaconf import DictConfig
from torch import nn


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.W_iou = nn.Linear(config.embedding_size, 3 * config.encoder_size, bias=False)
        self.U_iou = nn.Linear(config.encoder_size, 3 * config.encoder_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * config.encoder_size))
        self.U_f = nn.Linear(config.encoder_size, config.encoder_size)

    @staticmethod
    def message_func(edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_tilde = torch.sum(nodes.mailbox["h"], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = torch.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tilde), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * torch.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(self, config: DictConfig):
        super(TreeLSTM, self).__init__()
        self._encoder_size = config.encoder_size
        self.dropout = nn.Dropout(config.encoder_dropout)
        self.linear = nn.Linear(config.encoder_size, config.decoder_size)
        self.cell = ChildSumTreeLSTMCell(config)

    def forward(self, graph: dgl.DGLGraph):
        n_nodes = graph.number_of_nodes()
        embeddings = graph.ndata["x"]
        graph.ndata["iou"] = self.cell.W_iou(self.dropout(embeddings))
        graph.ndata["h"] = embeddings.new_zeros((n_nodes, self._encoder_size))
        graph.ndata["c"] = embeddings.new_zeros((n_nodes, self._encoder_size))
        # propagate
        dgl.prop_nodes_topo(
            graph,
            message_func=self.cell.message_func,
            reduce_func=self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )
        # compute logits
        h = self.dropout(graph.ndata.pop("h"))
        out = self.linear(h)
        return out
