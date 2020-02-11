from pickle import load as pkl_load
from typing import Dict, Tuple, Union

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

    def get_params(self) -> Dict:
        raise NotImplementedError


class EdgeChildSumTreeLSTMCell(_ITreeLSTMCell):
    """All calculations are happening in message function,
    reduce function only sum children features
    """

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)

        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        h_f = self.U_f(edges.src['h'])
        x_f = edges.dst['x_f']
        f = torch.sigmoid(x_f + h_f)
        return {
            'Uh': self.U_iou(edges.src['h']),
            'fc': edges.src['c'] * f
        }

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        """Using builtin functions"""
        raise NotImplementedError

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

        graph.register_message_func(self.message_func)
        graph.register_apply_node_func(self.apply_node_func)

        dgl.prop_nodes_topo(graph, reduce_func=[fn.sum('Uh', 'Uh_sum'), fn.sum('fc', 'fc_sum')])

        h = graph.ndata.pop('h')
        c = graph.ndata.pop('c')
        return h, c

    def get_params(self):
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.weight.t(), 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.weight.t(), 'b_f': self.b_f.data
        }


class NodeChildSumTreeLSTMCell(_ITreeLSTMCell):
    """All calculations are happening in reduce function
    message function only pass features from source to destination node
    """

    def __init__(self, x_size, h_size):
        super().__init__(x_size, h_size)
        self.W_iou = nn.Linear(self.x_size, 3 * self.h_size, bias=False)
        self.U_iou = nn.Linear(self.h_size, 3 * self.h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * self.h_size), requires_grad=True)

        self.W_f = nn.Linear(self.x_size, self.h_size, bias=False)
        self.U_f = nn.Linear(self.h_size, self.h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros((1, h_size)), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        """Using builtin functions"""
        raise NotImplementedError

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

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # register function for message passing
        graph.register_reduce_func(self.reduce_func)
        graph.register_apply_node_func(self.apply_node_func)

        features = graph.ndata['x']
        nodes_in_batch = graph.number_of_nodes()
        graph.ndata['node_iou'] = self.W_iou(features) + self.b_iou
        graph.ndata['node_f'] = self.W_f(features) + self.b_f
        graph.ndata['h'] = torch.zeros(nodes_in_batch, self.h_size).to(device)
        graph.ndata['c'] = torch.zeros(nodes_in_batch, self.h_size).to(device)
        graph.ndata['Uh_tilda'] = torch.zeros(nodes_in_batch, 3 * self.h_size).to(device)
        # propagate
        dgl.prop_nodes_topo(graph, message_func=[fn.copy_u('h', 'h'), fn.copy_u('c', 'c')])
        # get encoded output
        h = graph.ndata.pop('h')
        c = graph.ndata.pop('c')
        return h, c

    def get_params(self):
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.weight.t(), 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.weight.t(), 'b_f': self.b_f.data
        }


class EdgeSpecificTreeLSTMCell(EdgeChildSumTreeLSTMCell):

    def __init__(self, x_size, h_size, type_relationship: Union[str, Dict]):
        """Set matrix for each edge base on types of vertices
        :param type_relationship: if str, then path to pickle with Dict
        key: tuple with ids for src group, e.g. If and Switch statements
        value: list of lists, where each list corresponding to type ids of some group
        { ...
            (src_type_id_1, ..., src_type_id_n): [
                [dst_group_type_id_1, ..., dst_group_type_id_k],
                ...,
                [dst_group_type_id_1, ..., dst_group_type_id_m]
            ]
        ... }
        """
        super().__init__(x_size, h_size)
        if isinstance(type_relationship, str):
            with open(type_relationship, 'rb') as pkl_file:
                self.type_relationship = pkl_load(pkl_file)
        else:
            self.type_relationship = type_relationship
        count_diff_matrix = 1
        # dict of matrices ids, key: (src_type_id, dst_type_id), value: matrix_id
        self.edge_matrix_id = {}
        for type_ids, groups in self.type_relationship.items():
            for dst_group in groups:
                for child_id in dst_group:
                    for src_id in type_ids:
                        self.edge_matrix_id[(src_id, child_id)] = count_diff_matrix
                count_diff_matrix += 1
        print(count_diff_matrix)

        self.U_f = nn.Parameter(torch.rand(count_diff_matrix, self.h_size, self.h_size), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        _W = self.U_f[edges.data['matrix_id']]
        _x = edges.src['h'].unsqueeze(1)
        h_f = torch.bmm(_x, _W).squeeze(1)
        x_f = edges.dst['x_f']
        f = torch.sigmoid(x_f + h_f)
        return {
            'Uh': self.U_iou(edges.src['h']),
            'fc': edges.src['c'] * f
        }

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        return super().reduce_func(nodes)

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        return super().apply_node_func(nodes)

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix_id = torch.zeros(graph.number_of_edges(), dtype=torch.long)
        # because all edges in graph reversed (from child to parent)
        for edge_dst, edge_src, edge_id in zip(*graph.all_edges('all')):
            src_type = graph.ndata['type_id'][edge_src].item()
            dst_type = graph.ndata['type_id'][edge_dst].item()
            matrix_id[edge_id.item()] = self.edge_matrix_id.get((src_type, dst_type), 0)
        graph.edata['matrix_id'] = matrix_id.to(device)
        return super().forward(graph, device)

    def get_params(self) -> Dict:
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.weight.t(), 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.data, 'b_f': self.b_f.data
        }


class TypeSpecificTreeLSTMCell(EdgeChildSumTreeLSTMCell):

    def __init__(self, x_size: int, h_size: int, nary_types: Union[str, Dict]):
        """Use NAry cell type for given types
        :param nary_types: if str, then path to pickle with Dict
        contains information about types for NAry equations:
        { ...
            src_type_id: [
                ...
                [dst_type_id_1, ..., dst_type_id_k],
                ...
            ]
        ... }
        """
        super().__init__(x_size, h_size)
        if isinstance(nary_types, str):
            with open(nary_types, 'rb') as pkl_file:
                self.nary_types = pkl_load(pkl_file)
        else:
            self.nary_types = nary_types
        count_diff_matrix = 1
        # dict of matrices ids, key: src_type_id, value: {(dst_type_id_1, ..., dst_type_id_k): (m_id_1, ..., m_id_k))}
        self.edge_matrix_id = {}
        for src_type_id, dst_type_ids in self.nary_types.items():
            if src_type_id not in self.edge_matrix_id:
                self.edge_matrix_id[src_type_id] = {}
            cur_indexes = torch.tensor(range(count_diff_matrix, count_diff_matrix + len(dst_type_ids[0])))
            for dst_type_id in dst_type_ids:
                self.edge_matrix_id[src_type_id][tuple(sorted(dst_type_id))] = cur_indexes
            count_diff_matrix += len(dst_type_ids[0])

        self.U_f = nn.Parameter(torch.rand(count_diff_matrix, self.h_size, self.h_size), requires_grad=True)

    def message_func(self, edges: dgl.EdgeBatch) -> Dict:
        _W = self.U_f[edges.data['matrix_id']]
        _x = edges.src['h'].unsqueeze(1)
        h_f = torch.bmm(_x, _W).squeeze(1)
        x_f = edges.dst['x_f']
        f = torch.sigmoid(x_f + h_f)
        return {
            'Uh': self.U_iou(edges.src['h']),
            'fc': edges.src['c'] * f
        }

    def reduce_func(self, nodes: dgl.NodeBatch) -> Dict:
        return super().reduce_func(nodes)

    def apply_node_func(self, nodes: dgl.NodeBatch) -> Dict:
        return super().apply_node_func(nodes)

    def forward(self, graph: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix_id = torch.zeros(graph.number_of_edges(), dtype=torch.long)
        # because all edges in graph reversed (from child to parent)
        for node in range(graph.number_of_nodes()):
            children, _, edge_ids = graph.in_edges(node, form='all')
            children_ids = tuple(sorted(graph.ndata['type_id'][children].tolist()))
            if node in self.edge_matrix_id:
                if children_ids in self.edge_matrix_id[node]:
                    matrix_id[edge_ids] = self.edge_matrix_id[node][children_ids]

        graph.edata['matrix_id'] = matrix_id.to(device)
        return super().forward(graph, device)

    def get_params(self) -> Dict:
        return {
            'w_iou': self.W_iou.weight, 'u_iou': self.U_iou.weight.t(), 'b_iou': self.b_iou.data,
            'w_f': self.W_f.weight, 'u_f': self.U_f.data, 'b_f': self.b_f.data
        }


def get_tree_lstm_cell(tree_lstm_type: str) -> _ITreeLSTMCell:
    tree_lstm_cells = {
        EdgeChildSumTreeLSTMCell.__name__: EdgeChildSumTreeLSTMCell,
        NodeChildSumTreeLSTMCell.__name__: NodeChildSumTreeLSTMCell,
        EdgeSpecificTreeLSTMCell.__name__: EdgeSpecificTreeLSTMCell,
        TypeSpecificTreeLSTMCell.__name__: TypeSpecificTreeLSTMCell
    }
    if tree_lstm_type not in tree_lstm_cells:
        raise ValueError(f"unknown tree lstm cell: {tree_lstm_type}")
    return tree_lstm_cells[tree_lstm_type]
