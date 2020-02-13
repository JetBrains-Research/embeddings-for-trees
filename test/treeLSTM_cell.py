import unittest
from math import sqrt
from typing import Tuple, Union, Dict

import dgl
import torch

from model.treeLSTM_cell import EdgeChildSumTreeLSTMCell, NodeChildSumTreeLSTMCell, EdgeSpecificTreeLSTMCell, \
    TypeSpecificTreeLSTMCell, TypeAttentionTreeLSTMCell
from utils.common import get_device, fix_seed

ATOL = 1e-6


def _gen_node_with_children(number_of_children: int, edges_to_root: bool = True) -> dgl.DGLGraph:
    # create a graph with n+1 vertices and n edges (from 0 to 1, 2, ..., number_of_children)
    g = dgl.DGLGraph()
    g.add_nodes(number_of_children + 1)
    g.add_edges(0, [i for i in range(1, number_of_children + 1)])
    if edges_to_root:
        g = g.reverse()
    return g


def _calculate_childsum_tree_lstm_states(
        x: torch.Tensor,
        w_iou: torch.Tensor, u_iou: torch.Tensor, b_iou: torch.Tensor,
        w_f: torch.Tensor, u_f: torch.Tensor, b_f: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """compute hidden and memory states for the node and its children

    :param x: features, x[0] corresponds to parent, other for children [number_of_children + 1, x_size]
    :param w_iou: [3 * h_size, x_size]
    :param u_iou: [h_size, 3 * h_size]
    :param b_iou: [1, 3 * h_size]
    :param w_f: [3 * h_size, x_size]
    :param u_f: [h_size, h_size]
    :param b_f: [1, h_size]
    :return: hidden state: [number_of_children + 1, h_size], memory state: [number_of_children + 1, h_size]
    """
    nodes_x = x[1:]
    iou_children = nodes_x.matmul(w_iou.t()) + b_iou
    i, o, u = torch.chunk(iou_children, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
    c_children = i * u
    h_children = o * torch.tanh(c_children)

    x_root = x[0]
    h_sum = torch.sum(h_children, 0)
    iou_root = x_root.matmul(w_iou.t()) + h_sum.matmul(u_iou) + b_iou
    i, o, u = torch.chunk(iou_root, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
    f_root_child = torch.sigmoid(x_root.matmul(w_f.t()) + h_children.matmul(u_f) + b_f)
    c_root_child = c_children * f_root_child
    c_root = i * u + torch.sum(c_root_child, 0)
    h_root = o * torch.tanh(c_root)

    h_calculated = torch.cat((h_root, h_children), 0)
    c_calculated = torch.cat((c_root, c_children), 0)

    return h_calculated, c_calculated


def _calculate_nary_tree_lstm_states(
        x: torch.Tensor,
        w_iou: torch.Tensor, u_iou: torch.Tensor, b_iou: torch.Tensor,
        w_f: torch.Tensor, u_f: torch.Tensor, b_f: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """compute hidden and memory states for the node and its children

    :param x: features, x[0] corresponds to parent, other for children [number_of_children + 1, x_size]
    :param w_iou: [3 * h_size, x_size]
    :param u_iou: [h_size, 3 * h_size]
    :param b_iou: [1, 3 * h_size]
    :param w_f: [3 * h_size, x_size]
    :param u_f: [number_of_children, h_size, h_size]
    :param b_f: [1, h_size]
    :return: hidden state: [number_of_children + 1, h_size], memory state: [number_of_children + 1, h_size]
    """
    nodes_x = x[1:]
    iou_children = nodes_x.matmul(w_iou.t()) + b_iou
    i, o, u = torch.chunk(iou_children, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
    c_children = i * u
    h_children = o * torch.tanh(c_children)

    x_root = x[0]
    h_sum = torch.sum(h_children, 0)
    iou_root = x_root.matmul(w_iou.t()) + h_sum.matmul(u_iou) + b_iou
    i, o, u = torch.chunk(iou_root, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

    # [number_of_children, h_size]
    f_root_child = torch.zeros(nodes_x.shape[0], u_f.shape[2])
    for child in range(nodes_x.shape[0]):
        f_cur = torch.sigmoid(x_root.matmul(w_f.t()) + h_children[child].matmul(u_f[child]) + b_f)
        f_root_child[child] = f_cur

    c_root_child = c_children * f_root_child
    c_root = i * u + torch.sum(c_root_child, 0)
    h_root = o * torch.tanh(c_root)

    h_calculated = torch.cat((h_root, h_children), 0)
    c_calculated = torch.cat((c_root, c_children), 0)

    return h_calculated, c_calculated


class TreeLSTMCellTest(unittest.TestCase):
    x_sizes = [5, 7, 10, 128, 256]
    h_sizes = [5, 7, 15, 128, 128]
    numbers_of_children = [4, 7, 13, 10, 20]

    def _state_assert(self, h_tree_lstm: torch.Tensor, c_tree_lstm: torch.tensor,
                      h_calculated: torch.Tensor, c_calculated: torch.Tensor, atol: float = ATOL):
        self.assertTrue(torch.allclose(h_tree_lstm, h_calculated, atol=atol), msg=f"Unequal hidden state tensors")
        self.assertTrue(torch.allclose(c_tree_lstm, c_calculated, atol=atol), msg=f"Unequal memory state tensors")

    def _test_childsum(self, tree_lstm_type: Union[NodeChildSumTreeLSTMCell, EdgeChildSumTreeLSTMCell], x_size: int,
                       h_size: int, number_of_children: int, device: torch.device
                       ) -> None:
        tree_lstm_cell = tree_lstm_type(x_size, h_size)

        g = _gen_node_with_children(number_of_children)
        g.ndata['x'] = torch.rand(number_of_children + 1, x_size)

        h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)
        h_calculated, c_calculated = _calculate_childsum_tree_lstm_states(
            g.ndata['x'], **tree_lstm_cell.get_params()
        )

        self._state_assert(h_tree_lstm, c_tree_lstm, h_calculated, c_calculated)

    def _test_childsum_tree_lstm_cell(self, tree_lstm_type):
        device = get_device()
        fix_seed()
        for i in range(len(self.x_sizes)):
            x_size, h_size, number_of_children = self.x_sizes[i], self.h_sizes[i], self.numbers_of_children[i]
            with self.subTest(i=i):
                self._test_childsum(
                    tree_lstm_type, x_size, h_size, number_of_children, device
                )

    def test_node_childsum_tree_lstm_cell(self):
        self._test_childsum_tree_lstm_cell(NodeChildSumTreeLSTMCell)

    def test_edge_childsum_tree_lstm_cell(self):
        self._test_childsum_tree_lstm_cell(EdgeChildSumTreeLSTMCell)

    def test_childsum_tree_lstm_batch(self):
        device = get_device()
        fix_seed()

        x_size = 5
        h_size = 5
        numbers_of_children = [7, 7]

        tree_lstm_types = [EdgeChildSumTreeLSTMCell, NodeChildSumTreeLSTMCell]
        for tree_lstm_type in tree_lstm_types:
            with self.subTest(msg=f"test {tree_lstm_type.__name__} tree lstm cell"):
                tree_lstm_cell = tree_lstm_type(x_size, h_size)

                g1 = _gen_node_with_children(numbers_of_children[0])
                g2 = _gen_node_with_children(numbers_of_children[1])
                g1.ndata['x'] = torch.rand(numbers_of_children[0] + 1, x_size)
                g2.ndata['x'] = torch.rand(numbers_of_children[1] + 1, x_size)
                g = dgl.batch([g1, g2])

                h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)

                h1_calculated, c1_calculated = _calculate_childsum_tree_lstm_states(
                    g1.ndata['x'], **tree_lstm_cell.get_params()
                )
                h2_calculated, c2_calculated = _calculate_childsum_tree_lstm_states(
                    g2.ndata['x'], **tree_lstm_cell.get_params()
                )
                h_calculated = torch.cat([h1_calculated, h2_calculated], 0)
                c_calculated = torch.cat([c1_calculated, c2_calculated], 0)

                self._state_assert(h_tree_lstm, c_tree_lstm, h_calculated, c_calculated)

    def test_edge_specific_tree_lstm_cell(self):
        device = get_device()
        fix_seed()

        for i, (x_size, h_size, number_of_children) in enumerate(
                zip(self.x_sizes, self.h_sizes, self.numbers_of_children)):
            with self.subTest(i=i):
                g = _gen_node_with_children(number_of_children)
                g.ndata['x'] = torch.rand(number_of_children + 1, x_size)
                g.ndata['type_id'] = torch.tensor(range(0, number_of_children + 1))
                type_relationship = {
                    (0,): [list(range(1, number_of_children // 2))]
                }

                tree_lstm_cell = EdgeSpecificTreeLSTMCell(x_size, h_size, type_relationship)

                h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)

                tree_lstm_cell_params = tree_lstm_cell.get_params()
                u_f_indices = [
                    tree_lstm_cell.edge_matrix_id.get((0, i), 0) for i in range(1, number_of_children + 1)
                ]
                tree_lstm_cell_params['u_f'] = tree_lstm_cell_params['u_f'][u_f_indices]
                h_calculated, c_calculated = _calculate_nary_tree_lstm_states(g.ndata['x'], **tree_lstm_cell_params)
                self._state_assert(h_tree_lstm, c_tree_lstm, h_calculated, c_calculated)

    def _test_type_specific_tree_lstm_cell(self,
                                           x_size: int, h_size: int, number_of_children: int,
                                           g: dgl.DGLGraph, nary_types: Dict, device: torch.device):
        tree_lstm_cell = TypeSpecificTreeLSTMCell(x_size, h_size, nary_types)

        h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)

        tree_lstm_cell_params = tree_lstm_cell.get_params()
        children = list(range(1, number_of_children + 1))
        u_f_indices = [
            tree_lstm_cell.edge_matrix_id.get(0, {}).get(tuple(children), [0 for _ in children])
        ]
        tree_lstm_cell_params['u_f'] = tree_lstm_cell_params['u_f'][u_f_indices]
        h_calculated, c_calculated = _calculate_nary_tree_lstm_states(g.ndata['x'], **tree_lstm_cell_params)
        self._state_assert(h_tree_lstm, c_tree_lstm, h_calculated, c_calculated)

    def test_type_specific_tree_lstm_cell_with_types(self):
        device = get_device()
        fix_seed()

        for i, (x_size, h_size, number_of_children) in enumerate(
                zip(self.x_sizes, self.h_sizes, self.numbers_of_children)):
            with self.subTest(i=i):
                g = _gen_node_with_children(number_of_children)
                g.ndata['x'] = torch.rand(number_of_children + 1, x_size)
                g.ndata['type_id'] = torch.tensor(range(0, number_of_children + 1))
                nary_types = {
                    0: [list(range(1, number_of_children)), [1, 2], [10, 11, 12]]
                }
                self._test_type_specific_tree_lstm_cell(x_size, h_size, number_of_children, g, nary_types, device)

    def test_type_specific_tree_lstm_cell_without_types(self):
        device = get_device()
        fix_seed()

        for i, (x_size, h_size, number_of_children) in enumerate(
                zip(self.x_sizes, self.h_sizes, self.numbers_of_children)):
            with self.subTest(i=i):
                g = _gen_node_with_children(number_of_children)
                g.ndata['x'] = torch.rand(number_of_children + 1, x_size)
                g.ndata['type_id'] = torch.tensor(range(0, number_of_children + 1))
                nary_types = {
                    1: [[2, 3, 4], [5, 6, 7]]
                }
                self._test_type_specific_tree_lstm_cell(x_size, h_size, number_of_children, g, nary_types, device)

    def test_type_specific_tree_lstm_cell_complex(self):
        device = get_device()
        fix_seed()

        x_size = 5
        h_size = 5
        number_of_children = [3, 5]

        g1 = _gen_node_with_children(number_of_children[0])
        g1.ndata['x'] = torch.rand(number_of_children[0] + 1, x_size)
        g1.ndata['type_id'] = torch.tensor(range(0, number_of_children[0] + 1))
        g2 = _gen_node_with_children(number_of_children[1])
        g2.ndata['x'] = torch.rand(number_of_children[1] + 1, x_size)
        g2.ndata['type_id'] = torch.tensor(
            range(number_of_children[0] + 1, number_of_children[0] + number_of_children[1] + 2)
        )

        # only g1 root node match this type
        nary_types = {
            0: [
                [1, 2, 3], [4, 5, 6]
            ],
            1: [
                [1, 2, 3], [4, 5, 6]
            ]
        }

        g_full = dgl.batch([g1, g2])

        tree_lstm_cell = TypeSpecificTreeLSTMCell(x_size, h_size, nary_types)

        h_tree_lstm, c_tree_lstm = tree_lstm_cell(g_full, device)

        h_calculated_list = []
        c_calculated_list = []
        for g, i in zip([g1, g2], number_of_children):
            tree_lstm_cell_params = tree_lstm_cell.get_params()
            children = tuple(g.ndata['type_id'][1:].tolist())
            root_id = g.ndata['type_id'][0].item()
            u_f_indices = [
                tree_lstm_cell.edge_matrix_id.get(root_id, {}).get(tuple(children), [0 for _ in children])
            ]
            tree_lstm_cell_params['u_f'] = tree_lstm_cell_params['u_f'][u_f_indices]
            h_calculated, c_calculated = _calculate_nary_tree_lstm_states(g.ndata['x'], **tree_lstm_cell_params)
            h_calculated_list.append(h_calculated)
            c_calculated_list.append(c_calculated)

        h_calculated = torch.cat(h_calculated_list, 0)
        c_calculated = torch.cat(c_calculated_list, 0)

        self._state_assert(h_tree_lstm, c_tree_lstm, h_calculated, c_calculated)

    def test_type_attention_tree_lstm_cell(self):
        device = get_device()
        fix_seed()

        a_sizes = [5, 8, 13, 128, 256]
        for i, (x_size, h_size, number_of_children, a_size) in enumerate(
                zip(self.x_sizes, self.h_sizes, self.numbers_of_children, a_sizes)
        ):
            with self.subTest(i=i):
                g = _gen_node_with_children(number_of_children)
                g.ndata['x'] = torch.rand(number_of_children + 1, x_size, dtype=torch.float32, device=device)
                g.ndata['type_embeds'] = torch.rand(number_of_children + 1, x_size, dtype=torch.float32, device=device)

                tree_lstm_cell = TypeAttentionTreeLSTMCell(x_size, h_size, a_size).to(device)
                h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)

                params = tree_lstm_cell.get_params()

                nodes_x = g.ndata['x'][1:]
                iou_children = nodes_x.matmul(params['w_iou'].t()) + params['b_iou']
                i, o, u = torch.chunk(iou_children, 3, 1)
                i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
                c_children = i * u
                h_children = o * torch.tanh(c_children)

                x_root = g.ndata['x'][0]
                type_root = g.ndata['type_embeds'][0]
                type_children = g.ndata['type_embeds'][1:]

                _Q = type_root.matmul(params['w_query'].t())
                _K = type_children.matmul(params['w_key'].t())
                _V = h_children.matmul(params['w_value'].t())
                align = _Q.matmul(_K.t()) / sqrt(a_size)
                a = torch.softmax(align - torch.max(align), 0)
                h_attn = a.matmul(_V)

                iou_root = x_root.matmul(params['w_iou'].t()) + h_attn + params['b_iou']
                i, o, u = torch.chunk(iou_root, 3, 1)
                i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
                f_root_child = torch.sigmoid(
                    x_root.matmul(params['w_f'].t()) + h_children.matmul(params['u_f']) + params['b_f']
                )
                c_root_child = c_children * f_root_child
                c_root = i * u + torch.sum(c_root_child, 0)
                h_root = o * torch.tanh(c_root)

                h_calculated = torch.cat((h_root, h_children), 0)
                c_calculated = torch.cat((c_root, c_children), 0)

                print(h_tree_lstm)
                print(h_calculated)

                self._state_assert(h_tree_lstm, c_tree_lstm, h_calculated, c_calculated)


if __name__ == '__main__':
    unittest.main()
