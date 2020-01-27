import unittest
from typing import Tuple, Union

import dgl
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model.treeLSTM_cell import EdgeChildSumTreeLSTMCell, NodeChildSumTreeLSTMCell
from utils.common import get_device, fix_seed


def _gen_node_with_children(number_of_children: int, edges_to_root: bool = True) -> dgl.DGLGraph:
    # create a graph with n+1 vertices and n+1 edges (from 0 to 1, 2, ..., number_of_children)
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


def _test_childsum(tree_lstm_type: Union[NodeChildSumTreeLSTMCell, EdgeChildSumTreeLSTMCell], x_size: int,
                   h_size: int, number_of_children: int, device: torch.device
                   ) -> Tuple[bool, bool]:
    tree_lstm_cell = tree_lstm_type(x_size, h_size)

    g = _gen_node_with_children(number_of_children)
    g.ndata['x'] = torch.rand(number_of_children + 1, x_size)

    h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)
    h_calculated, c_calculated = _calculate_childsum_tree_lstm_states(
        g.ndata['x'], **tree_lstm_cell.get_params()
    )

    return torch.allclose(h_tree_lstm, h_calculated, atol=1e-6), torch.allclose(c_tree_lstm, c_calculated, atol=1e-6)


class TreeLSTMCellTest(unittest.TestCase):

    x_sizes = [5, 7, 10, 128, 256]
    h_sizes = [5, 7, 15, 128, 128]
    numbers_of_children = [4, 7, 13, 10, 20]

    def _test_childsum_tree_lstm_cell(self, tree_lstm_type): 
        device = get_device()
        fix_seed()
        for i in range(len(self.x_sizes)):
            x_size, h_size, number_of_children = self.x_sizes[i], self.h_sizes[i], self.numbers_of_children[i]
            with self.subTest(i=i):
                h_equal, c_equal = _test_childsum(
                    tree_lstm_type, x_size, h_size, number_of_children, device
                )
                self.assertTrue(
                    h_equal, msg=f"Unequal hidden state tensors for ({x_size}, {h_size}, {number_of_children}) params"
                )
                self.assertTrue(
                    c_equal, msg=f"Unequal memory state tensors for ({x_size}, {h_size}, {number_of_children}) params"
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

                self.assertTrue(torch.allclose(h_tree_lstm, h_calculated), msg=f"Unequal hidden state tensors")
                self.assertTrue(torch.allclose(c_tree_lstm, c_calculated), msg=f"Unequal memory state tensors")


if __name__ == '__main__':
    unittest.main()
