import unittest
from typing import Tuple

import dgl
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model.treeLSTM_cell import ChildSumTreeLSTMCell
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
        w_iou: nn.Linear, u_iou: nn.Linear, b_iou: nn.Parameter,
        w_f: nn.Linear, u_f: nn.Linear, b_f: nn.Parameter
) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes_x = x[1:]
    iou_children = w_iou(nodes_x) + b_iou
    i, o, u = torch.chunk(iou_children, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
    c_children = i * u
    h_children = o * torch.tanh(c_children)

    x_root = x[0]
    h_sum = torch.sum(h_children, 0)
    iou_root = w_iou(x_root) + u_iou(h_sum) + b_iou
    i, o, u = torch.chunk(iou_root, 3, 1)
    i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
    f_root_child = torch.sigmoid(w_f(x_root) + u_f(h_children) + b_f)
    c_root_child = c_children * f_root_child
    c_root = i * u + torch.sum(c_root_child, 0)
    h_root = o * torch.tanh(c_root)

    h_calculated = torch.cat((h_root, h_children), 0)
    c_calculated = torch.cat((c_root, c_children), 0)

    return h_calculated, c_calculated


class TreeLSTMCellTest(unittest.TestCase):

    def test_childsum_tree_lstm_single_node(self):
        device = get_device()
        fix_seed()

        x_sizes = [5, 7, 10]
        h_sizes = [5, 7, 15]
        numbers_of_children = [4, 7, 13]

        for x_size, h_size, number_of_children in tqdm(zip(x_sizes, h_sizes, numbers_of_children), total=len(x_sizes)):
            tree_lstm_cell = ChildSumTreeLSTMCell(x_size, h_size)

            g = _gen_node_with_children(number_of_children)
            g.ndata['x'] = torch.rand(number_of_children + 1, x_size)

            h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)
            h_calculated, c_calculated = _calculate_childsum_tree_lstm_states(
                g.ndata['x'], tree_lstm_cell.W_iou, tree_lstm_cell.U_iou, tree_lstm_cell.b_iou,
                tree_lstm_cell.W_f, tree_lstm_cell.U_f, tree_lstm_cell.b_f
            )

            self.assertTrue(
                torch.allclose(h_tree_lstm, h_calculated),
                msg=f"Unequal hidden state tensors for ({x_size}, {h_size}, {number_of_children}) params"
            )
            self.assertTrue(
                torch.allclose(c_tree_lstm, c_calculated),
                msg=f"Unequal memory state tensors for ({x_size}, {h_size}, {number_of_children}) params"
            )

    def test_childsum_tree_lstm_batch(self):
        device = get_device()
        fix_seed()

        x_size = 5
        h_size = 5
        numbers_of_children = [7, 7]

        tree_lstm_cell = ChildSumTreeLSTMCell(x_size, h_size)

        g1 = _gen_node_with_children(numbers_of_children[0])
        g2 = _gen_node_with_children(numbers_of_children[1])
        g1.ndata['x'] = torch.rand(numbers_of_children[0] + 1, x_size)
        g2.ndata['x'] = torch.rand(numbers_of_children[1] + 1, x_size)
        g = dgl.batch([g1, g2])

        h_tree_lstm, c_tree_lstm = tree_lstm_cell(g, device)

        h1_calculated, c1_calculated = _calculate_childsum_tree_lstm_states(
            g1.ndata['x'], tree_lstm_cell.W_iou, tree_lstm_cell.U_iou, tree_lstm_cell.b_iou,
            tree_lstm_cell.W_f, tree_lstm_cell.U_f, tree_lstm_cell.b_f
        )
        h2_calculated, c2_calculated = _calculate_childsum_tree_lstm_states(
            g2.ndata['x'], tree_lstm_cell.W_iou, tree_lstm_cell.U_iou, tree_lstm_cell.b_iou,
            tree_lstm_cell.W_f, tree_lstm_cell.U_f, tree_lstm_cell.b_f
        )
        h_calculated = torch.cat([h1_calculated, h2_calculated], 0)
        c_calculated = torch.cat([c1_calculated, c2_calculated], 0)

        self.assertTrue(torch.allclose(h_tree_lstm, h_calculated), msg=f"Unequal hidden state tensors")
        self.assertTrue(torch.allclose(c_tree_lstm, c_calculated), msg=f"Unequal memory state tensors")


if __name__ == '__main__':
    unittest.main()
