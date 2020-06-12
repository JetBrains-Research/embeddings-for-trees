from typing import List

import dgl
import numpy


def get_root_indexes(tree_sizes: List[int]) -> numpy.ndarray:
    """Get indexes of roots in given graph

    :param tree_sizes: list with tree sizes
    :return: list with indexes of roots [batch size]
    """
    idx_of_roots = numpy.cumsum([0] + tree_sizes)[:-1]
    return idx_of_roots


def get_tree_depth(tree: dgl.DGLGraph) -> int:
    return len(dgl.topological_nodes_generator(tree))
