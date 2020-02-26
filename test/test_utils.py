import dgl


def gen_node_with_children(number_of_children: int, edges_to_root: bool = True) -> dgl.DGLGraph:
    # create a graph with n+1 vertices and n edges (from 0 to 1, 2, ..., number_of_children)
    g = dgl.DGLGraph()
    g.add_nodes(number_of_children + 1)
    g.add_edges(0, [i for i in range(1, number_of_children + 1)])
    if edges_to_root:
        g = g.reverse()
    return g


def gen_tree(height: int, branch_factor: int, edges_to_root: bool = True) -> dgl.DGLGraph:
    """create full tree with given height and branching factor
    each level contains branch_factor^(level - 1) nodes, so total number of nodes is
    (bf^(level) - 1) / (bf - 1)
    node numeration from root to leaves, from the left child to right

    :param height: height of tree
    :param branch_factor: branch factor of tree
    :param edges_to_root: direct edges to root or not
    :return: dgl graph
    """
    g = dgl.DGLGraph()
    n_nodes = (branch_factor ** height - 1) // (branch_factor - 1)
    g.add_nodes(n_nodes)
    for u in range(n_nodes):
        if u * branch_factor + 1 >= n_nodes:
            continue
        v = [branch_factor * u + i for i in range(1, branch_factor + 1)]
        if edges_to_root:
            g.add_edges(v, u)
        else:
            g.add_edges(u, v)
    return g
