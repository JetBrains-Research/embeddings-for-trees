import dgl


def gen_node_with_children(number_of_children: int, edges_to_root: bool = True) -> dgl.DGLGraph:
    # create a graph with n+1 vertices and n edges (from 0 to 1, 2, ..., number_of_children)
    g = dgl.DGLGraph()
    g.add_nodes(number_of_children + 1)
    g.add_edges(0, [i for i in range(1, number_of_children + 1)])
    if edges_to_root:
        g = g.reverse()
    return g
