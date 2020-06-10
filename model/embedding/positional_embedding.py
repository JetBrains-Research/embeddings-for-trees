from typing import Dict

import dgl
import torch

from model.embedding import INodeEmbedding


class PositionalEmbedding(INodeEmbedding):
    """Implement positional embedding from
    https://papers.nips.cc/paper/9376-novel-positional-encodings-to-enable-tree-based-transformers.pdf
    """

    name = "positional"

    def __init__(self, token_to_id: Dict, type_to_id: Dict, h_emb: int, n: int, k: int, p: float = 1.) -> None:
        """

        :param n: the degree of tree
        :param k: the depth of tree
        :param p: regularization (Not Implemented)
        """
        super().__init__(token_to_id, type_to_id, h_emb)
        assert n * k == self.h_emb, f"n * k should be equal to the size of hidden state ({n} * {k} != {self.h_emb}"
        self.n, self.k, self.p = n, k, p
        self.p_emb = torch.tensor([self.p ** i for i in range(self.h_emb)])

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Forward pass for positional embedding

        @param graph: a batched graph with oriented edges from leaves to roots
        @return: positional embedding [n_nodes, n * k]
        """
        pos_embeds = graph.ndata['x'].new_zeros((graph.number_of_nodes(), self.h_emb))
        for layer in dgl.topological_nodes_generator(graph, reverse=True):
            for node in layer:
                children = graph.in_edges(node, form='uv')[0]
                pos_embeds[children, self.n:] = pos_embeds[node, :-self.n]
                eye_tensor = graph.ndata["x"].new_zeros((children.shape[0], self.n))
                diag_range = torch.arange(0, min(children.shape[0], self.n), dtype=torch.long)
                eye_tensor[diag_range, diag_range] = 1
                pos_embeds[children, :self.n] = eye_tensor
        # TODO: implement parametrized positional embedding with using p
        return pos_embeds
