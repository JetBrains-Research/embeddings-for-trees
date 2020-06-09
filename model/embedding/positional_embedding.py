from typing import Dict

import dgl
import torch

from embedding.node_embedding import INodeEmbedding


class PositionalEmbedding(INodeEmbedding):
    """Implement positional embedding from
    https://papers.nips.cc/paper/9376-novel-positional-encodings-to-enable-tree-based-transformers.pdf
    """

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

    def forward(self, graph: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
        """Forward pass for positional embedding

        @param graph: a batched graph with oriented edges from leaves to roots
        @param device: torch device
        @return: positional embedding [n_nodes, n * k]
        """
        pos_embeds = torch.zeros(graph.number_of_nodes(), self.h_emb, device=device)
        for layer in dgl.topological_nodes_generator(graph, reverse=True):
            for node in layer:
                children = graph.in_edges(node, form='uv')[0]
                pos_embeds[children, self.n:] = pos_embeds[node, :-self.n]
                pos_embeds[children, :self.n] = torch.eye(children.shape[0], self.n, device=device)
        # TODO: implement parametrized positional embedding with using p
        return pos_embeds
