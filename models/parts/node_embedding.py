import dgl
import torch
from omegaconf import DictConfig
from torch import nn

from utils.common import PAD, TOKEN, NODE
from utils.vocabulary import Vocabulary


class NodeFeaturesEmbedding(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()

        self._token_embedding = nn.Embedding(
            len(vocabulary.token_to_id), config.tree_lstm.embedding_size, padding_idx=vocabulary.token_to_id[PAD]
        )
        self._node_embedding = nn.Embedding(
            len(vocabulary.node_to_id), config.tree_lstm.embedding_size, padding_idx=vocabulary.node_to_id[PAD]
        )
        self._concat_linear = nn.Linear(2 * config.tree_lstm.embedding_size, config.tree_lstm.embedding_size)

    def forward(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        # [n nodes; embedding size]
        token_embedding = self._token_embedding(graph.ndata[TOKEN]).sum(1)
        # [n nodes; embedding size]
        node_embedding = self._node_embedding(graph.ndata[NODE])
        # [n nodes; 2 * embedding size]
        concat = torch.cat([token_embedding, node_embedding], dim=1)
        # [n nodes; embedding size]
        graph.ndata["x"] = self._concat_linear(concat)
        return graph
