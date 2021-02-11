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
            len(vocabulary.token_to_id), config.embedding_size, padding_idx=vocabulary.token_to_id[PAD]
        )
        self._node_embedding = nn.Embedding(
            len(vocabulary.node_to_id), config.embedding_size, padding_idx=vocabulary.node_to_id[PAD]
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # [n nodes; embedding size]
        token_embedding = self._token_embedding(graph.ndata[TOKEN])
        # [n nodes; embedding size]
        node_embedding = self._node_embedding(graph.ndata[NODE])
        # [n nodes; 2 * embedding size]
        return token_embedding + node_embedding
