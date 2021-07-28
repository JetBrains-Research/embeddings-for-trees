import dgl
import torch
from omegaconf import DictConfig
from torch import nn

from embeddings_for_trees.utils.common import TOKEN, NODE, TYPE
from embeddings_for_trees.data.vocabulary import Vocabulary, TypedVocabulary


class NodeEmbedding(nn.Module):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()

        self._token_embedding = nn.Embedding(
            len(vocabulary.token_to_id), config.embedding_size, padding_idx=vocabulary.token_to_id[vocabulary.PAD]
        )
        self._node_embedding = nn.Embedding(
            len(vocabulary.node_to_id), config.embedding_size, padding_idx=vocabulary.node_to_id[vocabulary.PAD]
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # [n nodes; embedding size]
        token_embedding = self._token_embedding(graph.ndata[TOKEN]).sum(1)
        # [n nodes; embedding size]
        node_embedding = self._node_embedding(graph.ndata[NODE])
        # [n nodes; 2 * embedding size]
        return token_embedding + node_embedding


class TypedNodeEmbedding(NodeEmbedding):
    def __init__(self, config: DictConfig, vocabulary: TypedVocabulary):
        super().__init__(config, vocabulary)
        self._type_embedding = nn.Embedding(
            len(vocabulary.type_to_id), config.embedding_size, padding_idx=vocabulary.type_to_id[vocabulary.PAD]
        )

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        token_emb = self._token_embedding(graph.ndata[TOKEN]).sum(1)
        type_emb = self._type_embedding(graph.ndata[TYPE]).sum(1)
        node_emb = self._node_embedding(graph.ndata[NODE])
        return token_emb + type_emb + node_emb
