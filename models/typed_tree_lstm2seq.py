import torch
from omegaconf import DictConfig

from models import TreeLSTM2Seq
from models.parts import TypedNodeEmbedding
from utils.vocabulary import Vocabulary


class TypedTreeLSTM2Seq(TreeLSTM2Seq):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__(config, vocabulary)

    def _get_embedding(self) -> torch.nn.Module:
        return TypedNodeEmbedding(self._config, self._vocabulary)
