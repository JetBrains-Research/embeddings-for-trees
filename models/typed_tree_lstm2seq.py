import torch
from omegaconf import DictConfig

from models import TreeLSTM2Seq
from models.parts import TypedNodeEmbedding
from utils.vocabulary import TypedVocabulary


class TypedTreeLSTM2Seq(TreeLSTM2Seq):
    def __init__(self, config: DictConfig, vocabulary: TypedVocabulary):
        super().__init__(config, vocabulary)
        self._vocabulary: TypedVocabulary

    def _get_embedding(self) -> torch.nn.Module:
        return TypedNodeEmbedding(self._config.model, self._vocabulary)
