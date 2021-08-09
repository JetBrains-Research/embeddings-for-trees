import torch
from omegaconf import DictConfig

from embeddings_for_trees.models import TreeLSTM2Seq
from embeddings_for_trees.models.modules import TypedNodeEmbedding
from embeddings_for_trees.data.vocabulary import TypedVocabulary


class TypedTreeLSTM2Seq(TreeLSTM2Seq):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: TypedVocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__(model_config, optimizer_config, vocabulary, teacher_forcing)
        self._vocabulary: TypedVocabulary

    def _get_embedding(self) -> torch.nn.Module:
        return TypedNodeEmbedding(self._model_config, self._vocabulary)
