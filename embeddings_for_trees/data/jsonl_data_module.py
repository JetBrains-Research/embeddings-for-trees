from os import path
from os.path import basename
from typing import List, Optional, Tuple

import dgl
import torch
from commode_utils.common import download_dataset
from commode_utils.vocabulary import build_from_scratch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from embeddings_for_trees.data.jsonl_dataset import JsonlASTDataset, JsonlTypedASTDataset
from embeddings_for_trees.data.vocabulary import Vocabulary, TypedVocabulary


class JsonlASTDatamodule(LightningDataModule):
    _train = "train"
    _val = "val"
    _test = "test"

    _vocabulary: Optional[Vocabulary] = None

    def __init__(self, config: DictConfig, data_folder: str):
        super().__init__()
        self._config = config
        self._data_folder = data_folder
        self._name = basename(self._data_folder)

    def prepare_data(self):
        if path.exists(self._data_folder):
            print(f"Dataset is already downloaded")
            return
        if "url" not in self._config:
            raise ValueError(f"Config doesn't contain url for, can't download it automatically")
        download_dataset(self._config.url, self._data_folder, self._name)

    def setup(self, stage: Optional[str] = None):
        if not path.exists(path.join(self._data_folder, Vocabulary.vocab_filename)):
            print("Can't find vocabulary, collect it from train holdout")
            build_from_scratch(path.join(self._data_folder, f"{self._train}.jsonl"), Vocabulary)
        vocabulary_path = path.join(self._data_folder, Vocabulary.vocab_filename)
        self._vocabulary = Vocabulary(vocabulary_path, self._config.max_labels, self._config.max_tokens)

    @staticmethod
    def _collate_batch(sample_list: List[Tuple[torch.Tensor, dgl.DGLGraph]]) -> Tuple[torch.Tensor, dgl.DGLGraph]:
        labels, graphs = zip(*filter(lambda sample: sample is not None, sample_list))
        return torch.cat(labels, dim=1), dgl.batch(graphs)

    def _shared_dataloader(self, holdout: str, shuffle: bool) -> DataLoader:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        holdout_file = path.join(self._data_folder, f"{holdout}.jsonl")
        dataset = JsonlASTDataset(holdout_file, self._vocabulary, self._config, holdout == self._train)
        batch_size = self._config.batch_size if holdout == self._train else self._config.test_batch_size
        return DataLoader(
            dataset, batch_size, shuffle=shuffle, num_workers=self._config.num_workers, collate_fn=self._collate_batch
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._train, True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._val, False)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._test, False)

    def transfer_batch_to_device(
        self, batch: Tuple[torch.Tensor, dgl.DGLGraph], device: torch.device, dataloader_idx: int
    ) -> Tuple[torch.Tensor, dgl.DGLGraph]:
        return batch[0].to(device), batch[1].to(device)

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary


class JsonlTypedASTDatamodule(JsonlASTDatamodule):
    _vocabulary: Optional[TypedVocabulary] = None

    @property
    def vocabulary(self) -> TypedVocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary

    def setup(self, stage: Optional[str] = None):
        if not path.exists(path.join(self._data_folder, Vocabulary.vocab_filename)):
            print("Can't find vocabulary, collect it from train holdout")
            build_from_scratch(path.join(self._data_folder, f"{self._train}.jsonl"), TypedVocabulary)
        vocabulary_path = path.join(self._data_folder, Vocabulary.vocab_filename)
        self._vocabulary = TypedVocabulary(
            vocabulary_path, self._config.max_labels, self._config.max_tokens, self._config.max_types
        )

    def _shared_dataloader(self, holdout: str, shuffle: bool) -> DataLoader:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        holdout_file = path.join(self._data_folder, f"{holdout}.jsonl")
        dataset = JsonlTypedASTDataset(holdout_file, self._vocabulary, self._config, holdout == self._train)
        batch_size = self._config.batch_size if holdout == self._train else self._config.test_batch_size
        return DataLoader(
            dataset, batch_size, shuffle=shuffle, num_workers=self._config.num_workers, collate_fn=self._collate_batch
        )
