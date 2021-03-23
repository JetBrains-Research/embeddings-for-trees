from os import path
from typing import List, Optional, Tuple

import dgl
import torch
from commode_utils.common import download_dataset
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_module.jsonl_dataset import JsonlASTDataset
from utils.vocabulary import Vocabulary


class JsonlDataModule(LightningDataModule):
    _vocabulary_file = "vocabulary.pkl"
    _train = "train"
    _val = "val"
    _test = "test"

    def __init__(self, config: DictConfig, data_folder: str):
        super().__init__()
        self._config = config
        self._data_folder = data_folder
        self._dataset_dir = path.join(data_folder, config.name)
        self._vocabulary: Optional[Vocabulary] = None

    def prepare_data(self):
        if path.exists(self._dataset_dir):
            print(f"Dataset is already downloaded")
            return
        if "url" not in self._config:
            raise ValueError(f"Config doesn't contain url for {self._config.name}, download it manually")
        download_dataset(self._config.url, self._dataset_dir, self._config.name)

    def setup(self, stage: Optional[str] = None):
        if not path.exists(path.join(self._dataset_dir, Vocabulary.vocab_file)):
            Vocabulary.build_from_scratch(path.join(self._dataset_dir, f"{self._config.name}.{self._train}.jsonl"))
        self._vocabulary = Vocabulary(self._config, self._data_folder)

    @staticmethod
    def _collate_batch(sample_list: List[Tuple[torch.Tensor, dgl.DGLGraph]]) -> Tuple[torch.Tensor, dgl.DGLGraph]:
        labels, graphs = zip(*filter(lambda sample: sample is not None, sample_list))
        return torch.cat(labels, dim=1), dgl.batch(graphs)

    def _shared_dataloader(self, holdout: str, shuffle: bool) -> DataLoader:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup vocabulary before creating data loaders")
        holdout_file = path.join(self._dataset_dir, f"{self._config.name}.{holdout}.jsonl")
        dataset = JsonlASTDataset(holdout_file, self._vocabulary, self._config)
        return DataLoader(
            dataset,
            self._config.batch_size,
            shuffle=shuffle,
            num_workers=self._config.num_workers,
            collate_fn=self._collate_batch,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._train, True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._val, False)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._shared_dataloader(self._test, False)

    def transfer_batch_to_device(
        self, batch: Tuple[torch.Tensor, dgl.DGLGraph], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, dgl.DGLGraph]:
        return batch[0].to(device), batch[1].to(device)

    @property
    def vocabulary(self) -> Vocabulary:
        if self._vocabulary is None:
            raise RuntimeError(f"Setup data module for initializing vocabulary")
        return self._vocabulary
