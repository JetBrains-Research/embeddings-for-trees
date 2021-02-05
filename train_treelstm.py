import os

import hydra
from omegaconf import DictConfig

from data_module.jsonl_data_module import JsonlDataModule
from data_module.jsonl_dataset import JsonlDataset
from utils.vocabulary import Vocabulary


@hydra.main(config_path="config", config_name="treelstm")
def train_treelstm(config: DictConfig):
    data_module = JsonlDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    loader = data_module.train_dataloader()
    for batch in loader:
        print(batch)
        break


if __name__ == "__main__":
    train_treelstm()
