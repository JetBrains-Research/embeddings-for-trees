import os

import hydra
from omegaconf import DictConfig, OmegaConf
from os import path

from data_module.jsonl_dataset import JsonlDataset
from utils.vocabulary import Vocabulary


@hydra.main(
    config_path="config",
    config_name="treelstm",
)
def train_treelstm(config: DictConfig):
    vocab = Vocabulary(config)
    train_holdout = JsonlDataset(
        os.path.join(
            config.data_folder, config.dataset, f"{config.dataset}.train.jsonl"
        ),
        vocab,
        config,
    )


if __name__ == "__main__":
    train_treelstm()
