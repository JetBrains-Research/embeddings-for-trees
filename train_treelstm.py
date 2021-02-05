import os

import hydra
from omegaconf import DictConfig

from data_module.jsonl_data_module import JsonlDataModule
from data_module.jsonl_dataset import JsonlDataset
from models.parts.node_embedding import NodeFeaturesEmbedding
from models.parts.tree_lstm import TreeLSTM
from utils.vocabulary import Vocabulary


@hydra.main(config_path="config", config_name="treelstm")
def train_treelstm(config: DictConfig):
    data_module = JsonlDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    loader = data_module.train_dataloader()

    emb = NodeFeaturesEmbedding(config, data_module.vocabulary)
    treelstm = TreeLSTM(config)

    for batch in loader:
        graph = emb(batch[1])
        graph = treelstm(graph)
        print(graph)
        break


if __name__ == "__main__":
    train_treelstm()
