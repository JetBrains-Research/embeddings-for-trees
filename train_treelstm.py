import os

import hydra
from omegaconf import DictConfig

from data_module.jsonl_data_module import JsonlDataModule
from data_module.jsonl_dataset import JsonlDataset
from models.parts.lstm_decoder import LSTMDecoder
from models.parts.node_embedding import NodeFeaturesEmbedding
from models.parts.tree_lstm_encoder import TreeLSTM
from utils.vocabulary import Vocabulary


@hydra.main(config_path="config", config_name="treelstm")
def train_treelstm(config: DictConfig):
    data_module = JsonlDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    loader = data_module.train_dataloader()

    emb = NodeFeaturesEmbedding(config, data_module.vocabulary)
    treelstm = TreeLSTM(config)
    decoder = LSTMDecoder(config, data_module.vocabulary)

    for batch in loader:
        graph = emb(batch[1])
        encoded_nodes = treelstm(graph)
        out = decoder(encoded_nodes, graph.batch_num_nodes(), 5)
        print(out.shape)
        print(out)
        break


if __name__ == "__main__":
    train_treelstm()
