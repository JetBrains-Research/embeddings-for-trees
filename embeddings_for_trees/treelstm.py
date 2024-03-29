from argparse import ArgumentParser
from typing import cast

import torch
from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig

from embeddings_for_trees.data.jsonl_data_module import JsonlASTDatamodule
from embeddings_for_trees.models import TreeLSTM2Seq
from embeddings_for_trees.utils.common import filter_warnings
from embeddings_for_trees.utils.test import test
from embeddings_for_trees.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("mode", help="Mode to run script", choices=["train", "test"])
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def train_treelstm(config: DictConfig):
    filter_warnings()

    if config.print_config:
        print_config(config, fields=["model", "data", "train", "optimizer"])

    # Load data module
    data_module = JsonlASTDatamodule(config.data, config.data_folder)

    # Load model
    treelstm2seq = TreeLSTM2Seq(config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing)

    train(treelstm2seq, data_module, config)


def test_treelstm(config: DictConfig):
    filter_warnings()

    # Load data module
    data_module = JsonlASTDatamodule(config.data, config.data_folder)

    # Load model
    treelstm2seq = TreeLSTM2Seq.load_from_checkpoint(config.checkpoint, map_location=torch.device("cpu"))

    test(treelstm2seq, data_module, config.seed, config.train.gpu)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()

    __config = cast(DictConfig, OmegaConf.load(__args.config))
    if __args.mode == "train":
        train_treelstm(__config)
    else:
        test_treelstm(__config)
