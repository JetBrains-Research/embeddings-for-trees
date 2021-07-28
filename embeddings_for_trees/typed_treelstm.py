from argparse import ArgumentParser
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig

from embeddings_for_trees.data.jsonl_data_module import JsonlTypedASTDatamodule
from embeddings_for_trees.models import TypedTreeLSTM2Seq
from embeddings_for_trees.utils.common import filter_warnings
from embeddings_for_trees.utils.train import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def train_treelstm(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))

    if config.print_config:
        print_config(config, fields=["model", "data", "train", "optimizer"])

    # Load data module
    data_module = JsonlTypedASTDatamodule(config.data, config.data_folder)
    data_module.prepare_data()
    data_module.setup()

    # Load model
    typed_treelstm2seq = TypedTreeLSTM2Seq(
        config.model, config.optimizer, data_module.vocabulary, config.train.teacher_forcing
    )

    train(typed_treelstm2seq, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    train_treelstm(__args.config)
