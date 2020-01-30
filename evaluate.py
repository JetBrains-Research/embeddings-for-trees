from argparse import ArgumentParser
from json import load as json_load
from typing import Dict

import torch.nn as nn

from data_workers.dataset import JavaDataset
from model.tree2seq import load_model
from utils.common import fix_seed, get_device
from utils.training import evaluate_dataset


def evaluate(params: Dict) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    evaluation_set = JavaDataset(params['paths']['evaluate'], params['batch_size'], True)

    model, _, _ = load_model(params['paths']['model'], device)

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.decoder.pad_index).to(device)

    # evaluation loop
    print("ok, let's evaluate it")
    eval_epoch_info = evaluate_dataset(evaluation_set, model, criterion, device)

    print(eval_epoch_info.get_state_dict())


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        evaluate(json_load(config_file))
