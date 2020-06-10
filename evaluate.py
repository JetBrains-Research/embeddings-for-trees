from argparse import ArgumentParser
from json import load as json_load
from typing import Dict

import torch
import torch.nn as nn

from data_workers.dataset import JavaDataset
from model.tree2seq import Tree2Seq
from utils.common import fix_seed, get_device, PAD
from utils.training import evaluate_on_dataset


def evaluate(params: Dict) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    checkpoint = torch.load(params['model'], map_location=device)

    print('model initializing...')
    # create model
    model = Tree2Seq(**checkpoint['configuration']).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    evaluation_set = JavaDataset(params['dataset'], params['batch_size'], device, True)

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.label_to_id[PAD]).to(device)

    # evaluation loop
    print("ok, let's evaluate it")
    eval_epoch_info = evaluate_on_dataset(evaluation_set, model, criterion)

    print(eval_epoch_info.get_state_dict())


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('config', type=str, help='path to config json')
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        evaluate(json_load(config_file))
