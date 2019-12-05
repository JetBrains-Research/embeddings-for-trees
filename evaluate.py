from argparse import ArgumentParser
from json import load as json_load
from typing import Dict
from pickle import load as pkl_load

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data_workers.dataset import JavaDataset
from model.tree2seq import ModelFactory, Tree2Seq
from utils.common import fix_seed, get_device, split_tokens_to_subtokens, PAD
from utils.logging import get_possible_loggers, TerminalLogger, FileLogger, WandBLogger, FULL_DATASET
from utils.training import evaluate_dataset


def evaluate(params: Dict, logging: str) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    evaluation_set = JavaDataset(params['paths']['evaluate'], params['batch_size'], True)

    print("processing labels...")
    with open(params['paths']['labels'], 'rb') as pkl_file:
        label_to_id = pkl_load(pkl_file)
    sublabel_to_id, label_to_sublabel = split_tokens_to_subtokens(label_to_id, device=device)

    print("loading model...")
    checkpoint = torch.load(params['paths']['model'], map_location=device)
    configuration = checkpoint['configuration']
    model_factory = ModelFactory(configuration['embedding'], configuration['encoder'], configuration['decoder'])
    model: Tree2Seq = model_factory.construct_model(device)
    model.load_state_dict(checkpoint['state_dict'])

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=sublabel_to_id[PAD]).to(device)

    # # init logging class
    # logger = None
    # if logging == TerminalLogger.name:
    #     logger = TerminalLogger(params['checkpoints_folder'])
    # elif logging == FileLogger.name:
    #     logger = FileLogger(params, params['logging_folder'], params['checkpoints_folder'])
    # elif logging == WandBLogger.name:
    #     logger = WandBLogger('treeLSTM', params, model, params['checkpoints_folder'])

    # evaluation loop
    print("ok, let's evaluate it")
    eval_epoch_info = evaluate_dataset(evaluation_set, model, criterion, sublabel_to_id, device)

    print(eval_epoch_info.get_state_dict())
    # logger.log(eval_epoch_info.get_state_dict(), 0, FULL_DATASET, False)


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    arg_parse.add_argument('--logging', choices=get_possible_loggers(), required=True)
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        evaluate(json_load(config_file), args.logging)
