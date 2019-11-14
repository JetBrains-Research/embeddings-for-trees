from argparse import ArgumentParser
from json import load as json_load
from pickle import load as pkl_load
from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data_workers.dataset import JavaDataset
from model.tree2seq import ModelFactory, Tree2Seq
from utils.common import fix_seed, get_device, split_tokens_to_subtokens, PAD, UNK, convert_tokens_to_subtokens
from utils.metrics import calculate_metrics
from utils.training import train_on_batch, eval_on_batch


def update_epoch_info(old_info: Dict, batch_info: Dict) -> Dict:
    """Updating epoch info based on batch info,
    if epoch info is None use only batch info (init)

    :param old_info: dict with epoch info or None
    :param batch_info: batch info
    :return: new epoch info
    """
    if old_info is None:
        return batch_info
    for key in ['loss', 'batch_count']:
        old_info[key] += batch_info[key]
    for statistic in ['true_positive', 'false_positive', 'false_negative']:
        old_info['statistics'][statistic] += batch_info['statistics'][statistic]
    return old_info


def print_info(info: Dict, prefix: str):
    metrics = calculate_metrics(info['statistics'])
    print(f"{prefix}:\n"
          f"loss: {info['loss'] / info['batch_count']}\n"
          f"{', '.join(f'{key}: {value}' for key, value in metrics.items())}")


def train(params: Dict) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    training_set = JavaDataset(params['paths']['train_batches'], params['batch_size'], True)
    validation_set = JavaDataset(params['paths']['validation_batches'], params['batch_size'], True)

    print('processing labels...')
    with open(params['paths']['labels_path'], 'rb') as pkl_file:
        label_to_id = pkl_load(pkl_file)
    sublabel_to_id, label_to_sublabel = split_tokens_to_subtokens(label_to_id, device=device)

    print('processing vocabulary...')
    with open(params['paths']['vocabulary_path'], 'rb') as pkl_file:
        vocabulary = pkl_load(pkl_file)
        token_to_id = vocabulary['token_to_id']
        type_to_id = vocabulary['type_to_id']
    subtoken_to_id, token_to_subtoken = split_tokens_to_subtokens(
        token_to_id, required_tokens=[UNK, PAD, 'METHOD_NAME', 'NAN'], return_ids=True, device=device
    )

    print('initializing model set up...')
    # create models
    params['embedding']['params']['token_vocab_size'] = len(subtoken_to_id)
    params['embedding']['params']['token_to_subtoken'] = token_to_subtoken
    params['embedding']['params']['padding_index'] = subtoken_to_id[PAD]
    params['embedding']['params']['type_vocab_size'] = len(type_to_id)
    params['decoder']['params']['out_size'] = len(sublabel_to_id)
    model_factory = ModelFactory(params['embedding'], params['encoder'], params['decoder'])
    model: Tree2Seq = model_factory.construct_model(device)

    # create optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=sublabel_to_id[PAD]).to(device)

    # train loop
    print("ok, let's train it")
    for epoch in range(params['n_epochs']):
        train_epoch_info = None
        eval_epoch_info = None

        # iterate over training set
        for batch_id in tqdm(range(len(training_set))):
            graph, labels = training_set[batch_id]
            graph.ndata['token_id'] = graph.ndata['token_id'].to(device)
            batch_info = train_on_batch(
                model, criterion, optimizer, graph, labels,
                label_to_sublabel, sublabel_to_id, params, device
            )
            train_epoch_info = update_epoch_info(train_epoch_info, batch_info)
            if batch_id % params['verbosity_step'] == 0:
                print_info(train_epoch_info, f"training batch #{batch_id}")
        print_info(train_epoch_info, f"training epoch #{epoch}")

        # iterate over validation set
        for batch_id in tqdm(range(len(validation_set))):
            graph, labels = validation_set[batch_id]
            graph.ndata['token_id'] = graph.ndata['token_id'].to(device)
            eval_label_to_sublabel = convert_tokens_to_subtokens(labels, sublabel_to_id, device)
            batch_info = eval_on_batch(
                model, criterion, graph, labels,
                eval_label_to_sublabel, sublabel_to_id, device
            )
            eval_epoch_info = update_epoch_info(eval_epoch_info, batch_info)
        print_info(eval_epoch_info, f"evaluating epoch #{epoch}")


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        train(json_load(config_file))
