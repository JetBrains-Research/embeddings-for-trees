from argparse import ArgumentParser
from copy import deepcopy
from json import load as json_load
from pickle import load as pkl_load
from typing import Dict

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data_workers.dataset import JavaDataset
from model.tree2seq import ModelFactory, Tree2Seq
from utils.common import fix_seed, get_device, split_tokens_to_subtokens, PAD, UNK, convert_tokens_to_subtokens
from utils.logging import get_possible_loggers, FileLogger, WandBLogger, FULL_BATCH
from utils.metrics import calculate_metrics
from utils.training import train_on_batch, eval_on_batch


def accumulate_info(old_info: Dict, batch_info: Dict) -> Dict:
    """Updating epoch info based on batch info,
    if epoch info is None use only batch info (init)

    :param old_info: dict with epoch info or None
    :param batch_info: batch info
    :return: new epoch info
    """
    if old_info is None:
        return batch_info
    old_info['loss'] += batch_info['loss']
    for statistic in ['true_positive', 'false_positive', 'false_negative']:
        old_info['statistics'][statistic] += batch_info['statistics'][statistic]
    return old_info


def acc_info_to_state_dict(accumulated_info: Dict, logging_step: int) -> Dict:
    state_dict = {
        'loss': accumulated_info['loss'] / logging_step
    }
    state_dict.update(calculate_metrics(accumulated_info['statistics']))
    return state_dict


def train(params: Dict, logging: str) -> None:
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

    print('model initializing...')
    # create models
    extended_params = deepcopy(params)
    extended_params['embedding']['params']['token_vocab_size'] = len(subtoken_to_id)
    extended_params['embedding']['params']['token_to_subtoken'] = token_to_subtoken
    extended_params['embedding']['params']['padding_index'] = subtoken_to_id[PAD]
    extended_params['embedding']['params']['type_vocab_size'] = len(type_to_id)
    extended_params['decoder']['params']['out_size'] = len(sublabel_to_id)
    extended_params['decoder']['params']['padding_index'] = sublabel_to_id[PAD]
    model_factory = ModelFactory(extended_params['embedding'], extended_params['encoder'], extended_params['decoder'])
    model: Tree2Seq = model_factory.construct_model(device)

    # create optimizer
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=sublabel_to_id[PAD]).to(device)

    # init logging class
    logger = None
    if logging == FileLogger.name:
        logger = FileLogger(params, params['logging_folder'], params['checkpoints_folder'])
    elif logging == WandBLogger.name:
        logger = WandBLogger('treeLSTM', params, model, params['checkpoints_folder'])

    # train loop
    print("ok, let's train it")
    for epoch in range(params['n_epochs']):
        train_acc_info = None
        eval_epoch_info = None

        # iterate over training set
        for batch_id in tqdm(range(len(training_set))):
            graph, labels = training_set[batch_id]
            graph.ndata['token_id'] = graph.ndata['token_id'].to(device)
            batch_info = train_on_batch(
                model, criterion, optimizer, graph, labels,
                label_to_sublabel, sublabel_to_id, params, device
            )
            train_acc_info = accumulate_info(train_acc_info, batch_info)
            if batch_id % params['logging_step'] == 0:
                state_dict = acc_info_to_state_dict(train_acc_info, params['logging_step'] if batch_id != 0 else 1)
                logger.log(state_dict, epoch, batch_id)
                train_acc_info = None

        # iterate over validation set
        for batch_id in tqdm(range(len(validation_set))):
            graph, labels = validation_set[batch_id]
            graph.ndata['token_id'] = graph.ndata['token_id'].to(device)
            eval_label_to_sublabel = convert_tokens_to_subtokens(labels, sublabel_to_id, device)
            batch_info = eval_on_batch(
                model, criterion, graph, labels,
                eval_label_to_sublabel, sublabel_to_id, device
            )
            eval_epoch_info = accumulate_info(eval_epoch_info, batch_info)
        state_dict = acc_info_to_state_dict(eval_epoch_info, len(validation_set))
        logger.log(state_dict, epoch, FULL_BATCH, False)

        if epoch % params['checkpoint_step'] == 0:
            logger.save_model(model, epoch)


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    arg_parse.add_argument('--logging', choices=get_possible_loggers(), required=True)
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        train(json_load(config_file), args.logging)
