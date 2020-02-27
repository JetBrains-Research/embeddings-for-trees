from argparse import ArgumentParser
from json import load as json_load
from pickle import load as pkl_load
from typing import Dict

import dgl
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data_workers.dataset import JavaDataset
from model.tree2seq import ModelFactory, Tree2Seq, load_model
from utils.common import fix_seed, get_device, is_current_step_match, vaswani_lr_scheduler_lambda
from utils.learning_info import LearningInfo
from utils.logging import get_possible_loggers, FileLogger, WandBLogger, TerminalLogger
from utils.training import train_on_batch, evaluate_dataset


def train(params: Dict, logging: str) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    training_set = JavaDataset(params['paths']['train'], params['batch_size'], True)
    validation_set = JavaDataset(params['paths']['validate'], params['batch_size'], True)

    with open(params['paths']['vocabulary'], 'rb') as pkl_file:
        vocabulary = pkl_load(pkl_file)
        token_to_id = vocabulary['token_to_id']
        type_to_id = vocabulary['type_to_id']
        label_to_id = vocabulary['label_to_id']

    print('model initializing...')
    is_resumed = 'resume' in params
    if is_resumed:
        # load model
        model, checkpoint = load_model(params['resume'], device)
        start_batch_id = checkpoint['batch_id'] + 1
        configuration = checkpoint['configuration']
    else:
        # create model
        model_factory = ModelFactory(
            params['embedding'], params['encoder'], params['decoder'],
            params['hidden_states'], token_to_id, type_to_id, label_to_id
        )
        model: Tree2Seq = model_factory.construct_model(device)
        configuration = model_factory.save_configuration()
        start_batch_id = 0

    # create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    # create scheduler
    warm_start = int(len(training_set) * params['n_epochs'] / 100 * params['warm_start_percent'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=vaswani_lr_scheduler_lambda(warm_start, params['hidden_states']['encoder'])
    )
    # set current lr
    for i in range(start_batch_id):
        scheduler.step()

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.decoder.pad_index).to(device)

    # init logging class
    logger = None
    if logging == TerminalLogger.name:
        logger = TerminalLogger(params['checkpoints_folder'])
    elif logging == FileLogger.name:
        logger = FileLogger(params, params['logging_folder'], params['checkpoints_folder'])
    elif logging == WandBLogger.name:
        logger_args = ['treeLSTM', params, model, params['checkpoints_folder']]
        if 'resume_wandb_id' in params:
            logger_args.append(params['resume_wandb_id'])
        logger = WandBLogger(*logger_args)

    # train loop
    print("ok, let's train it")
    for epoch in range(params['n_epochs']):
        train_acc_info = LearningInfo()

        if epoch > 0:
            # specify start batch id only for first epoch
            start_batch_id = 0
        tqdm_batch_iterator = tqdm(range(start_batch_id, len(training_set)), total=len(training_set))
        tqdm_batch_iterator.update(start_batch_id)
        tqdm_batch_iterator.refresh()

        # iterate over training set
        for batch_id in tqdm_batch_iterator:
            graph, labels = training_set[batch_id]
            graph.ndata['token_id'] = graph.ndata['token_id'].to(device)
            graph.ndata['type_id'] = graph.ndata['type_id'].to(device)
            batch_info = train_on_batch(
                model, criterion, optimizer, scheduler, graph, labels, params, device
            )
            train_acc_info.accumulate_info(batch_info)
            # log current train process
            if is_current_step_match(batch_id, params['logging_step']):
                logger.log(train_acc_info.get_state_dict(), epoch, batch_id)
                train_acc_info = LearningInfo()
            # validate current model
            if is_current_step_match(batch_id, params['evaluation_step']) and batch_id != 0:
                eval_epoch_info = evaluate_dataset(validation_set, model, criterion, device)
                logger.log(eval_epoch_info.get_state_dict(), epoch, batch_id, False)
            # save current model
            if is_current_step_match(batch_id, params['checkpoint_step']) and batch_id != 0:
                logger.save_model(
                    model, f'epoch_{epoch}_batch_{batch_id}.pt', configuration, batch_id=batch_id
                )

        logger.log(train_acc_info.get_state_dict(), epoch, len(training_set))
        eval_epoch_info = evaluate_dataset(validation_set, model, criterion, device)
        logger.log(eval_epoch_info.get_state_dict(), epoch, len(training_set), False)

        logger.save_model(model, f'epoch_{epoch}.pt', configuration)


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--config', type=str, required=True, help='path to config json')
    arg_parse.add_argument('--logging', choices=get_possible_loggers(), required=True)
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        train(json_load(config_file), args.logging)
