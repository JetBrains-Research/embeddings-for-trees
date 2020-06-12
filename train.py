from argparse import ArgumentParser
from json import load as json_load
from pickle import load as pkl_load
from typing import Dict

import torch
import torch.nn as nn

from data_loaders import TreeDGLDataset
from logger import known_loggers, create_logger
from model.tree2seq import Tree2Seq
from trainer import evaluate_on_dataset, train_on_dataset
from utils.common import fix_seed, get_device, PAD
from utils.scheduler import get_scheduler


def train(params: Dict, logger_name: str) -> None:
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    is_resumed = 'resume' in params
    checkpoint = {}
    if is_resumed:
        checkpoint = torch.load(params['resume'], map_location=device)
        resume_wandb_id = params.get('resume_wandb_id', False)
        params = checkpoint['config']
        params['resume_wandb_id'] = resume_wandb_id

    training_set = TreeDGLDataset(
        params['paths']['train'], params['batch_size'], device, True,
        params.get('max_n_nodes', -1), params.get('max_depth', -1)
    )
    validation_set = TreeDGLDataset(params['paths']['validate'], params['batch_size'], device, True)

    with open(params['paths']['vocabulary'], 'rb') as pkl_file:
        vocabulary = pkl_load(pkl_file)
        token_to_id = vocabulary['token_to_id']
        type_to_id = vocabulary['type_to_id']
        label_to_id = vocabulary['label_to_id']

    print('model initializing...')
    # create model
    model = Tree2Seq(
        params['embedding'], params['encoder'], params['decoder'],
        params['hidden_states'], token_to_id, type_to_id, label_to_id
    ).to(device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # create scheduler
    scheduler = get_scheduler(params['scheduler'], optimizer, len(training_set) * params['n_epochs'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=label_to_id[PAD]).to(device)

    # init logger
    logger = create_logger(logger_name, params['logging_folder'], params['checkpoints_folder'], params)
    logger.add_to_saving('configuration', model.get_configuration())

    start_batch_id = checkpoint.get('batch_id', -1) + 1
    # train loop
    print("ok, let's train it")
    for epoch in range(params['n_epochs']):
        logger.epoch = epoch

        # train 1 epoch
        train_on_dataset(
            training_set, validation_set, model, criterion, optimizer, scheduler, params['clip_norm'], logger,
            start_batch_id, params['logging_step'], params['evaluation_step'], params['checkpoint_step']
        )

        eval_epoch_info = evaluate_on_dataset(validation_set, model, criterion)
        logger.log(eval_epoch_info.get_state_dict(), len(training_set), is_train=False)

        model_dump = {
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        logger.save_model(f'epoch_{epoch}.pt', model_dump)


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('config', type=str, help='path to config json')
    arg_parse.add_argument('logger', choices=known_loggers.keys())
    args = arg_parse.parse_args()

    with open(args.config) as config_file:
        train(json_load(config_file), args.logger)
