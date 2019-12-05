from datetime import datetime
from json import dump as json_dump
from os.path import join as join_path
from typing import Dict

import wandb
import torch

from model.tree2seq import Tree2Seq
from utils.common import create_folder


FULL_DATASET = 'full_dataset'


def get_possible_loggers():
    return [
        TerminalLogger.name,
        FileLogger.name,
        WandBLogger.name
    ]


class _ILogger:

    name = '_ILogger'

    def __init__(self, checkpoints_folder: str):
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.checkpoints_folder = join_path(checkpoints_folder, self.timestamp)
        create_folder(self.checkpoints_folder)

    def log(self, state_dict: Dict, epoch_num: int, batch_num: int, is_train: bool = True) -> None:
        raise NotImplementedError

    def save_model(self, model: Tree2Seq, epoch_num: int, params: Dict) -> str:
        saving_path = join_path(self.checkpoints_folder, f'epoch_{epoch_num}.pt')
        torch.save({
            'state_dict': model.state_dict(),
            'configuration': params
        }, saving_path)
        return saving_path


class WandBLogger(_ILogger):

    name = 'wandb'
    step = 0

    def __init__(self, project_name: str, config: Dict, model: Tree2Seq, checkpoints_folder: str):
        super().__init__(checkpoints_folder)
        wandb.init(project=project_name, config=config)
        # wandb can't work with graphs?
        # wandb.watch(model, log='all')

    def log(self, state_dict: Dict, epoch_num: int, batch_num: int, is_train: bool = True) -> None:
        group = 'train' if is_train else 'validation'
        state_dict = {f'{group}/{key}': value for key, value in state_dict.items()}
        state_dict['epoch'] = epoch_num
        if not is_train:
            # set step for validation the same as last for training or zero
            self.step = max(0, self.step - 1)
        wandb.log(state_dict, step=self.step)
        self.step += 1

    def save_model(self, model: Tree2Seq, epoch_num: int, params: Dict) -> str:
        checkpoint_path = super().save_model(model, epoch_num, params)
        wandb.save(checkpoint_path)
        return checkpoint_path


class TerminalLogger(_ILogger):

    name = 'terminal'
    _dividing_line = '=' * 100 + '\n'

    def __init__(self, checkpoints_folder: str):
        super().__init__(checkpoints_folder)

    def log(self, state_dict: Dict, epoch_num: int, batch_num: int, is_train: bool = True) -> None:
        print(self._create_log_message(state_dict, epoch_num, batch_num, is_train))

    def _create_log_message(self, state_dict: Dict, epoch_num: int, batch_num: int, is_train: bool) -> str:
        log_info = f"{'train' if is_train else 'validation'} {epoch_num}.{batch_num}:\n" + \
                   ', '.join(f'{key}: {value}' for key, value in state_dict.items()) + \
                   '\n' + self._dividing_line
        if batch_num == 0:
            log_info = self._dividing_line + log_info
        return log_info

    def save_model(self, model: Tree2Seq, epoch_num: int, params: Dict) -> str:
        return super().save_model(model, epoch_num, params)


class FileLogger(TerminalLogger):

    name = 'file'

    def __init__(self, config: Dict, logging_folder: str, checkpoints_folder: str):
        super().__init__(checkpoints_folder)
        self.file_path = join_path(logging_folder, f'{self.timestamp}.log')
        with open(self.file_path, 'w') as logging_file:
            json_dump(config, logging_file)
            logging_file.write('\n' + self._dividing_line)

    def log(self, state_dict: Dict, epoch_num: int, batch_num: int, is_train: bool = True) -> None:
        with open(self.file_path, 'a') as logging_file:
            logging_file.write(self._create_log_message(state_dict, epoch_num, batch_num, is_train))

    def save_model(self, model: Tree2Seq, epoch_num: int, params: Dict) -> str:
        return super().save_model(model, epoch_num, params)
