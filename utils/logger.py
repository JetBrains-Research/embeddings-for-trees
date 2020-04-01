from datetime import datetime
from json import dump as json_dump
from os.path import join as join_path
from typing import Dict, Union

import torch
import wandb

from utils.common import create_folder


def get_possible_loggers():
    return [
        Logger.name,
        FileLogger.name,
        WandBLogger.name
    ]


class Logger:
    name = 'terminal'
    _dividing_line = '=' * 100 + '\n'

    def __init__(self, checkpoints_folder: str, config: Dict):
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.checkpoints_folder = join_path(checkpoints_folder, self.timestamp)
        self.additional_save_info = {
            'config': config
        }
        create_folder(self.checkpoints_folder)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        print(self._create_log_message(state_dict, batch_id, is_train))

    def _create_log_message(self, state_dict: Dict, batch_id: int, is_train: bool) -> str:
        log_info = f"{'train' if is_train else 'validation'} {batch_id}:\n" + \
                   ', '.join(f'{key}: {value}' for key, value in state_dict.items()) + \
                   '\n' + self._dividing_line
        if batch_id == 0:
            log_info = self._dividing_line + log_info
        return log_info

    def save_model(self, output_name: str, configuration: Dict) -> str:
        saving_path = join_path(self.checkpoints_folder, output_name)
        configuration.update(self.additional_save_info)
        torch.save(configuration, saving_path)
        return saving_path


class WandBLogger(Logger):
    name = 'wandb'
    project_name = 'treeLSTM'

    def __init__(self, checkpoints_folder: str, config: Dict, resume: Union[bool, str] = False) -> None:
        super().__init__(checkpoints_folder, config)
        wandb.init(project=self.project_name, config=config, resume=resume)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        group = 'train' if is_train else 'validation'
        state_dict = {f'{group}/{key}': value for key, value in state_dict.items()}
        state_dict['batch_id'] = batch_id
        wandb.log(state_dict)

    def save_model(self, output_name: str, configuration: Dict) -> str:
        checkpoint_path = super().save_model(output_name, configuration)
        wandb.save(checkpoint_path)
        return checkpoint_path


class FileLogger(Logger):
    name = 'file'

    def __init__(self, checkpoints_folder: str, config: Dict, logging_folder: str):
        super().__init__(checkpoints_folder, config)
        self.file_path = join_path(logging_folder, f'{self.timestamp}.log')
        with open(self.file_path, 'w') as logging_file:
            json_dump(config, logging_file)
            logging_file.write('\n' + self._dividing_line)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        with open(self.file_path, 'a') as logging_file:
            logging_file.write(self._create_log_message(state_dict, batch_id, is_train))

    def save_model(self, output_name: str, configuration: Dict) -> str:
        return super().save_model(output_name, configuration)
