from datetime import datetime
from os.path import join as join_path
from typing import Dict

import torch

from utils.common import create_folder


class AbstractLogger:
    name = None
    _dividing_line = '=' * 100 + '\n'
    _additional_save_info = {}

    def add_to_saving(self, name: str, info: Dict):
        self._additional_save_info.update({name: info})

    def __init__(self, log_dir: str, checkpoints_dir: str, config: Dict):
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        self.log_file = join_path(log_dir, f'{self.timestamp}.log')
        self.checkpoints_dir = join_path(checkpoints_dir, self.timestamp)
        create_folder(self.checkpoints_dir)
        self.add_to_saving('config', config)

    def _create_log_message(self, state_dict: Dict, batch_id: int, is_train: bool) -> str:
        log_info = f"{'train' if is_train else 'validation'} {batch_id}:\n" + \
                   ', '.join(f'{key}: {value}' for key, value in state_dict.items()) + \
                   '\n' + self._dividing_line
        if batch_id == 0:
            log_info = self._dividing_line + log_info
        return log_info

    def _save_conf_to_disk(self, output_path: str, configuration: Dict):
        configuration.update(self._additional_save_info)
        torch.save(configuration, output_path)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        raise NotImplementedError

    def save_model(self, output_name: str, configuration: Dict) -> str:
        raise NotImplementedError


class PrintLogger(AbstractLogger):
    name = 'print'

    def __init__(self, log_dir: str, checkpoints_dir: str, config: Dict):
        super().__init__(log_dir, checkpoints_dir, config)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        print(self._create_log_message(state_dict, batch_id, is_train))

    def save_model(self, output_name: str, configuration: Dict) -> str:
        saving_path = join_path(self.checkpoints_dir, output_name)
        self._save_conf_to_disk(saving_path, configuration)
        return saving_path
