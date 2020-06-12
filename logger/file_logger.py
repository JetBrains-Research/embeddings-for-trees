from json import dump as json_dump
from os.path import join as join_path
from typing import Dict

from logger import AbstractLogger


class FileLogger(AbstractLogger):
    name = 'file'

    def __init__(self, log_dir: str, checkpoints_dir: str, config: Dict):
        super().__init__(log_dir, checkpoints_dir, config)
        with open(self.log_file, 'w') as logging_file:
            json_dump(config, logging_file)
            logging_file.write('\n' + self._dividing_line)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        with open(self.log_file, 'a') as logging_file:
            logging_file.write(self._create_log_message(state_dict, batch_id, is_train))

    def save_model(self, output_name: str, configuration: Dict) -> str:
        saving_path = join_path(self.checkpoints_dir, output_name)
        self._save_conf_to_disk(saving_path, configuration)
        return saving_path
