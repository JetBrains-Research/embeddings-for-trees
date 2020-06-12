from os.path import join as join_path
from typing import Dict

import wandb

from logger import AbstractLogger


class WandBLogger(AbstractLogger):
    name = 'wandb'

    def __init__(self, log_dir: str, checkpoints_dir: str, config: Dict):
        super().__init__(log_dir, checkpoints_dir, config)
        resume = config.get('resume_wandb_id', False)
        wandb.init(project=config['wandb_project'], config=config, resume=resume)

    def log(self, state_dict: Dict, batch_id: int, is_train: bool = True) -> None:
        group = 'train' if is_train else 'validation'
        state_dict = {f'{group}/{key}': value for key, value in state_dict.items()}
        state_dict['batch_id'] = batch_id
        wandb.log(state_dict)

    def save_model(self, output_name: str, configuration: Dict) -> str:
        saving_path = join_path(wandb.run.dir, output_name)
        self._save_conf_to_disk(saving_path, configuration)
        wandb.save(saving_path)
        return saving_path
