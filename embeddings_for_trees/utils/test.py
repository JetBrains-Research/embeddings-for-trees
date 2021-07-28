from typing import Optional

import torch
from pytorch_lightning import LightningModule, LightningDataModule, seed_everything, Trainer


def test(model: LightningModule, data_module: LightningDataModule, seed: Optional[int] = None):
    seed_everything(seed)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)
