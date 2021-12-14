from typing import Optional

import dgl
from pytorch_lightning import LightningModule, LightningDataModule, seed_everything, Trainer


def test(
    model: LightningModule, data_module: LightningDataModule, seed: Optional[int] = None, gpu: Optional[int] = None
):
    seed_everything(seed)
    dgl.seed(seed)
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)
