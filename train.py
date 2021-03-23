from typing import Dict, Type

import dgl
import hydra
import torch
from commode_utils.callback import UploadCheckpointCallback, PrintEpochResultCallback
from commode_utils.common import print_config
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_module.jsonl_data_module import JsonlASTDatamodule, JsonlTypedASTDatamodule
from models import TreeLSTM2Seq, TypedTreeLSTM2Seq
from utils.common import filter_warnings

MODELS: Dict[str, Type[TreeLSTM2Seq]] = {"Tree-LSTM": TreeLSTM2Seq, "Typed Tree-LSTM": TypedTreeLSTM2Seq}


@hydra.main(config_path="config", config_name="main")
def train(config: DictConfig):
    filter_warnings()
    seed_everything(config.seed)
    dgl.seed(config.seed)

    print_config(config, fields=["model", "datamodule", "trainer"])

    if config.datamodule.get("use_types", False):
        data_module: JsonlASTDatamodule = JsonlTypedASTDatamodule(config.datamodule, config.data_folder)
    else:
        data_module = JsonlASTDatamodule(config.datamodule, config.data_folder)
    data_module.prepare_data()
    data_module.setup()

    if config.model.name not in MODELS:
        print(f"Unknown model: {config.model.name}")
        return
    model = MODELS[config.model.name](config, data_module.vocabulary)

    params = config.trainer
    resume = params.get("resume_checkpoint", None)

    # define logger
    wandb_logger = WandbLogger(
        project=f"tree-lstm-{config.datamodule.name}", log_model=False, offline=params.log_offline
    )

    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        period=params.save_every_epoch,
        save_top_k=-1,
    )
    upload_checkpoint_callback = UploadCheckpointCallback(wandb_logger.experiment.dir)
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=params.patience, monitor="val_loss", verbose=True, mode="min")
    # define callback for printing intermediate result
    print_epoch_result_callback = PrintEpochResultCallback("train", "val")
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")
    trainer = Trainer(
        max_epochs=params.n_epochs,
        gradient_clip_val=params.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=params.val_every_epoch,
        log_every_n_steps=params.log_every_step,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=params.progress_bar_refresh_rate,
        callbacks=[
            lr_logger,
            early_stopping_callback,
            checkpoint_callback,
            upload_checkpoint_callback,
            print_epoch_result_callback,
        ],
        resume_from_checkpoint=resume,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    train()
