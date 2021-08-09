from warnings import filterwarnings

from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

SEPARATOR = "|"

# data storage keys
LABEL = "label"
AST = "tree"
NODE = "nodeType"
TOKEN = "token"
TYPE = "tokenType"
CHILDREN = "children"


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.data_loading", lineno=105)
    # "Checkpoint directory {dirpath} exists and is not empty."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.callbacks.model_checkpoint", lineno=446)
    # "DataModule.setup has already been called, so it will not be called again."
    filterwarnings(
        "ignore", category=LightningDeprecationWarning, module="pytorch_lightning.core.datamodule", lineno=423
    )
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=216)  # save
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=234)  # load
