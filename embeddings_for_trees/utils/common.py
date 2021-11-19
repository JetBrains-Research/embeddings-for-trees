from warnings import filterwarnings

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
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.data_loading", lineno=110)
    # "Checkpoint directory {dirpath} exists and is not empty."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.callbacks.model_checkpoint", lineno=617)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=216)  # save
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=234)  # load
    # "Trying to infer the `batch_size` from an ambiguous collection."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities.data", lineno=56)
    filterwarnings("ignore", module="dgl.base", lineno=45)
