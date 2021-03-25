from warnings import filterwarnings

from dgl.base import DGLWarning

SEPARATOR = "|"

# data storage keys
LABEL = "label"
AST = "AST"
NODE = "node"
TOKEN = "token"
TYPE = "type"
CHILDREN = "children"


def filter_warnings():
    # "DGLGraph.__len__ is deprecated.Please directly call DGLGraph.number_of_nodes."
    filterwarnings("ignore", category=DGLWarning, module="dgl.base", lineno=45)
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities.distributed", lineno=52)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=216)  # save
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=234)  # load
