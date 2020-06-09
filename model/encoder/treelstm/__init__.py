from importlib import import_module
from inspect import isclass
from pkgutil import iter_modules

from .treelstm import TreeLSTM, ITreeLSTMCell

for module_info in iter_modules(__path__, f'{__package__}.'):
    module = import_module(module_info.name)
    for attribute_name, attribute in module.__dict__.items():

        if isclass(attribute) and issubclass(attribute, ITreeLSTMCell) and attribute != ITreeLSTMCell:
            TreeLSTM.register_cell(attribute)
