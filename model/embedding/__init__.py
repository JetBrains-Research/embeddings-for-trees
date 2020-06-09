from importlib import import_module
from inspect import isclass
from pkgutil import iter_modules

from .embedding import Embedding
from .node_embedding import INodeEmbedding
from .reduction import IReduction

for module_info in iter_modules(__path__, f'{__package__}.'):
    module = import_module(module_info.name)
    for attribute_name, attribute in module.__dict__.items():

        if isclass(attribute) and issubclass(attribute, INodeEmbedding) and attribute != INodeEmbedding:
            Embedding.register_node_embedding(attribute)
        if isclass(attribute) and issubclass(attribute, IReduction) and attribute != IReduction:
            Embedding.register_reduction(attribute)
