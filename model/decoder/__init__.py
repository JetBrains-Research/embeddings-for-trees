from importlib import import_module
from inspect import isclass
from pkgutil import iter_modules

from .decoder import Decoder, ITreeDecoder

for module_info in iter_modules(__path__, f'{__package__}.'):
    module = import_module(module_info.name)
    for attribute_name, attribute in module.__dict__.items():

        if isclass(attribute) and issubclass(attribute, ITreeDecoder) and attribute != ITreeDecoder:
            Decoder.register_decoder(attribute)
