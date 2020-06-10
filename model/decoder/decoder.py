from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn

from utils.common import PAD, UNK


class ITreeDecoder(nn.Module):

    name = "Interface for decoding tree vectors"

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict) -> None:
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.label_to_id = label_to_id

        if UNK not in self.label_to_id:
            self.label_to_id[UNK] = len(self.label_to_id)

        self.out_size = len(self.label_to_id)
        self.pad_index = self.label_to_id[PAD] if PAD in self.label_to_id else -1

    def forward(
            self, encoded_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
            root_indexes: torch.LongTensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Decode given encoded vectors of nodes

        :param encoded_data: tensor or tuple of tensors with encoded data
        :param labels: tensor of labels [sequence len, batch size]
        :param root_indexes: indexes of roots in encoded data
        :return: logits [sequence len, batch size, labels vocab size]
        """
        raise NotImplementedError


class Decoder(nn.Module):
    """Decode sequence given hidden states of nodes"""

    _known_decoders = {}

    def __init__(self, h_enc: int, h_dec: int, label_to_id: Dict, name: str, params: Dict):
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.label_to_id = label_to_id
        self.decoder_name = name

        if self.decoder_name not in self._known_decoders:
            raise ValueError(f"Unknown decoder: {self.decoder_name}")
        self.decoder = self._known_decoders[self.decoder_name](self.h_enc, self.h_dec, self.label_to_id, **params)

    def forward(
            self, encoded_data: Union[torch.Tensor, Tuple[torch.Tensor, ...]], labels: torch.Tensor,
            root_indexes: torch.LongTensor
    ) -> torch.Tensor:
        return self.decoder(encoded_data, labels, root_indexes)

    @staticmethod
    def register_decoder(tree_decoder: ITreeDecoder):
        if not issubclass(tree_decoder, ITreeDecoder):
            raise ValueError(f"Attempt to register not a Tree Decoder class "
                             f"({tree_decoder.__name__} not a subclass of {ITreeDecoder.__name__})")
        Decoder._known_decoders[tree_decoder.name] = tree_decoder

    @staticmethod
    def get_known_decoders() -> List[str]:
        return list(Decoder._known_decoders.keys())
