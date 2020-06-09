from typing import List, Dict

import torch
import torch.nn as nn


class ISubtreeAttention(nn.Module):

    name = "Subtree attention interface"

    def __init__(self, h_enc: int, h_dec: int) -> None:
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec

    def forward(self, hidden_states: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List) -> torch.Tensor:
        """ Compute attention weights based on previous decoder state and encoder output

        :param hidden_states: [batch size, hidden size]
        :param encoder_output: [number of nodes in batch, hidden size]
        :param tree_sizes: [batch size]
        :return: attention weights [number of nodes in batch, 1]
        """
        raise NotImplementedError


class Attention(nn.Module):

    _known_attentions = {}

    def __init__(self, h_enc: int, h_dec: int, name: str, params: Dict):
        super().__init__()
        self.h_enc = h_enc
        self.h_dec = h_dec
        self.attention_name = name

        if self.attention_name not in self._known_attentions:
            raise ValueError(f"Unknown attention: {self.attention_name}")
        self.attention = self._known_attentions[self.attention_name](self.h_enc, self.h_dec, **params)

    def forward(self, hidden_states: torch.Tensor, encoder_output: torch.Tensor, tree_sizes: List) -> torch.Tensor:
        return self.attention(hidden_states, encoder_output, tree_sizes)

    @staticmethod
    def register_attention(subtree_attention: ISubtreeAttention):
        if not issubclass(subtree_attention, ISubtreeAttention):
            raise ValueError(f"Attempt to register not an Attention class "
                             f"({subtree_attention.__name__} not a subclass of {ISubtreeAttention.__name__})")
        Attention._known_attentions[subtree_attention.name] = subtree_attention

    @staticmethod
    def get_known_attentions() -> List[str]:
        return list(Attention._known_attentions.keys())
