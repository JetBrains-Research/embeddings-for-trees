from typing import Tuple, Dict

import dgl
import torch
import torch.nn as nn

from model.encoder import _IEncoder
from model.treeLSTM_cell import get_tree_lstm_cell


class TokenTreeLSTM(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, cell_type: str,
                 cell_args: Dict = None, dropout_prob: float = 0.) -> None:
        super().__init__(h_emb, h_enc)
        if cell_args is None:
            cell_args = {}
        self.cell = get_tree_lstm_cell(cell_type)(self.h_emb, self.h_enc, **cell_args)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, batch: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dropout_tokens = self.dropout(batch.ndata['token_embeds'])
        batch.ndata['x'] = dropout_tokens
        return self.cell(batch, device)


class TokenTypeTreeLSTM(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, cell_type: str,
                 cell_args: Dict = None, dropout_prob: float = 0.) -> None:
        super().__init__(h_emb, h_enc)
        if cell_args is None:
            cell_args = {}
        self.cell = get_tree_lstm_cell(cell_type)(2 * self.h_emb, self.h_enc, **cell_args)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, batch: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([batch.ndata['token_embeds'], batch.ndata['type_embeds']], 1)
        batch.ndata['x'] = self.dropout(features)
        return self.cell(batch, device)


class LinearTreeLSTM(_IEncoder):

    def __init__(self, h_emb: int, h_enc: int, cell_type: str,
                 cell_args: Dict = None, dropout_prob: float = 0.) -> None:
        super().__init__(h_emb, h_enc)
        if cell_args is None:
            cell_args = {}
        self.cell = get_tree_lstm_cell(cell_type)(self.h_emb, self.h_enc, **cell_args)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(self.h_emb * 2, self.h_enc)
        self.activation = nn.Tanh()

    def forward(self, batch: dgl.BatchedDGLGraph, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([batch.ndata['token_embeds'], batch.ndata['type_embeds']], 1)
        features = self.activation(self.linear(features))
        batch.ndata['x'] = self.dropout(features)
        return self.cell(batch, device)
