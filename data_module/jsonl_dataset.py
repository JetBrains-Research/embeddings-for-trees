import json
from json import JSONDecodeError
from os.path import exists
from typing import Optional, Tuple

import dgl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.common import LABEL, AST, CHILDREN, TOKEN, PAD, NODE, SEPARATOR, UNK
from utils.vocabulary import Vocabulary


class JsonlDataset(Dataset):
    def __init__(self, data_file: str, vocabulary: Vocabulary, config: DictConfig):
        if not exists(data_file):
            raise ValueError(f"Can't find file with data: {data_file}")
        self._data_file = data_file
        self._vocab = vocabulary
        self._config = config

        self._token_unk = self._vocab.token_to_id[UNK]
        self._node_unk = self._vocab.node_to_id[UNK]
        self._label_unk = self._vocab.label_to_id[UNK]

        self._line_offsets = []
        cumulative_offset = 0
        with open(self._data_file, "r") as file:
            for line in file:
                self._line_offsets.append(cumulative_offset)
                cumulative_offset += len(line.encode(file.encoding))
        self._n_samples = len(self._line_offsets)

    def __len__(self):
        return self._n_samples

    def _read_line(self, index: int) -> str:
        with open(self._data_file, "r") as data_file:
            data_file.seek(self._line_offsets[index])
            line = data_file.readline().strip()
        return line

    def __getitem__(self, index) -> Optional[Tuple[torch.Tensor, dgl.DGLGraph]]:
        raw_sample = self._read_line(index)
        try:
            sample = json.loads(raw_sample)
        except JSONDecodeError as e:
            print(f"Can't decode json for sample #{index}, failed with {e.msg}")
            return None

        # convert label
        label = torch.full((self._config.max_label_parts, 1), self._vocab.label_to_id[PAD])
        sublabels = sample[LABEL].split(SEPARATOR)[: self._config.max_label_parts]
        label[: len(sublabels), 0] = torch.tensor(
            [self._vocab.label_to_id.get(sl, self._label_unk) for sl in sublabels]
        )

        # prepare ast feature tensors
        ast = sample[AST]
        us, vs = [], []
        token_ids = torch.full((len(ast), self._config.max_token_parts), self._vocab.token_to_id[PAD])
        node_ids = torch.full((len(ast),), self._vocab.node_to_id[PAD])

        # iterate through nodes
        for n_id, node in enumerate(ast):
            if CHILDREN in node:
                us += node[CHILDREN]
                vs += [n_id] * len(node[CHILDREN])
            subtokens = node[TOKEN].split(SEPARATOR)[: self._config.max_token_parts]
            token_ids[n_id, : len(subtokens)] = torch.tensor(
                [self._vocab.token_to_id.get(st, self._token_unk) for st in subtokens]
            )
            node_ids[n_id] = self._vocab.node_to_id.get(node[NODE], self._node_unk)

        # convert to dgl graph
        graph = dgl.graph((us, vs))
        graph.ndata[TOKEN] = token_ids
        graph.ndata[NODE] = node_ids

        return label, graph
