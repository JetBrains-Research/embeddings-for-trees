import pickle
from os import path
from os.path import exists
from typing import Dict

from omegaconf import DictConfig

from utils.common import PAD, UNK, NODE, SOS, EOS, TOKEN, LABEL


class Vocabulary:
    _vocabulary_filename = "vocabulary.pkl"

    def __init__(self, config: DictConfig):
        vocabulary_file = path.join(
            config.data_folder, config.dataset, self._vocabulary_filename
        )
        if not exists(vocabulary_file):
            raise ValueError(f"Can't find vocabulary file ({vocabulary_file})")
        with open(vocabulary_file, "rb") as f_in:
            counters = pickle.load(f_in)
        self._node_to_id = {PAD: 0, UNK: 1}
        self._node_to_id.update((node, i + 2) for i, node in enumerate(counters[NODE]))

        self._token_to_id = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self._token_to_id.update(
            (token[0], i + 4)
            for i, token in enumerate(counters[TOKEN].most_common(config.max_tokens))
        )

        self._label_to_id = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self._label_to_id.update(
            (label[0], i + 4)
            for i, label in enumerate(counters[LABEL].most_common(config.max_labels))
        )

    @property
    def node_to_id(self) -> Dict[str, int]:
        return self._node_to_id

    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._token_to_id

    @property
    def label_to_id(self) -> Dict[str, int]:
        return self._label_to_id
