import pickle
from collections import Counter
from json import JSONDecodeError, loads
from os import path
from os.path import exists
from typing import Dict
from typing import Counter as CounterType

from omegaconf import DictConfig
from tqdm import tqdm

from utils.common import PAD, UNK, NODE, SOS, EOS, TOKEN, LABEL, SEPARATOR, AST, get_lines_in_file


class Vocabulary:
    vocab_file = "vocabulary.pkl"
    _log_file = "bad_samples.log"

    def __init__(self, config: DictConfig):
        vocabulary_file = path.join(config.data_folder, config.dataset, self.vocab_file)
        if not exists(vocabulary_file):
            raise ValueError(f"Can't find vocabulary file ({vocabulary_file})")
        with open(vocabulary_file, "rb") as f_in:
            counters = pickle.load(f_in)
        self._node_to_id = {PAD: 0, UNK: 1}
        self._node_to_id.update((node, i + 2) for i, node in enumerate(counters[NODE]))

        self._token_to_id = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self._token_to_id.update(
            (token[0], i + 4) for i, token in enumerate(counters[TOKEN].most_common(config.max_tokens))
        )

        self._label_to_id = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
        self._label_to_id.update(
            (label[0], i + 4) for i, label in enumerate(counters[LABEL].most_common(config.max_labels))
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

    @staticmethod
    def build_from_scratch(train_data: str):
        total_samples = get_lines_in_file(train_data)
        label_counter: CounterType[str] = Counter()
        node_counter: CounterType[str] = Counter()
        token_counter: CounterType = Counter()
        with open(train_data, "r") as f_in:
            for sample_id, sample_json in tqdm(enumerate(f_in), total=total_samples):
                try:
                    sample = loads(sample_json)
                except JSONDecodeError:
                    with open(Vocabulary._log_file, "a") as log_file:
                        log_file.write(sample_json + "\n")
                    continue

                label_counter.update(sample[LABEL].split(SEPARATOR))
                for node in sample[AST]:
                    node_counter.update([node[NODE]])
                    token_counter.update(node[TOKEN].split(SEPARATOR))

        print(f"Count {len(label_counter)} labels, top-5: {label_counter.most_common(5)}")
        print(f"Count {len(node_counter)} nodes, top-5: {node_counter.most_common(5)}")
        print(f"Count {len(token_counter)} tokens, top-5: {token_counter.most_common(5)}")

        dataset_dir = path.dirname(train_data)
        vocabulary_file = path.join(dataset_dir, Vocabulary.vocab_file)
        with open(vocabulary_file, "wb") as f_out:
            pickle.dump(
                {
                    LABEL: label_counter,
                    NODE: node_counter,
                    TOKEN: token_counter,
                },
                f_out,
            )
