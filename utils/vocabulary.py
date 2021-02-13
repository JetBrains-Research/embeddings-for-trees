import pickle
from argparse import ArgumentParser
from collections import Counter
from json import JSONDecodeError, loads
from os import path
from os.path import exists
from typing import Dict
from typing import Counter as CounterType

from omegaconf import DictConfig
from tqdm import tqdm

from utils.common import PAD, UNK, NODE, SOS, EOS, TOKEN, LABEL, SEPARATOR, AST, get_lines_in_file, CHILDREN, TYPE


class Vocabulary:
    vocab_file = "vocabulary.pkl"
    _log_file = "bad_samples.log"

    _split_features = [LABEL, TOKEN, TYPE]

    def __init__(self, config: DictConfig):
        vocabulary_file = path.join(config.data_folder, config.dataset, self.vocab_file)
        if not exists(vocabulary_file):
            raise ValueError(f"Can't find vocabulary file ({vocabulary_file})")
        with open(vocabulary_file, "rb") as f_in:
            counters = pickle.load(f_in)
        max_parts = {LABEL: config.max_labels, TOKEN: config.max_tokens, TYPE: config.get("max_types", None)}
        self._dicts = {}
        for key, counter in counters.items():
            if key in self._split_features:
                service_tokens = {PAD: 0, UNK: 1, SOS: 2, EOS: 3}
            else:
                service_tokens = {PAD: 0, UNK: 1}
            skip_id = len(service_tokens)
            self._dicts[key] = service_tokens
            self._dicts[key].update(
                (token[0], i + skip_id) for i, token in enumerate(counter.most_common(max_parts.get(key, None)))
            )

    @property
    def node_to_id(self) -> Dict[str, int]:
        return self._dicts[NODE]

    @property
    def token_to_id(self) -> Dict[str, int]:
        return self._dicts[TOKEN]

    @property
    def label_to_id(self) -> Dict[str, int]:
        return self._dicts[LABEL]

    @property
    def type_to_id(self) -> Dict[str, int]:
        return self._dicts[TYPE]

    @staticmethod
    def build_from_scratch(train_data: str):
        total_samples = get_lines_in_file(train_data)
        label_counter: CounterType[str] = Counter()
        feature_counters: Dict[str, CounterType[str]] = {}
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
                    for feature, value in node.items():
                        if feature == CHILDREN:
                            continue
                        if feature in Vocabulary._split_features:
                            value = value.split(SEPARATOR)
                        else:
                            value = [value]
                        if feature not in feature_counters:
                            feature_counters[feature] = Counter()
                        feature_counters[feature].update(value)

        feature_counters[LABEL] = label_counter
        for feature, counter in feature_counters.items():
            print(f"Count {len(counter)} {feature}, top-5: {counter.most_common(5)}")

        dataset_dir = path.dirname(train_data)
        vocabulary_file = path.join(dataset_dir, Vocabulary.vocab_file)
        with open(vocabulary_file, "wb") as f_out:
            pickle.dump(feature_counters, f_out)


if __name__ == "__main__":
    arg_parse = ArgumentParser()
    arg_parse.add_argument("data", type=str, help="Path to file with data")
    args = arg_parse.parse_args()
    Vocabulary.build_from_scratch(args.data)
