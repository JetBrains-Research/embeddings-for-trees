from argparse import ArgumentParser
from collections import Counter
from json import JSONDecodeError, loads
from typing import Counter as CounterType, Optional
from typing import Dict

from commode_utils.vocabulary import BaseVocabulary, build_from_scratch

from embeddings_for_trees.utils.common import AST


class Vocabulary(BaseVocabulary):

    NODE = "nodeType"

    @staticmethod
    def process_raw_sample(raw_sample: str, counters: Dict[str, CounterType[str]]):
        try:
            sample = loads(raw_sample)
        except JSONDecodeError:
            with open(Vocabulary._log_filename, "a") as log_file:
                log_file.write(raw_sample + "\n")
            return

        counters[Vocabulary.LABEL].update(sample[Vocabulary.LABEL].split(Vocabulary._separator))
        for node in sample[AST]:
            counters[Vocabulary.TOKEN].update(node[Vocabulary.TOKEN].split(Vocabulary._separator))
            counters[Vocabulary.NODE].update([node[Vocabulary.NODE]])


class TypedVocabulary(Vocabulary):

    TYPE = "tokenType"

    def __init__(
        self,
        vocabulary_file: str,
        max_labels: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_types: Optional[int] = None,
    ):
        super().__init__(vocabulary_file, max_labels, max_tokens)

        self._type_to_id = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        self._type_to_id.update(
            (token[0], i + 4) for i, token in enumerate(self._counters[self.TYPE].most_common(max_types))
        )

    @property
    def type_to_id(self) -> Dict[str, int]:
        return self._type_to_id

    @staticmethod
    def process_raw_sample(raw_sample: str, counters: Dict[str, CounterType[str]]):
        try:
            sample = loads(raw_sample)
        except JSONDecodeError:
            with open(TypedVocabulary._log_filename, "a") as log_file:
                log_file.write(raw_sample + "\n")
            return

        if TypedVocabulary.TYPE not in counters:
            counters[TypedVocabulary.TYPE] = Counter()

        counters[TypedVocabulary.LABEL].update(sample[TypedVocabulary.LABEL].split(TypedVocabulary._separator))
        for node in sample[AST]:
            counters[TypedVocabulary.TOKEN].update(node[TypedVocabulary.TOKEN].split(TypedVocabulary._separator))
            counters[TypedVocabulary.NODE].update([node[TypedVocabulary.NODE]])
            counters[TypedVocabulary.TYPE].update(node[TypedVocabulary.TYPE].split(TypedVocabulary._separator))


if __name__ == "__main__":
    arg_parse = ArgumentParser()
    arg_parse.add_argument("data", type=str, help="Path to file with data")
    arg_parse.add_argument("--typed", action="store_true", help="Use typed vocabulary")
    args = arg_parse.parse_args()

    _vocab_cls = TypedVocabulary if args.typed else Vocabulary
    build_from_scratch(args.data, _vocab_cls)
