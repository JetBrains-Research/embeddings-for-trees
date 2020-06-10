import unittest
from typing import List, Dict, Tuple

from utils.common import EOS, SOS, PAD, UNK
from utils.token_processing import get_dict_of_subtokens


def _get_simple_token_to_id(add_sos_eos: bool = False) -> Tuple[Dict, List, Dict]:
    token_to_id = {
        'token|name|first': 0,
        'token|second': 1,
        'token|third|name': 2
    }
    list_of_subtokens = ['token', 'name', 'first', 'second', 'third', UNK, PAD, SOS, EOS]
    correct_splits = {
        'token|name|first': ['token', 'name', 'first'],
        'token|second': ['token', 'second'],
        'token|third|name': ['token', 'third', 'name']
    }
    if add_sos_eos:
        for token, split in correct_splits.items():
            correct_splits[token] = [SOS] + split + [EOS]
    return token_to_id, list_of_subtokens, correct_splits


class TokenProcessingTest(unittest.TestCase):

    def _test_dividing_to_subtokens(
            self, list_of_subtokens: List[str], correct_splits: Dict,
            test_subtoken_to_ids: Dict, test_token_to_subtokens: Dict
    ):
        self.assertEqual(len(list_of_subtokens), len(test_subtoken_to_ids))
        for st in list_of_subtokens:
            self.assertTrue(st in test_subtoken_to_ids)

        self.assertEqual(len(correct_splits), len(test_token_to_subtokens))
        for token, split in correct_splits.items():
            list_of_ids = [test_subtoken_to_ids[st] for st in split]
            self.assertListEqual(list_of_ids, test_token_to_subtokens[token])

    def test_get_dict_of_subtokens(self):
        token_to_id, list_of_subtokens, correct_splits = _get_simple_token_to_id()
        subtoken_to_ids, token_to_subtokens = get_dict_of_subtokens(token_to_id)
        self._test_dividing_to_subtokens(list_of_subtokens, correct_splits, subtoken_to_ids, token_to_subtokens)

    def test_get_dict_of_subtokens_with_sos_eos(self):
        token_to_id, list_of_subtokens, correct_splits = _get_simple_token_to_id(True)
        subtoken_to_ids, token_to_subtokens = get_dict_of_subtokens(token_to_id, add_sos_eos=True)
        self._test_dividing_to_subtokens(list_of_subtokens, correct_splits, subtoken_to_ids, token_to_subtokens)


if __name__ == '__main__':
    unittest.main()
