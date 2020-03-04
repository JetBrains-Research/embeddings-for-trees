from collections import Counter
from typing import Dict, List, Tuple

from utils.common import UNK, SOS, EOS, PAD


def get_dict_of_subtokens(
        token_to_id: Dict, n_most_common: int = -1, delimiter: str = '|', required_tokens: List = None,
        add_sos_eos: bool = False
) -> Tuple[Dict, Dict]:
    """Create new dicts for converting:
    - subtoken to id
    - token to ids of subtokens

    :param token_to_id: dict for converting tokens to ids
    :param n_most_common: if passed, use only most common subtokens
    :param delimiter: used in tokens to divide into subtokens
    :param required_tokens: list of required tokens in subtoken_to_id dict, if None, than use [UNK, SOS, EOS, PAD]
    :param add_sos_eos: if True, than add SOS and EOS to each token
    :return: subtoken_to_id, token_to_subtokens
    """
    if required_tokens is None:
        required_tokens = [UNK, SOS, EOS, PAD]

    subtoken_counter = Counter()
    token_to_subtokens = {}
    for token, i in token_to_id.items():
        token_split = token.split(delimiter)
        token_to_subtokens[token] = token_split
        subtoken_counter.update(token_split)

    subtoken_to_id = {}
    if n_most_common == -1:
        n_most_common = len(subtoken_counter)
    subtoken_to_id.update(
        [(label[0], num)
         for num, label in enumerate(subtoken_counter.most_common(n_most_common))]
    )

    for req_token in required_tokens:
        if req_token not in subtoken_to_id:
            subtoken_to_id[req_token] = len(subtoken_to_id)

    unk_id = subtoken_to_id[UNK]
    for token, subtokens in token_to_subtokens.items():
        subtoken_ids = [subtoken_to_id.get(st, unk_id) for st in subtokens]
        if add_sos_eos:
            subtoken_ids.insert(0, subtoken_to_id[SOS])
            subtoken_ids.append(subtoken_to_id[EOS])
        token_to_subtokens[token] = subtoken_ids

    return subtoken_to_id, token_to_subtokens


def convert_tokens_to_subtokens_id(
        subtoken_to_id: Dict, tokens: List[str], delimiter: str = '|', add_sos_eos: bool = False
) -> Tuple[List[List[int]], List[int]]:
    unk_id = subtoken_to_id[UNK]
    sos_id = subtoken_to_id[SOS]
    eos_id = subtoken_to_id[EOS]
    subtokens = [
        [subtoken_to_id.get(st, unk_id) for st in token.split(delimiter)]
        for token in tokens
    ]
    if add_sos_eos:
        subtokens = [[sos_id] + st + [eos_id] for st in subtokens]
    lens = [len(st) for st in subtokens]
    return subtokens, lens
