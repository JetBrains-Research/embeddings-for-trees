from collections import Counter
from typing import Dict, List, Tuple

import torch

from utils.common import UNK, SOS, EOS, PAD


def get_dict_of_subtokens(
        token_to_id: Dict, n_most_common: int = -1, delimiter: str = '|', required_tokens: List = None
) -> Dict:
    """Create a new dict for converting subtokens to id
    Information about subtokens gets from token_to_id dict

    :param token_to_id: an original dict
    :param n_most_common: if passed, use only most common subtokens
    :param delimiter: used in tokens to divide into subtokens
    :param required_tokens: which tokens have been in a new dict, if None, than use [UNK, SOS, EOS, PAD]
    :return: the new dict of subtokens
    """
    if required_tokens is None:
        required_tokens = [UNK, SOS, EOS, PAD]
    subtoken_counter = Counter()
    for token, i in token_to_id.items():
        subtoken_counter.update(token.split(delimiter))
    for token in required_tokens:
        if token in subtoken_counter:
            del subtoken_counter[token]
    subtoken_to_id = {}
    subtoken_to_id.update(
        [(token, num) for num, token in enumerate(required_tokens)]
    )
    if n_most_common == -1:
        n_most_common = len(subtoken_counter)
    subtoken_to_id.update(
        [(label[0], num + len(required_tokens))
         for num, label in enumerate(subtoken_counter.most_common(n_most_common))]
    )
    return subtoken_to_id


def get_token_to_subtoken_dict(
        tokens: List[str], subtoken_to_id: Dict, device: torch.device, delimiter: str = '|'
) -> Dict:
    """Create a dict for converting token to corresponding tensor with subtoken's ids

    :param tokens: list of tokens
    :param subtoken_to_id: dict for converting subtoken to its id
    :param device: torch device, where allocate the tensor
    :param delimiter: used in tokens to divide into subtokens
    :return: new dict
    """
    token_to_subtoken = {}
    unk_index = subtoken_to_id[UNK]
    for token in tokens:
        cur_split = torch.tensor([subtoken_to_id.get(tok, unk_index) for tok in token.split(delimiter)]).to(device)
        token_to_subtoken[token] = cur_split
    return token_to_subtoken


def convert_label_to_sublabels(
        labels: List[str], sublabel_to_id: Dict, device: torch.device, delimiter: str = '|'
) -> torch.Tensor:
    """Convert batch of labels to torch tensor with ids of corresponding sublabels
    SOS token is added at the beginning and EOS token is added at the ending for each label
    PAD token is used for fill empty slots in tensor

    :param labels: list of labels (shape: [batch_size])
    :param sublabel_to_id: dict for converting sublabels to ids
    :param device: torch device
    :param delimiter: used in tokens to divide into subtokens
    :return: tensor with information about sublabels (shape: [max_length_of_sublabels + 2, batch_size])
    """
    label_to_sublabel = get_token_to_subtoken_dict(labels, sublabel_to_id, device, delimiter)

    sublabels_length = torch.tensor([label_to_sublabel[label].shape[0] for label in labels])
    max_sublabel_length = sublabels_length.max()
    torch_labels = torch.full((max_sublabel_length.item() + 2, len(labels)), sublabel_to_id[PAD],
                              dtype=torch.long)
    torch_labels[0, :] = sublabel_to_id[SOS]
    torch_labels[sublabels_length + 1, torch.arange(0, len(labels))] = sublabel_to_id[EOS]
    for sample, label in enumerate(labels):
        torch_labels[1:sublabels_length[sample] + 1, sample] = label_to_sublabel[label]
    return torch_labels


def get_token_id_to_subtoken_dict(
    token_ids: List[int], id_to_token: Dict, subtoken_to_id: Dict, device: torch.device, delimiter: str = '|'
) -> Dict:
    """Create a dict for converting token's id to tensor of corresponding subtoken's ids

    :param token_ids: list of token's ids
    :param id_to_token: dict for converting token's id to token
    :param subtoken_to_id: dict for converting subtoken to id
    :param device: torch device
    :param delimiter: used in tokens to divide into subtokens
    :return: new dict
    """
    tokens = [id_to_token[token_id] for token_id in token_ids]
    token_to_subtoken = get_token_to_subtoken_dict(tokens, subtoken_to_id, device, delimiter)
    token_id_to_subtoken = {
        token_id: token_to_subtoken[id_to_token[token_id]] for token_id in token_ids
    }
    return token_id_to_subtoken
