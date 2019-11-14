from os import mkdir
from os.path import exists
from shutil import rmtree
from tarfile import open as tar_open
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm


SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'


def get_device() -> torch.device:
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def fix_seed(seed: int = 7) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def extract_tar_gz(tar_path: str, extract_path: str) -> None:
    def tqdm_progress(members):
        extract_progress_bar = tqdm(total=len(list(members.getnames())))
        for member in members:
            extract_progress_bar.update()
            yield member
        extract_progress_bar.close()

    with tar_open(tar_path, 'r:gz') as tarball:
        tarball.extractall(extract_path, members=tqdm_progress(tarball))


def create_folder(path: str, is_clean: bool = True) -> None:
    if is_clean and exists(path):
        rmtree(path)
    mkdir(path)


def split_tokens_to_subtokens(
        token_to_id: Dict, device: torch.device, n_most_common: int = -1,
        delimiter: str = '|', required_tokens: List = None, return_ids: bool = False
) -> Tuple[Dict, Dict]:
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
    token_to_subtoken = {}
    for token, i in token_to_id.items():
        cur_split = torch.tensor([subtoken_to_id.get(tok, 0) for tok in token.split(delimiter)]).to(device)
        if return_ids:
            token_to_subtoken[i] = cur_split
        else:
            token_to_subtoken[token] = cur_split
    return subtoken_to_id, token_to_subtoken
