import random
from os import mkdir
from os.path import exists
from shutil import rmtree
from tarfile import open as tar_open
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'
NAN = 'NAN'
METHOD_NAME = 'METHOD_NAME'
SELF = '<SELF>'


def get_device() -> torch.device:
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    return device


def fix_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    if not exists(path):
        mkdir(path)


def segment_sizes_to_slices(sizes: List) -> List:
    cum_sums = np.cumsum(sizes)
    start_of_segments = np.append([0], cum_sums[:-1])
    return [slice(start, end) for start, end in zip(start_of_segments, cum_sums)]


def get_root_indexes(tree_sizes: List[int]) -> np.ndarray:
    """Get indexes of roots in given graph

    :param tree_sizes: list with tree sizes
    :return: list with indexes of roots [batch size]
    """
    idx_of_roots = np.cumsum([0] + tree_sizes)[:-1]
    return idx_of_roots


def is_step_match(current_step: int, template: int, ignore_zero: bool = True) -> bool:
    match_template = template != -1 and current_step % template == 0
    if ignore_zero:
        return match_template and current_step != 0
    return match_template
