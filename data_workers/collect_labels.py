from argparse import ArgumentParser, Namespace
from collections import Counter
from os import listdir
from os.path import join as path_join
from pickle import dump as pkl_dump
from pickle import load as pkl_load

from tqdm.auto import tqdm

from utils.common import UNK


def main(args: Namespace) -> None:
    labels = Counter()
    batches = listdir(args.path)
    for batch in tqdm(batches):
        with open(path_join(args.path, batch), 'rb') as pkl_file:
            data = pkl_load(pkl_file)
            labels.update(filter(lambda label: isinstance(label, str), data['labels']))
    print(f'total {len(labels)} labels')
    label_to_id = {
        UNK: 0
    }
    label_to_id.update(
        [(label, num + 1) for num, label in enumerate(labels)]
    )
    with open(args.output, 'wb') as pkl_file:
        pkl_dump(label_to_id, pkl_file)


if __name__ == '__main__':
    argument_parser = ArgumentParser(description='collect labels based on batched data')
    argument_parser.add_argument('--path', type=str, required=True, help='path to folder with batches')
    argument_parser.add_argument('--output', type=str, required=True, help='path to output pickle')

    main(argument_parser.parse_args())
