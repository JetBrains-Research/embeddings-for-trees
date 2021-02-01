import json
import pickle
from argparse import ArgumentParser
from json import JSONDecodeError
from typing import Counter
from os import path

from tqdm import tqdm

from utils.common import get_lines_in_file, LABEL, AST, NODE, TOKEN


def build_vocabulary(train_data: str):
    total_samples = get_lines_in_file(train_data)
    label_counter = Counter[str]()
    node_counter = Counter[str]()
    token_counter = Counter[str]()
    with open(train_data, "r") as f_in:
        for sample_id, sample_json in tqdm(enumerate(f_in), total=total_samples):
            try:
                sample = json.loads(sample_json)
            except JSONDecodeError as e:
                print(f"Can't parse sample #{sample_id}, failed with {e.msg}")
                continue

            label_counter.update(sample[LABEL].split("|"))
            for node in sample[AST]:
                node_counter.update([node[NODE]])
                token_counter.update(node[TOKEN].split("|"))

    print(f"Count {len(label_counter)} labels, top-5: {label_counter.most_common(5)}")
    print(f"Count {len(node_counter)} nodes, top-5: {node_counter.most_common(5)}")
    print(f"Count {len(token_counter)} tokens, top-5: {token_counter.most_common(5)}")

    dataset_dir = path.dirname(train_data)
    vocabulary_file = path.join(dataset_dir, "vocabulary.pkl")
    with open(vocabulary_file, "wb") as f_out:
        pickle.dump({
            LABEL: label_counter,
            NODE: node_counter,
            TOKEN: token_counter,
        }, f_out)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("train_data", help="File with training data", type=str)
    args = arg_parser.parse_args()

    build_vocabulary(args.train_data)
