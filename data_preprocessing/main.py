import os
from argparse import ArgumentParser, Namespace
from pickle import load as pickle_load

from data_preprocessing.data_information import JavaSmallDataset, JavaMediumDataset, JavaLargeDataset, JavaTestDataset
from data_preprocessing.dot2dgl import convert_holdout
from data_preprocessing.preprocess_steps import download_dataset, build_dataset_asts, collect_vocabulary, upload_dataset
from utils.common import create_folder, fix_seed

DATA_FOLDER = 'data'
VOCABULARY_NAME = 'vocabulary.pkl'
ASTMINER_PATH = 'utils/astminer-cli.jar'

known_datasets = {
    JavaSmallDataset.name: JavaSmallDataset,
    JavaMediumDataset.name: JavaMediumDataset,
    JavaLargeDataset.name: JavaLargeDataset,
    JavaTestDataset.name: JavaTestDataset,
}


def main(args: Namespace) -> None:
    fix_seed()
    if args.dataset not in known_datasets:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dataset_info = known_datasets[args.dataset]
    dataset_path = os.path.join(DATA_FOLDER, dataset_info.name)
    vocabulary_path = os.path.join(dataset_path, VOCABULARY_NAME)
    create_folder(dataset_path, is_clean=False)

    if args.download:
        download_dataset(dataset_info, dataset_path)

    if args.build_ast:
        if not all([os.path.exists(os.path.join(dataset_path, holdout)) for holdout in dataset_info.holdout_folders]):
            raise RuntimeError("download and extract data before building ast")
        if not os.path.exists(ASTMINER_PATH):
            raise RuntimeError(f"can't find astminer-cli in this location {ASTMINER_PATH}")
        build_dataset_asts(dataset_info, dataset_path, ASTMINER_PATH)

    if args.collect_vocabulary:
        train_asts = os.path.join(dataset_path, f'{dataset_info.holdout_folders[0]}_asts')
        if not os.path.exists(train_asts):
            raise RuntimeError("build training asts before collecting vocabulary")
        collect_vocabulary(
            train_asts, vocabulary_path, args.n_tokens, args.n_types, args.n_labels,
            args.split_vocabulary, args.wrap_tokens, args.wrap_labels, '|'
        )

    if args.convert:
        if not os.path.exists(vocabulary_path):
            raise RuntimeError("collect vocabulary before converting data to DGL format")
        with open(vocabulary_path, 'rb') as pkl_file:
            vocab = pickle_load(pkl_file)
        token_to_id, type_to_id, label_to_id = vocab['token_to_id'], vocab['type_to_id'], vocab['label_to_id']
        for holdout in dataset_info.holdout_folders:
            ast_folder = os.path.join(dataset_path, f'{holdout}_asts')
            if not os.path.exists(ast_folder):
                raise RuntimeError(f"build asts for {holdout} before converting it to DGL format")
            output_folder = os.path.join(dataset_path, f'{holdout}_preprocessed')
            create_folder(output_folder)
            convert_holdout(ast_folder, output_folder, args.batch_size, token_to_id, type_to_id, label_to_id,
                            args.tokens_to_leaves, args.split_vocabulary, args.max_token_len, args.max_label_len,
                            args.wrap_tokens, args.wrap_labels, '|', True, args.n_jobs)

    if args.upload:
        if not all([os.path.exists(os.path.join(dataset_path, f'{holdout}_preprocessed'))
                    for holdout in dataset_info.holdout_folders]):
            raise RuntimeError("preprocess data before uploading it to the cloud")
        upload_dataset(dataset_info, dataset_path, VOCABULARY_NAME, args.store, args.tar_suffix)

    preprocessed_paths = [os.path.join(dataset_path, f'{holdout}_preprocessed')
                          for holdout in dataset_info.holdout_folders]
    if all([os.path.exists(path) for path in preprocessed_paths]):
        for holdout, path in zip(dataset_info.holdout_folders, preprocessed_paths):
            number_of_batches = len(os.listdir(path))
            print(f"There are {number_of_batches} batches in {holdout} data")


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(known_datasets.keys()))
    arg_parser.add_argument('--download', action='store_true')

    arg_parser.add_argument('--build_ast', action='store_true')

    arg_parser.add_argument('--collect_vocabulary', action='store_true')
    arg_parser.add_argument('--n_tokens', type=int, default=-1)
    arg_parser.add_argument('--n_types', type=int, default=-1)
    arg_parser.add_argument('--n_labels', type=int, default=-1)
    arg_parser.add_argument('--split_vocabulary', action='store_true')
    arg_parser.add_argument('--wrap_labels', action='store_true')
    arg_parser.add_argument('--wrap_tokens', action='store_true')

    arg_parser.add_argument('--convert', action='store_true')
    arg_parser.add_argument('--n_jobs', type=int, default=-1)
    arg_parser.add_argument('--batch_size', type=int, default=100)
    arg_parser.add_argument('--tokens_to_leaves', action='store_true')
    arg_parser.add_argument('--max_token_len', type=int, default=-1)
    arg_parser.add_argument('--max_label_len', type=int, default=-1)

    arg_parser.add_argument('--upload', action='store_true')
    arg_parser.add_argument('--store', choices=['s3', 'drive'], default='drive')
    arg_parser.add_argument('--tar_suffix', type=str, default='default')

    main(arg_parser.parse_args())
