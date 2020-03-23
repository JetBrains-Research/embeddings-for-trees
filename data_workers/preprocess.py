import os
from argparse import ArgumentParser, Namespace
from pickle import load as pickle_load

from data_workers.convert import convert_holdout
from data_workers.preprocess_steps import download_dataset, build_holdout_asts, collect_vocabulary, upload_dataset
from utils.common import create_folder, fix_seed

data_folder = 'data'
vocabulary_name = 'vocabulary.pkl'
holdout_folders = ['training', 'validation', 'test']

dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'
astminer_cli_path = 'utils/astminer-cli.jar'

s3_bucket_name = 'voudy'

dataset_mapping = {
    'small': 'java-small',
    'medium': 'java-med',
    'large': 'java-large',
    'test': 'java-test'
}


def main(args: Namespace) -> None:
    fix_seed()
    dataset_name = dataset_mapping[args.dataset]
    data_path = os.path.join(data_folder, dataset_name)
    create_folder(data_folder, is_clean=False)
    create_folder(data_path, is_clean=False)

    if args.download:
        src_paths = download_dataset(dataset_name, data_path, dataset_url, holdout_folders)
    else:
        src_paths = [os.path.join(data_path, folder) for folder in holdout_folders]

    if args.build_ast:
        if not all([os.path.exists(holdout_path) for holdout_path in src_paths]):
            raise RuntimeError("download and extract data before building ast")
        if not os.path.exists(astminer_cli_path):
            raise RuntimeError(f"can't find astminer-cli in this location {astminer_cli_path}")
        ast_paths = [build_holdout_asts(data_path, holdout, astminer_cli_path) for holdout in holdout_folders]
    else:
        ast_paths = [os.path.join(data_path, f'{holdout}_asts') for holdout in holdout_folders]

    vocabulary_path = os.path.join(data_path, vocabulary_name)
    if args.collect_vocabulary:
        if not os.path.exists(ast_paths[0]):
            raise RuntimeError("build training asts before collecting vocabulary")
        collect_vocabulary(
            ast_paths[0], vocabulary_path, args.n_tokens, args.n_types, args.n_labels, args.split_vocabulary
        )

    if args.convert:
        if not all([os.path.exists(ast_path) for ast_path in ast_paths]):
            raise RuntimeError("build ast before converting it to DGL format")
        if not os.path.exists(vocabulary_path):
            raise RuntimeError("collect vocabulary before using it")
        with open(vocabulary_path, 'rb') as pkl_file:
            vocab = pickle_load(pkl_file)
        token_to_id, type_to_id, label_to_id = vocab['token_to_id'], vocab['type_to_id'], vocab['label_to_id']
        preprocessed_paths = [convert_holdout(
                data_path, holdout, args.batch_size, token_to_id, type_to_id, label_to_id, args.tokens_to_leaves,
                args.split_vocabulary, args.max_token_len, args.max_label_len, '|', True, args.n_jobs
            ) for holdout in holdout_folders]
    else:
        preprocessed_paths = [os.path.join(data_path, f'{holdout}_preprocessed') for holdout in holdout_folders]

    if args.upload:
        if not all([os.path.exists(path) for path in preprocessed_paths]):
            raise RuntimeError("convert ast before uploading it")
        upload_dataset(dataset_name, args.tar_suffix, data_path, vocabulary_name, holdout_folders, s3_bucket_name)

    if all([os.path.exists(path) for path in preprocessed_paths]):
        for holdout, path in zip(holdout_folders, preprocessed_paths):
            number_of_batches = len(os.listdir(path))
            print(f"There are {number_of_batches} batches in {holdout} data")


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(dataset_mapping.keys()))
    arg_parser.add_argument('--download', action='store_true')

    arg_parser.add_argument('--build_ast', action='store_true')

    arg_parser.add_argument('--collect_vocabulary', action='store_true')
    arg_parser.add_argument('--n_tokens', type=int, default=-1)
    arg_parser.add_argument('--n_types', type=int, default=-1)
    arg_parser.add_argument('--n_labels', type=int, default=-1)
    arg_parser.add_argument('--split_vocabulary', action='store_true')

    arg_parser.add_argument('--convert', action='store_true')
    arg_parser.add_argument('--n_jobs', type=int, default=-1)
    arg_parser.add_argument('--batch_size', type=int, default=100)
    arg_parser.add_argument('--tokens_to_leaves', action='store_true')
    arg_parser.add_argument('--max_token_len', type=int, default=-1)
    arg_parser.add_argument('--max_label_len', type=int, default=-1)

    arg_parser.add_argument('--upload', action='store_true')
    arg_parser.add_argument('--tar_suffix', type=str, default='default')

    main(arg_parser.parse_args())
