import os
from argparse import ArgumentParser, Namespace
from collections import Counter
from pickle import dump as pkl_dump
from pickle import load as pkl_load
from subprocess import run as subprocess_run
from tarfile import open as tar_open
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from requests import get
from tqdm.auto import tqdm

from data_workers.convert import convert_holdout
from utils.common import extract_tar_gz, create_folder, UNK
from utils.s3_worker import upload_file, download_file

data_folder = 'data'
vocabulary_name = 'vocabulary.pkl'
holdout_folders = ['training', 'validation', 'test']

dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'
astminer_cli_path = 'utils/astminer-cli.jar'

s3_bucket_name = 'voudy'

dataset_mapping = {
    'small': 'java-small',
    'medium': 'java-med',
    'large': 'java-large'
}

random_seed = 7
np.random.seed(random_seed)


def download_dataset(name: str, download_path: str, block_size: int = 1024) -> str:
    r = get(dataset_url.format(name), stream=True)
    assert r.status_code == 200
    total_size = int(r.headers.get('content-length', 0))
    download_progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    file_path = os.path.join(download_path, f'{name}.tar.gz')
    with open(file_path, 'wb') as f:
        for data in r.iter_content(chunk_size=block_size):
            download_progress_bar.update(len(data))
            f.write(data)
    download_progress_bar.close()
    return file_path


def extract_dataset(file_path: str, extract_path: str, dataset_name: str) -> [str]:
    extract_tar_gz(file_path, extract_path)
    return [os.path.join(extract_path, dataset_name, folder) for folder in holdout_folders]


def build_project_asts(project_path: str, output_path: str) -> bool:
    completed_process = subprocess_run(
        ['java', '-Xmx30g', '-jar', astminer_cli_path, 'parse',
         '--project', project_path, '--output', output_path,
         '--storage', 'dot', '--granularity', 'method',
         '--lang', 'java', '--hide-method-name', '--split-tokens',
         '--filter-modifiers', 'abstract', '--filter-annotations', 'Override',
         '--remove-constructors', '--remove-nodes', 'Javadoc',
         '--java-parser', 'gumtree']
    )
    if completed_process.returncode != 0:
        print(f"can't build ASTs for project {project_path}, failed with:\n{completed_process.stdout}")
        return False
    return True


def build_holdout_asts(data_path: str, holdout_name: str) -> str:
    print(f"build asts for {holdout_name} data...")
    projects = os.listdir(os.path.join(data_path, holdout_name))
    output_folder_path = os.path.join(data_path, f'{holdout_name}_asts')
    create_folder(output_folder_path)
    successful_builds = 0
    for project in tqdm(projects):
        print(f"working with {project} project")
        project_path = os.path.join(data_path, holdout_name, project)
        output_project_path = os.path.join(output_folder_path, project)
        create_folder(output_project_path)
        if build_project_asts(project_path, output_project_path):
            successful_builds += 1
            desc_path = os.path.join(output_project_path, 'java', 'description.csv')

            # remove asts with nan labels
            project_description = pd.read_csv(desc_path)
            bad_labels_mask = project_description['label'].isna()
            filenames = project_description[bad_labels_mask]['dot_file'].unique()
            source_files = project_description[bad_labels_mask]['source_file'].unique()
            print(f"remove functions from {source_files} for {project} project")
            for filename in filenames:
                filepath = os.path.join(output_project_path, 'java', 'asts', filename)
                os.remove(filepath)
            project_description.dropna(subset=['label'], inplace=True)
            project_description.to_csv(desc_path, index=False)
    print(f"create asts for {successful_builds}/{len(projects)} {holdout_name} projects")
    return output_folder_path


def collect_vocabulary(train_path: str) -> Tuple[Dict, Dict, Dict]:
    token_vocabulary = Counter()
    type_vocabulary = Counter()
    label_vocabulary = Counter()
    projects = os.listdir(train_path)
    print("collect vocabulary from training holdout")
    for project in tqdm(projects):
        project_description = pd.read_csv(os.path.join(train_path, project, 'java', 'description.csv'))
        project_description['token'].fillna('NAN', inplace=True)
        token_vocabulary.update(project_description['token'].values)
        type_vocabulary.update(project_description['type'].values)
        label_vocabulary.update(project_description['label'].values)
    print(f"found {len(token_vocabulary)} tokens")
    print(f"found {len(type_vocabulary)} types")
    print(f"found {len(label_vocabulary)} labels")
    token_to_id = {UNK: 0}
    type_to_id = {UNK: 0}
    label_to_id = {UNK: 0}
    for id_dict, counter in \
            [(token_to_id, token_vocabulary), (type_to_id, type_vocabulary), (label_to_id, label_vocabulary)]:
        id_dict.update(
            [(token, num + 1) for num, token in enumerate(counter)]
        )
    return token_to_id, type_to_id, label_to_id


def main(args: Namespace) -> None:
    dataset_name = dataset_mapping[args.dataset]
    data_path = os.path.join(data_folder, dataset_name)
    create_folder(data_folder, is_clean=False)
    create_folder(data_path, is_clean=False)

    if args.download:
        print(f"download {dataset_name} dataset...")
        tar_file_path = download_dataset(dataset_name, data_folder)
        print(f"extract files from tar archive {tar_file_path}...")
        train_path, val_path, test_path = extract_dataset(tar_file_path, data_folder, dataset_name)
        print("remove tar file...")
        os.remove(tar_file_path)
    else:
        train_path, val_path, test_path = [os.path.join(data_path, folder) for folder in holdout_folders]

    if args.build_ast:
        if not all([os.path.exists(holdout_path) for holdout_path in [train_path, val_path, test_path]]):
            raise RuntimeError("download and extract data before processing it via --download arg")
        if not os.path.exists(astminer_cli_path):
            raise RuntimeError(f"can't find astminer-cli in this location {astminer_cli_path}")
        holdout_ast_paths = {}
        for holdout in holdout_folders:
            holdout_ast_paths[holdout] = build_holdout_asts(data_path, holdout)
    else:
        holdout_ast_paths = {
            holdout: os.path.join(data_path, f'{holdout}_asts') for holdout in holdout_folders
        }

    vocabulary_path = os.path.join(data_path, vocabulary_name)
    if args.collect_vocabulary:
        token_to_id, type_to_id, label_to_id = collect_vocabulary(os.path.join(data_path, f'{holdout_folders[0]}_asts'))
        with open(vocabulary_path, 'wb') as pkl_file:
            pkl_dump({'token_to_id': token_to_id, 'type_to_id': type_to_id, 'label_to_id': label_to_id}, pkl_file)

    if args.convert:
        if not all([os.path.exists(path[1]) for path in holdout_ast_paths.items()]):
            raise RuntimeError("build ast before converting it via --build_ast arg")
        if not os.path.exists(vocabulary_path):
            raise RuntimeError("collect vocabulary before converting it via --build_ast arg")
        with open(vocabulary_path, 'rb') as pkl_file:
            pkl_data = pkl_load(pkl_file)
            token_to_id = pkl_data['token_to_id']
            type_to_id = pkl_data['type_to_id']

        holdout_preprocessed_paths = {}
        for holdout in holdout_folders:
            holdout_preprocessed_paths[holdout] = convert_holdout(
                data_path, holdout, token_to_id, type_to_id, args.n_jobs, args.batch_size, args.high_memory
            )
    else:
        holdout_preprocessed_paths = {
            holdout: os.path.join(data_path, f'{holdout}_preprocessed') for holdout in holdout_folders
        }

    if args.upload:
        if not all([os.path.exists(path[1]) for path in holdout_preprocessed_paths.items()]):
            raise RuntimeError("convert ast before uploading or using it via --convert arg")
        tar_file_name = f'{dataset_name}_{args.tar_suffix}.tar.gz'
        completed_process = subprocess_run(
            ['tar', '-czf', tar_file_name, vocabulary_name] +
            [f'{holdout}_preprocessed' for holdout in holdout_folders],
            cwd=data_path
        )
        if completed_process.returncode != 0:
            print(f"can't create tar for preprocessed data, failed with\n{completed_process.stdout}")
        else:
            upload_file(os.path.join(data_path, tar_file_name), s3_bucket_name, tar_file_name)

    if args.download_preprocessed:
        for holdout, path in holdout_preprocessed_paths.items():
            tar_file_name = f'{dataset_name}_{holdout}_preprocessed.tar.gz'
            tar_path = os.path.join(data_path, tar_file_name)
            download_file(tar_path, s3_bucket_name, tar_file_name)
            create_folder(path)
            extract_tar_gz(tar_path, path)
        vocabulary_path = os.path.join(data_path, vocabulary_name)
        download_file(vocabulary_path, s3_bucket_name, f'{dataset_name}_{vocabulary_name}')

    if all([os.path.exists(holdout_path) for _, holdout_path in holdout_preprocessed_paths.items()]):
        for holdout, path in holdout_preprocessed_paths.items():
            number_of_batches = len(os.listdir(path))
            print(f"There are {number_of_batches} batches in {holdout} data")


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(dataset_mapping.keys()))
    arg_parser.add_argument('--download', action='store_true')
    arg_parser.add_argument('--build_ast', action='store_true')
    arg_parser.add_argument('--collect_vocabulary', action='store_true')
    arg_parser.add_argument('--convert', action='store_true')
    arg_parser.add_argument('--upload', action='store_true')
    arg_parser.add_argument('--download_preprocessed', action='store_true')
    arg_parser.add_argument('--n_jobs', type=int, default=-1)
    arg_parser.add_argument('--batch_size', type=int, default=100)
    arg_parser.add_argument('--high_memory', action='store_true')
    arg_parser.add_argument('--tar_suffix', type=str, default='default')

    main(arg_parser.parse_args())
