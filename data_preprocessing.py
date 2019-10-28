import os
from argparse import ArgumentParser, Namespace
from collections import Counter
from pickle import dump as pkl_dump
from pickle import load as pkl_load
from shutil import rmtree
from subprocess import run as subprocess_run
from tarfile import open as tar_open
from tarfile import TarInfo
from typing import Tuple, Dict, List
from multiprocessing import Pool, cpu_count, Manager, Queue

import pandas as pd
import numpy as np
from dgl import DGLGraph
from dgl import batch as dgl_batch
from networkx.drawing.nx_pydot import read_dot
from requests import get
from tqdm.auto import tqdm
from torch import tensor

from s3_worker import upload_file

data_folder = 'data'
vocabulary_name = 'vocabulary.pkl'
holdout_folders = ['training', 'validation', 'test']

dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'
astminer_cli_path = 'astminer-cli.jar'

s3_bucket_name = 'voudy'

dataset_mapping = {
    'small': 'java-small',
    'medium': 'java-med',
    'large': 'java-large'
}

random_seed = 7
np.random.seed(random_seed)


def create_folder(path: str, is_clean: bool = True) -> None:
    if is_clean and os.path.exists(path):
        rmtree(path)
    os.mkdir(path)


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


def extract_dataset(file_path: str, extract_path: str, folder_name: str) -> [str]:
    def tqdm_progress(members):
        extract_progress_bar = tqdm(total=len(list(members.getnames())))
        for member in members:
            extract_progress_bar.update()
            yield member
        extract_progress_bar.close()

    with tar_open(file_path, 'r:gz') as tarball:
        tarball.extractall(extract_path, members=tqdm_progress(tarball))
    return [os.path.join(extract_path, folder_name, folder) for folder in holdout_folders]


def build_project_asts(project_path: str, output_path: str) -> bool:
    completed_process = subprocess_run(
        ['java', '-Xmx15g', '-jar', astminer_cli_path, 'parse',
         '--project', project_path, '--output', output_path,
         '--storage', 'dot', '--granularity', 'method', '--lang', 'java', '--hide-method-name']
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
        project_path = os.path.join(data_path, holdout_name, project)
        output_project_path = os.path.join(output_folder_path, project)
        create_folder(output_project_path)
        if build_project_asts(project_path, output_project_path):
            successful_builds += 1
    print(f"create asts for {successful_builds}/{len(projects)} {holdout_name} projects")
    return output_folder_path


def convert_ast(ast_path: str, description: pd.DataFrame) -> Tuple[DGLGraph, str]:
    g_nx = read_dot(ast_path)
    g_dgl = DGLGraph(g_nx)
    mask = description['dot_file'] == ast_path
    g_dgl.ndata['token'] = description.loc[mask, 'token_id'].values
    g_dgl.ndata['type'] = description.loc[mask, 'type_id'].values
    label = description.loc[mask, 'label'].values[0]
    return g_dgl, label


def collect_ast_description(projects_paths: List[str], asts_batch: List[str]) -> pd.DataFrame:
    asts_per_project = {
        project_path: [ast for ast in asts_batch if ast.startswith(project_path)] for project_path in projects_paths
    }
    asts_description = pd.DataFrame()
    for project_path, asts in asts_per_project.items():
        if len(asts) == 0:
            continue
        project_description = pd.read_csv(os.path.join(project_path, 'description.csv'))
        ast_names = [os.path.basename(ast) for ast in asts]
        mask = project_description['dot_file'].isin(ast_names)
        project_description.loc[mask, 'dot_file'] = project_description.loc[mask, 'dot_file'].apply(
            lambda dot_file: os.path.join(project_path, 'asts', dot_file)
        )
        asts_description = pd.concat([asts_description, project_description[mask]], ignore_index=True)
    return asts_description


def convert_holdout(data_path: str, holdout_name: str, token_to_id: Dict,
                    type_to_id: Dict, n_jobs: int, batch_size: int) -> str:
    print(f"Convert asts for {holdout_name} data...")
    holdout_path = os.path.join(data_path, f'{holdout_name}_asts')
    output_holdout_path = os.path.join(data_path, f'{holdout_name}_preprocessed')
    create_folder(output_holdout_path)
    projects_paths = [os.path.join(holdout_path, project, 'java') for project in os.listdir(holdout_path)]
    asts = [os.path.join(project_path, 'asts', ast)
            for project_path in projects_paths
            for ast in os.listdir(os.path.join(project_path, 'asts'))
            ]
    np.random.shuffle(asts)
    n_batches = len(asts) // batch_size + (1 if len(asts) % batch_size > 0 else 0)
    pool = Pool(cpu_count() if n_jobs == -1 else n_jobs)
    for batch_num in tqdm(range(n_batches)):
        current_asts = asts[batch_num * batch_size: min((batch_num + 1) * batch_size, len(asts))]
        current_description = collect_ast_description(projects_paths, current_asts)
        current_description['token'].fillna(value='NAN', inplace=True)
        current_description['token_id'] = current_description['token'].apply(lambda token: token_to_id.get(token, 1))
        current_description['type_id'] = current_description['type'].apply(lambda type_all: type_to_id.get(type, 0))
        batch = pool.starmap_async(convert_ast, [(ast, current_description) for ast in current_asts]).get()
        graphs, labels = map(list, zip(*batch))
        batched_graph = dgl_batch(graphs)
        with open(os.path.join(output_holdout_path, f'batch_{batch_num}.pkl'), 'wb') as pkl_file:
            pkl_dump({'bathed_graph': batched_graph, 'labels': labels}, pkl_file)
    pool.close()
    return output_holdout_path


def collect_vocabulary(train_path: str, n_most_common_tokens: int = 1_000_000) -> Tuple[Dict, Dict]:
    token_vocabulary = Counter()
    type_vocabulary = Counter()
    projects = os.listdir(train_path)
    print("collect vocabulary from training holdout")
    for project in tqdm(projects):
        project_description = pd.read_csv(os.path.join(train_path, project, 'java', 'description.csv'))
        token_vocabulary.update(project_description['token'].values)
        type_vocabulary.update(project_description['type'].values)
    del token_vocabulary['METHOD_NAME']
    del token_vocabulary['NAN']
    print(f"found {len(token_vocabulary)} tokens, using {n_most_common_tokens} from it")
    print(f"found {len(type_vocabulary)} types, using all of them")
    token_to_id = {
        'UNK': 0,
        'METHOD_NAME': 1,
        'NAN': 2
    }
    token_to_id.update(
        [(token[0], num + 3) for num, token in enumerate(token_vocabulary.most_common(n_most_common_tokens))]
    )
    type_to_id = {
        'UNK': 0
    }
    type_to_id.update([(node_type[0], num + 1) for num, node_type in enumerate(type_vocabulary)])
    return token_to_id, type_to_id


def main(args: Namespace) -> None:
    dataset_name = dataset_mapping[args.dataset]
    data_path = os.path.join(data_folder, dataset_name)

    if args.download:
        create_folder(data_path)
        print(f"download {dataset_name} dataset...")
        tar_file_path = download_dataset(dataset_name, data_folder)
        print(f"extract files from tar archive {tar_file_path}...")
        train_path, val_path, test_path = extract_dataset(tar_file_path, data_folder, dataset_name)
        print("remove tar file...")
        os.remove(tar_file_path)
    else:
        train_path, val_path, test_path = [os.path.join(data_path, folder) for folder in holdout_folders]

    if not all([os.path.exists(holdout_path) for holdout_path in [train_path, val_path, test_path]]):
        raise RuntimeError("download and extract data before processing it via --download arg")

    if args.build_ast:
        if not os.path.exists(astminer_cli_path):
            raise RuntimeError(f"can't find astminer-cli in this location {astminer_cli_path}")
        holdout_ast_paths = {}
        for holdout in holdout_folders:
            holdout_ast_paths[holdout] = build_holdout_asts(data_path, holdout)
    else:
        holdout_ast_paths = {
            holdout: os.path.join(data_path, f'{holdout}_asts') for holdout in holdout_folders
        }

    if not all([os.path.exists(path[1]) for path in holdout_ast_paths.items()]):
        raise RuntimeError("build ast before converting it via --build_ast arg")

    if args.convert:
        vocabulary_path = os.path.join(data_path, vocabulary_name)
        if not os.path.exists(vocabulary_path):
            token_to_id, type_to_id = collect_vocabulary(os.path.join(data_path, f'{holdout_folders[0]}_asts'))
            with open(vocabulary_path, 'wb') as pkl_file:
                pkl_dump({'token_to_id': token_to_id, 'type_to_id': type_to_id}, pkl_file)
        else:
            with open(vocabulary_path, 'rb') as pkl_file:
                pkl_data = pkl_load(pkl_file)
                token_to_id = pkl_data['token_to_id']
                type_to_id = pkl_data['type_to_id']

        holdout_preprocessed_paths = {}
        for holdout in holdout_folders:
            holdout_preprocessed_paths[holdout] = convert_holdout(
                data_path, holdout, token_to_id, type_to_id, args.n_jobs, args.batch_size
            )
    else:
        holdout_preprocessed_paths = {
            holdout: os.path.join(data_path, f'{holdout}_preprocessed') for holdout in holdout_folders
        }

    if not all([os.path.exists(path[1]) for path in holdout_preprocessed_paths.items()]):
        raise RuntimeError("convert ast before uploading or using it via --convert arg")

    for holdout, path in holdout_preprocessed_paths.items():
        number_of_batches = len(os.listdir(path))
        print(f"There are {number_of_batches} batches in {holdout} data")

    if args.upload:
        for holdout, path in holdout_preprocessed_paths.items():
            tar_file_name = os.path.join(data_path, f'{holdout}_preprocessed.tar.gz')
            with tar_open(tar_file_name, 'w:gz') as tar_file:
                for file in tqdm(os.listdir(path)):
                    tar_file.add(os.path.join(path, file), file)
            upload_file(tar_file_name, s3_bucket_name)


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(dataset_mapping.keys()))
    arg_parser.add_argument('--download', action='store_true')
    arg_parser.add_argument('--build_ast', action='store_true')
    arg_parser.add_argument('--convert', action='store_true')
    arg_parser.add_argument('--upload', action='store_true')
    arg_parser.add_argument('--n_jobs', type=int, default=-1)
    arg_parser.add_argument('--batch_size', type=int, default=100)

    main(arg_parser.parse_args())
