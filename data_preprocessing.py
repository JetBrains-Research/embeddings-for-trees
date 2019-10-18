import os
from argparse import ArgumentParser, Namespace
from collections import Counter
from pickle import dump as pkl_dump
from shutil import rmtree
from subprocess import run as subprocess_run
from tarfile import open as tar_open
from tarfile import TarInfo
from typing import Tuple, Dict

import pandas as pd
from dgl import DGLGraph
from networkx.drawing.nx_pydot import read_dot
from requests import get
from tqdm import tqdm

from s3_worker import upload_file

data_folder = 'data'
holdout_folders = ['training', 'validation', 'test']

dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'
astminer_cli_path = 'astminer-cli.jar'

s3_bucket_name = 'voudy'

dataset_mapping = {
    'small': 'java-small',
    'medium': 'java-med',
    'large': 'java-large'
}


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


def convert_ast(ast_path: str, output_path: str, ast_description: pd.DataFrame) -> None:
    g_nx = read_dot(ast_path)
    g_dgl = DGLGraph(g_nx)
    g_dgl.ndata['token'] = ast_description['token_id'].values
    g_dgl.ndata['type'] = ast_description['type_id'].values
    label = ast_description['label'].values[0]
    with open(output_path, 'wb') as file_out:
        pkl_dump({'ast': g_dgl, 'method_name': label}, file_out)


def convert_project(
        holdout_asts_path: str, holdout_output_path: str, project_name: str,
        token_to_id: Dict, type_to_id: Dict
) -> None:
    project_path = os.path.join(holdout_asts_path, project_name, 'java')
    ast_folder = os.path.join(project_path, 'asts')
    description = pd.read_csv(os.path.join(project_path, 'description.csv'))
    description['token'].fillna(value='NAN', inplace=True)
    description['token_id'] = description['token'].apply(lambda token: token_to_id.get(token, 1))
    description['type_id'] = description['type'].apply(lambda type_all: type_to_id.get(type, 0))
    asts = os.listdir(ast_folder)
    print(f"working with {project_name}...")
    for ast in tqdm(asts):
        convert_ast(
            os.path.join(project_path, 'asts', ast),
            os.path.join(holdout_output_path, f'{project_name}_{ast[:-4]}.pkl'),
            description[description['dot_file'] == ast].sort_values(by='node_id')
        )


def convert_holdout(data_path: str, holdout_name: str, token_to_id: Dict, type_to_id: Dict) -> str:
    print(f"Convert asts for {holdout_name} data...")
    holdout_path = os.path.join(data_path, f'{holdout_name}_asts')
    output_holdout_path = os.path.join(data_path, f'{holdout_name}_preprocessed')
    create_folder(output_holdout_path)
    projects = os.listdir(holdout_path)
    for project in tqdm(projects):
        convert_project(holdout_path, output_holdout_path, project, token_to_id, type_to_id)
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
        token_to_id, type_to_id = collect_vocabulary(os.path.join(data_path, f'{holdout_folders[0]}_asts'))
        holdout_preprocessed_paths = {}
        # for holdout in holdout_folders:
        for holdout in ['test']:
            holdout_preprocessed_paths[holdout] = convert_holdout(
                data_path, holdout, token_to_id, type_to_id
            )
    else:
        holdout_preprocessed_paths = {
            holdout: os.path.join(data_path, f'{holdout}_preprocessed') for holdout in holdout_folders
        }

    if not all([os.path.exists(path[1]) for path in holdout_preprocessed_paths.items()]):
        raise RuntimeError("convert ast before uploading or using it via --convert arg")

    for holdout, path in holdout_preprocessed_paths.items():
        number_of_functions = len(os.listdir(path))
        print(f"There are {number_of_functions} functions in {holdout} data")

    if args.upload:
        for holdout, path in holdout_preprocessed_paths.items():
            tar_file_name = os.path.join(data_path, f'{holdout}_preprocessed.tar.gz')
            with tar_open(tar_file_name, 'w:gz') as tar_file:
                for file in os.listdir(path):
                    tar_file.addfile(TarInfo(file), open(os.path.join(path, file)))
            upload_file(tar_file_name, s3_bucket_name)


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(dataset_mapping.keys()))
    arg_parser.add_argument('--download', action='store_true')
    arg_parser.add_argument('--build_ast', action='store_true')
    arg_parser.add_argument('--convert', action='store_true')
    arg_parser.add_argument('--upload', action='store_true')

    main(arg_parser.parse_args())
