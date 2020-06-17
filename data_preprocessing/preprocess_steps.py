import os
from collections import Counter
from pickle import dump as pickle_dump
from subprocess import run as subprocess_run
from typing import List

import pandas as pd
from requests import get
from tqdm.auto import tqdm

from data_information import IDatasetInfo
from data_preprocessing.drive_workers import upload_file as drive_upload_file
from data_preprocessing.s3_worker import upload_file as s3_upload_file
from utils.common import extract_tar_gz, create_folder, UNK, PAD, SOS, EOS


def _download_dataset_archive(dataset_info: IDatasetInfo, dataset_path: str, block_size: int = 1024) -> str:
    r = get(dataset_info.url, stream=True)
    assert r.status_code == 200
    total_size = int(r.headers.get('content-length', 0))
    download_progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    file_path = os.path.join(dataset_path, f'{dataset_info.name}.tar.gz')
    with open(file_path, 'wb') as f:
        for data in r.iter_content(chunk_size=block_size):
            download_progress_bar.update(len(data))
            f.write(data)
    download_progress_bar.close()
    return file_path


def download_dataset(dataset_info: IDatasetInfo, dataset_path: str) -> None:
    print(f"download {dataset_info.name} dataset...")
    tar_file_path = _download_dataset_archive(dataset_info, dataset_path)
    print(f"extract files from tar archive {tar_file_path}...")
    extract_tar_gz(tar_file_path, os.path.dirname(dataset_path))
    print("remove tar file...")
    os.remove(tar_file_path)


def build_asts(input_path: str, output_path: str, astminer_path: str, astminer_params: List[str]) -> bool:
    completed_process = subprocess_run(['java', '-Xmx30g', '-jar', astminer_path, 'parse', '--project',
                                        input_path, '--output', output_path, *astminer_params])
    if completed_process.returncode != 0:
        print(f"can't build ASTs for project {input_path}, failed with:\n{completed_process.stdout}")
        return False
    return True


def build_projects_asts(projects_folder: str, output_folder: str, astminer_path: str, astminer_params: List[str]) -> int:
    print(f"build asts for projects in {projects_folder} folder")
    projects = os.listdir(projects_folder)
    successful_builds = 0
    for project in tqdm(projects):
        print(f"build asts for {project} project")
        project_path = os.path.join(projects_folder, project)
        output_path = os.path.join(output_folder, project)
        create_folder(output_path)
        if build_asts(project_path, output_path, astminer_path, astminer_params):
            successful_builds += 1
    print(f"create asts for {successful_builds} out of {len(projects)} projects")
    return successful_builds


def build_dataset_asts(dataset_info: IDatasetInfo, dataset_path: str, astminer_path: str) -> None:
    for holdout in dataset_info.holdout_folders:
        holdout_folder = os.path.join(dataset_path, holdout)
        output_folder = os.path.join(dataset_path, f'{holdout}_asts')
        create_folder(output_folder)
        build_projects_asts(holdout_folder, output_folder, astminer_path, dataset_info.astminer_params)


def _update_vocab_counter(counter: Counter, values: List, is_split: bool = False, delimiter: str = '|') -> Counter:
    values = filter(lambda t: isinstance(t, str), values)
    if is_split:
        _values = map(lambda t: t.split(delimiter), values)
        values = []
        for _sv in _values:
            values += _sv
    counter.update(values)
    return counter


def collect_vocabulary(
        train_path: str, vocabulary_path: str, n_tokens: int = -1, n_types: int = -1, n_labels: int = -1,
        is_split: bool = False, wrap_tokens: bool = False, wrap_labels: bool = False, delimiter: str = '|'
):
    token_to_id = {UNK: 0, PAD: 1}
    type_to_id = {UNK: 0, PAD: 1}
    label_to_id = {UNK: 0, PAD: 1}

    if wrap_labels:
        label_to_id[SOS] = 2
        label_to_id[EOS] = 3
    if wrap_tokens:
        token_to_id[SOS] = 2
        token_to_id[EOS] = 3

    projects = os.listdir(train_path)
    print("collect vocabulary from training holdout")
    for id_dict, n_max, column in [
        (token_to_id, n_tokens, 'token'), (type_to_id, n_types, 'type'), (label_to_id, n_labels, 'label')
    ]:
        print(f"collecting {column} vocabulary...")
        counter = Counter()
        for project in tqdm(projects):
            project_description = pd.read_csv(os.path.join(train_path, project, 'java', 'description.csv'))
            counter = _update_vocab_counter(counter, project_description[column], is_split, delimiter)
        if n_max == -1:
            n_max = len(counter)
        print(f"found {len(counter)} {column}, use {n_max} most common")
        st_index = len(id_dict)
        id_dict.update(
            [(token, num + st_index) for num, (token, _) in enumerate(counter.most_common(n_max))]
        )

    assert all([t in token_to_id for t in [UNK, PAD] + ([SOS, EOS] if wrap_tokens else [])])
    assert all([t in type_to_id for t in [UNK, PAD]])
    assert all([t in label_to_id for t in [UNK, PAD] + ([SOS, EOS] if wrap_labels else [])])

    with open(vocabulary_path, 'wb') as vocab_file:
        pickle_dump({
            'token_to_id': token_to_id, 'type_to_id': type_to_id, 'label_to_id': label_to_id,
        }, vocab_file)


def upload_dataset(dataset_info: IDatasetInfo, dataset_path: str, vocabulary_name: str, store: str, tar_suffix: str) -> None:
    tar_file_name = f'{dataset_info.name}_{tar_suffix}.tar.gz'
    completed_process = subprocess_run(
        ['tar', '-czf', tar_file_name, vocabulary_name] +
        [f'{holdout}_preprocessed' for holdout in dataset_info.holdout_folders],
        cwd=dataset_path
    )
    if completed_process.returncode != 0:
        print(f"can't create tar for preprocessed data, failed with\n{completed_process.stdout}")
        return
    if store == 's3':
        s3_upload_file(os.path.join(dataset_path, tar_file_name), tar_file_name)
    elif store == 'drive':
        drive_upload_file(os.path.join(dataset_path, tar_file_name))
    else:
        raise ValueError("Unsupported store, try one of: s3, drive")
