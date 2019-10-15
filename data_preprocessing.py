import os
from argparse import ArgumentParser, Namespace
from shutil import rmtree
from tarfile import open as tar_open
from subprocess import run as subprocess_run
from typing import Tuple
from requests import get
from tqdm import tqdm

data_folder = 'data'
holdout_folders = ['training', 'validation', 'test']

dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'
astminer_cli_path = 'astminer-cli.jar'

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
        ['java', '-jar', astminer_cli_path, 'parse',
         '--project', project_path, '--output', output_path,
         '--storage', 'dot', '--granularity', 'method', '--lang', 'java']
    )
    if completed_process.returncode != 0:
        print(f"can't build ASTs for project {project_path}, failed with:\n{completed_process.stdout}")
        return False
    return True


def build_holdout_asts(data_path: str, holdout_name: str) -> Tuple[int, int]:
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
    return successful_builds, len(projects)


def main(args: Namespace):
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

    train_projects = os.listdir(train_path)
    val_projects = os.listdir(val_path)
    test_projects = os.listdir(test_path)
    print(f"found {len(train_projects)} train projects, " +
          f"{len(val_projects)} validation projects, {len(test_projects)} test projects")

    if args.build_ast:
        if not os.path.exists(astminer_cli_path):
            raise RuntimeError(f"can't find astminer-cli in this location {astminer_cli_path}")
        for holdout in holdout_folders:
            success, total = build_holdout_asts(data_path, holdout)
            print(f"create asts for {success}/{total} {holdout} projects")


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(dataset_mapping.keys()))
    arg_parser.add_argument('--download', action='store_true')
    arg_parser.add_argument('--build_ast', action='store_true')

    main(arg_parser.parse_args())
