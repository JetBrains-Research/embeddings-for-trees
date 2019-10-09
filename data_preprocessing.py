import os
from argparse import ArgumentParser
from shutil import rmtree
from tarfile import open as tar_open
from requests import get
from tqdm import tqdm

data_folder = 'data'

dataset_url = 'https://s3.amazonaws.com/code2seq/datasets/{}.tar.gz'

dataset_mapping = {
    'small': 'java-small',
    'medium': 'java-med',
    'large': 'java-large'
}


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
    return [os.path.join(extract_path, folder_name, folder) for folder in ['training', 'validation', 'test']]


if __name__ == '__main__':
    arg_parser = ArgumentParser(description="download and preprocess data from Java small/medium/large dataset")
    arg_parser.add_argument('dataset', choices=list(dataset_mapping.keys()))
    arg_parser.add_argument('--force', action='store_true')
    args = arg_parser.parse_args()

    dataset_name = dataset_mapping[args.dataset]
    data_path = os.path.join(data_folder, dataset_name)
    if os.path.exists(data_path):
        if args.force:
            rmtree(data_path)
        else:
            print(f"{args.dataset} dataset has already been preprocessed")
            exit(0)

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    print(f"download {dataset_name} dataset...")
    tar_file_path = download_dataset(dataset_name, data_folder)
    print(f"extract files from tar archive {tar_file_path}...")
    train_path, val_path, test_path = extract_dataset(tar_file_path, data_folder, dataset_name)
    print("remove tar file...")
    os.remove(tar_file_path)
