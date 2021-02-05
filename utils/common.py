import subprocess
from os.path import join

SEPARATOR = "|"

# data storage keys
LABEL = "label"
AST = "AST"
NODE = "node"
TOKEN = "token"
CHILDREN = "children"

# vocabulary keys
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"


def get_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path], capture_output=True, encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(f"Counting lines in {file_path} failed with error\n{command_result.stderr}")
    return int(command_result.stdout.split()[0])


def download_dataset(url: str, dataset_dir: str, dataset_name: str):
    download_command_result = subprocess.run(["wget", url, "-P", dataset_dir], capture_output=True, encoding="utf-8")
    if download_command_result.returncode != 0:
        raise RuntimeError(f"Failed to download dataset. Error: {download_command_result.stderr}")
    tar_name = join(dataset_dir, f"{dataset_name}.tar.gz")
    untar_command_result = subprocess.run(["tar", "-xzvf", tar_name], capture_output=True, encoding="utf-8")
    if untar_command_result.returncode != 0:
        raise RuntimeError(f"Failed to untar dataset. Error: {untar_command_result.stderr}")
