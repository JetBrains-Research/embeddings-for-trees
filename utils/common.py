import subprocess

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
    command_result = subprocess.run(
        ["wc", "-l", file_path], capture_output=True, encoding="utf-8"
    )
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    return int(command_result.stdout.split()[0])
