import os
from argparse import ArgumentParser
from pickle import load as pkl_load
from subprocess import run as subprocess_run

import pandas as pd
import torch

from data_workers.convert import convert_dot_to_dgl
from model.tree2seq import load_model
from utils.common import fix_seed, get_device, SOS, EOS, split_tokens_to_subtokens

TMP_FOLDER = '.tmp'
astminer_cli_path = 'utils/astminer-cli.jar'

vocab_path = 'data/java-small/java-small_vocabulary.pkl'
labels_path = 'data/java-small/java-small_labels.pkl'


def build_ast(path_to_function: str) -> bool:
    completed_process = subprocess_run(
        ['java', '-jar', astminer_cli_path, 'parse',
         '--project', path_to_function, '--output', TMP_FOLDER,
         '--storage', 'dot', '--granularity', 'method',
         '--lang', 'java', '--hide-method-name', '--split-tokens',
         '--remove-nodes', 'Javadoc']
    )
    if completed_process.returncode != 0:
        print(f"can't build AST for given function, failed with:\n{completed_process.stdout}")
        return False
    return True


def interactive(path_to_function: str, path_to_model: str):
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    print(f"prepare ast...")
    if not build_ast(path_to_function):
        return
    ast_folder = os.path.join(TMP_FOLDER, 'java', 'asts')
    ast = os.listdir(ast_folder)
    if len(ast) == 0:
        print("didn't find any functions in given file")
        return
    if len(ast) > 1:
        print("too many functions in given file, for interactive prediction you need only one")
        return
    dgl_ast = convert_dot_to_dgl(os.path.join(ast_folder, ast[0]))
    ast_desc = pd.read_csv(os.path.join(TMP_FOLDER, 'java', 'description.csv'))
    with open(vocab_path, 'rb') as pkl_file:
        vocab = pkl_load(pkl_file)
        token_to_id, type_to_id = vocab['token_to_id'], vocab['type_to_id']
    ast_desc['token_id'] = ast_desc['token'].apply(lambda _t: token_to_id.get(_t, 0))
    ast_desc['type_id'] = ast_desc['type'].apply(lambda _t: type_to_id.get(_t, 0))
    ast_desc.sort_values('node_id', inplace=True)
    dgl_ast.ndata['token_id'] = ast_desc['token_id'].to_numpy()
    dgl_ast.ndata['type_id'] = ast_desc['type_id'].to_numpy()

    model = load_model(path_to_model, device)

    with open(labels_path, 'rb') as pkl_file:
        label_to_id = pkl_load(pkl_file)
    sublabel_to_id, label_to_sublabel = split_tokens_to_subtokens(label_to_id, device=device)
    id_to_sublabel = {v: k for k, v in sublabel_to_id.items()}

    print("forward pass...")
    logits = model(dgl_ast, torch.tensor([0], dtype=torch.long),
                   torch.full((50, 1), sublabel_to_id[SOS], dtype=torch.long), 0, device)
    predict = model.predict(logits).numpy()[:, 0]
    label = ''
    for i in predict:
        if i == sublabel_to_id[EOS]:
            break
        label += '|' + id_to_sublabel[i]
    label = label[1:]
    print(label)


if __name__ == '__main__':
    arg_parser = ArgumentParser(description=f"predict function name for given function by given model")
    arg_parser.add_argument("function", type=str, help="path to file with function")
    arg_parser.add_argument("model", type=str, help="path to freeze model")

    args = arg_parser.parse_args()
    interactive(args.function, args.model)
