import os
from argparse import ArgumentParser
from pickle import load as pkl_load
from subprocess import run as subprocess_run

import dgl
import pandas as pd
import torch.nn as nn

from data_workers.convert import convert_dot_to_dgl, prepare_batch, transform_keys
from model.tree2seq import load_model
from utils.common import fix_seed, get_device, EOS, split_tokens_to_subtokens, PAD, create_folder, \
    convert_tokens_to_subtokens
from utils.learning_info import LearningInfo
from utils.training import eval_on_batch

TMP_FOLDER = '.tmp'
astminer_cli_path = 'utils/astminer-cli.jar'

vocab_path = 'data/java-small/vocabulary.pkl'
labels_path = 'data/java-small/labels.pkl'


def build_ast(path_to_function: str) -> bool:
    completed_process = subprocess_run(
        ['java', '-jar', astminer_cli_path, 'parse',
         '--project', path_to_function, '--output', TMP_FOLDER,
         '--storage', 'dot', '--granularity', 'method',
         '--lang', 'java', '--hide-method-name', '--split-tokens',
         '--java-parser', 'antlr']
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
    create_folder(TMP_FOLDER)
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
    ast_desc['token'].fillna('NAN', inplace=True)
    with open(vocab_path, 'rb') as pkl_file:
        vocab = pkl_load(pkl_file)
        token_to_id, type_to_id = vocab['token_to_id'], vocab['type_to_id']
    ast_desc = transform_keys(ast_desc, token_to_id, type_to_id)
    batched_graph, labels, paths = prepare_batch(ast_desc, ['ast_0.dot'], lambda: [dgl_ast])
    batched_graph = dgl.batch(
        list(map(lambda g: dgl.reverse(g, share_ndata=True), dgl.unbatch(batched_graph)))
    )
    model = load_model(path_to_model, device)

    with open(labels_path, 'rb') as pkl_file:
        label_to_id = pkl_load(pkl_file)
    sublabel_to_id, label_to_sublabel = split_tokens_to_subtokens(label_to_id, device=device)
    id_to_sublabel = {v: k for k, v in sublabel_to_id.items()}
    eval_label_to_sublabel = convert_tokens_to_subtokens(labels, sublabel_to_id, device)

    criterion = nn.CrossEntropyLoss(ignore_index=sublabel_to_id[PAD]).to(device)
    info = LearningInfo()

    print("forward pass...")
    batch_info, prediction = eval_on_batch(model, criterion, batched_graph, labels,
                                           eval_label_to_sublabel, sublabel_to_id, device)

    info.accumulate_info(batch_info)
    label = ''
    for cur_sublabel in prediction:
        if cur_sublabel.item() == sublabel_to_id[EOS]:
            break
        label += '|' + id_to_sublabel[cur_sublabel.item()]
    label = label[1:]
    print(label)
    print(info.get_state_dict())


if __name__ == '__main__':
    arg_parser = ArgumentParser(description=f"predict function name for given function by given model")
    arg_parser.add_argument("function", type=str, help="path to file with function")
    arg_parser.add_argument("model", type=str, help="path to freeze model")

    args = arg_parser.parse_args()
    interactive(args.function, args.model)
