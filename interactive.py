import os
from argparse import ArgumentParser
from itertools import takewhile
from pickle import load as pkl_load

import dgl
import numpy
import torch

from data_workers.convert import convert_project
from data_workers.preprocess_steps import build_project_asts
from model.tree2seq import load_model
from utils.common import fix_seed, get_device, create_folder, EOS

tmp_folder = '.tmp'
astminer_cli_path = 'utils/astminer-cli.jar'


def interactive(path_to_function: str, path_to_model: str):
    fix_seed()
    device = get_device()
    print(f"using {device} device")

    # load model
    print("loading model...")
    model, checkpoint = load_model(path_to_model, device)
    token_to_id = checkpoint['configuration']['token_to_id']
    type_to_id = checkpoint['configuration']['type_to_id']
    label_to_id = checkpoint['configuration']['label_to_id']
    id_to_label = {v: k for k, v in label_to_id.items()}

    # convert function to dgl format
    print("convert function to dgl format...")
    create_folder(tmp_folder)
    build_project_asts(path_to_function, tmp_folder, astminer_cli_path)
    project_folder = os.path.join(tmp_folder, 'java')
    convert_project(project_folder, token_to_id, type_to_id, label_to_id, True, True, 5, 6, False, True, '|')

    # load function
    with open(os.path.join(project_folder, 'converted.pkl'), 'rb') as pkl_file:
        data = pkl_load(pkl_file)
    assert len(data['labels']) == 1, f"found {len(data['labels'])} functions, instead of 1"
    ast = dgl.unbatch(data['graphs'])[0]
    ast = dgl.reverse(ast, share_ndata=True)
    ast.ndata['token'] = ast.ndata['token'].to(device)
    ast.ndata['type'] = ast.ndata['type'].to(device)
    labels = torch.tensor(numpy.stack(data['labels']).T, device=device)
    root_indexes = torch.tensor([0], dtype=torch.long)

    # forward pass
    model.eval()
    with torch.no_grad():
        logits = model(ast, root_indexes, labels, device)
    logits = logits[1:]
    prediction = model.predict(logits).reshape(-1)
    sublabels = [id_to_label[label_id.item()] for label_id in prediction]
    label = '|'.join(takewhile(lambda sl: sl != EOS, sublabels))
    print(f"the predicted label is:\n{label}")


if __name__ == '__main__':
    arg_parser = ArgumentParser(description=f"predict function name for given function by given model")
    arg_parser.add_argument("function", type=str, help="path to the file with function")
    arg_parser.add_argument("model", type=str, help="path to the frozen model")

    args = arg_parser.parse_args()
    interactive(args.function, args.model)
