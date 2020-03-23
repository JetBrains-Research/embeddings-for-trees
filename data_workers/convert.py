import os
import re
from functools import partial
from multiprocessing import Pool, cpu_count
from pickle import dump as pkl_dump
from pickle import load as pkl_load
from typing import Dict

import numpy as np
import pandas as pd
from dgl import DGLGraph, batch, unbatch
from tqdm.auto import tqdm

from utils.common import create_folder, UNK, PAD


def convert_dot_to_dgl(dot_path: str) -> DGLGraph:
    with open(dot_path, 'r') as dot_file:
        lines = dot_file.readlines()
    edges = []
    for line in lines[1:-1]:
        edges.append([int(i) for i in re.findall(r'\d+', line)])
    g_dgl = DGLGraph()
    g_dgl.add_nodes(max(edges)[0] + 1)
    for e in edges:
        if len(e) > 1:
            g_dgl.add_edges(e[0], e[1:])
    return g_dgl


def _move_tokens_to_leaves(graph: DGLGraph, pad_token_index: int, pad_type_index: int) -> DGLGraph:
    old_token = graph.ndata['token'].numpy()
    n_old_nodes = old_token.shape[0]

    type_mask = graph.ndata['type'].numpy() != pad_type_index
    type_mask = np.tile(type_mask.reshape(-1, 1), old_token.shape[1])
    mask = np.logical_and(old_token != pad_token_index, type_mask)
    n_new_nodes = mask.sum()

    new_token = np.full((n_old_nodes + n_new_nodes, 1), pad_token_index, dtype=np.int)
    new_token[:n_old_nodes] = np.where(~type_mask, old_token, pad_token_index)[:, [0]]
    new_token[n_old_nodes:] = old_token[mask].reshape(-1, 1)

    us, _ = np.nonzero(mask)
    vs = np.arange(n_new_nodes) + n_old_nodes

    graph.add_nodes(n_new_nodes)
    graph.add_edges(us, vs)

    graph.ndata['type'][n_old_nodes:] = pad_type_index
    graph.ndata['token'] = new_token
    return graph


def _split_token(data: pd.Series, to_id: Dict, max_len: int, delimiter: str = '|') -> pd.Series:
    unk_id = to_id[UNK]
    pad_id = to_id[PAD]
    data = data.apply(
        lambda _token:
        [to_id.get(t, unk_id) for t in _token.split(delimiter)[:max_len]]
        if isinstance(_token, str) else []
    ).apply(
        lambda _l: _l + [pad_id] * (max_len - len(_l))
    )
    return data


def prepare_project_description(project_path: str, token_to_id: Dict, type_to_id: Dict, label_to_id: Dict,
                                is_split: bool, max_token_len: int = -1, max_label_len: int = -1, delimiter: str = '|'
                                ) -> pd.DataFrame:
    description = pd.read_csv(os.path.join(project_path, 'description.csv')).sort_values(['dot_file', 'node_id'])

    unk_type_id = type_to_id[UNK]
    description['type_feature'] = description['type'].apply(lambda _type: type_to_id.get(_type, unk_type_id))
    # convert token and label
    if is_split:
        description['token_feature'] = _split_token(description['token'], token_to_id, max_token_len, delimiter)
        description['label_feature'] = _split_token(description['label'], label_to_id, max_label_len, delimiter)
    else:
        unk_token_id = token_to_id[UNK]
        unk_label_id = label_to_id[UNK]
        description['token_feature'] = description['token'].apply(lambda _token: token_to_id.get(_token, unk_token_id))
        description['label_feature'] = description['label'].apply(lambda _label: label_to_id.get(_label, unk_label_id))

    return description


def convert_project(project_path: str, token_to_id: Dict, type_to_id: Dict, label_to_id: Dict,
                    token_to_leaves: bool, is_split: bool, max_token_len: int = -1, max_label_len: int = -1,
                    delimiter: str = '|'):
    # print("node description preparation")
    description = prepare_project_description(
        project_path, token_to_id, type_to_id, label_to_id, is_split, max_token_len, max_label_len, delimiter
    )

    asts = os.listdir(os.path.join(project_path, 'asts'))
    asts.sort(key=lambda _ast: int(re.findall(r'\d+', _ast)[0]))
    # print("converting to dgl format...")
    graphs = batch([convert_dot_to_dgl(os.path.join(project_path, 'asts', ast)) for ast in asts])

    description_ordered = description.set_index('dot_file').loc[asts]
    graphs.ndata['token'] = np.vstack(description_ordered['token_feature'])
    graphs.ndata['type'] = description_ordered['type_feature'].to_numpy()

    if token_to_leaves:
        # print("move tokens to leaves...")
        graphs = batch([_move_tokens_to_leaves(g, token_to_id[PAD], type_to_id[PAD]) for g in unbatch(graphs)])

    mask_per_ast = description_ordered['node_id'] == 0
    labels = description_ordered[mask_per_ast]['label_feature'].values
    source_paths = description_ordered[mask_per_ast]['source_file'].values

    with open(os.path.join(project_path, 'converted.pkl'), 'wb') as out_file:
        pkl_dump({'graphs': graphs, 'labels': labels, 'paths': source_paths}, out_file)


def _convert_project_safe(project_path, **kwargs):
    try:
        convert_project(project_path, **kwargs)
    except Exception as err:
        with open(os.path.join(os.getcwd(), 'convert.txt'), 'a') as log_file:
            log_file.write(f"can't convert {project_path} project, failed with\n{err}")


def convert_holdout(data_path: str, holdout_name: str, batch_size: int,
                    token_to_id: Dict, type_to_id: Dict, label_to_id: Dict,
                    tokens_to_leaves: bool = False, is_split: bool = False,
                    max_token_len: int = -1, max_label_len: int = -1,
                    delimiter: str = '|', shuffle: bool = True, n_jobs: int = -1) -> str:
    print(f"Convert asts for {holdout_name} data...")
    holdout_path = os.path.join(data_path, f'{holdout_name}_asts')
    output_holdout_path = os.path.join(data_path, f'{holdout_name}_preprocessed')
    create_folder(output_holdout_path)

    projects_paths = [os.path.join(holdout_path, project, 'java') for project in os.listdir(holdout_path)]

    print("converting projects...")
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    with Pool(n_jobs) as pool:
        pool_func = partial(
            _convert_project_safe, token_to_id=token_to_id, type_to_id=type_to_id, label_to_id=label_to_id,
            token_to_leaves=tokens_to_leaves, is_split=is_split, max_token_len=max_token_len,
            max_label_len=max_label_len, delimiter=delimiter
        )
        results = pool.imap(pool_func, projects_paths)
        for _ in tqdm(results, total=len(projects_paths)):
            pass

    graphs = []
    labels = []
    paths = []
    print("load graph to memory...")
    for project_path in tqdm(projects_paths):
        if not os.path.exists(os.path.join(project_path, 'converted.pkl')):
            with open(os.path.join(os.getcwd(), 'convert.txt'), 'a') as log_file:
                log_file.write(f"can't load graphs for {project_path} project\n")
            continue
        with open(os.path.join(project_path, 'converted.pkl'), 'rb') as pkl_file:
            project_data = pkl_load(pkl_file)
            graphs += unbatch(project_data['graphs'])
            labels.append(project_data['labels'])
            paths.append(project_data['paths'])
    graphs = np.array(graphs)
    labels = np.concatenate(labels)
    paths = np.concatenate(paths)

    assert len(graphs) == len(labels) == len(paths), "unequal lengths of graphs, labels and paths"
    print(f"total number of graphs: {len(graphs)}")

    if shuffle:
        order = np.random.permutation(len(graphs))
        graphs = graphs[order]
        labels = labels[order]
        paths = paths[order]

    print(f"save batches...")
    n_batches = len(graphs) // batch_size + (1 if len(graphs) % batch_size > 0 else 0)
    for batch_num in tqdm(range(n_batches)):
        current_slice = slice(batch_num * batch_size, min((batch_num + 1) * batch_size, len(graphs)))
        with open(os.path.join(output_holdout_path, f'batch_{batch_num}.pkl'), 'wb') as pkl_file:
            pkl_dump(
                {'batched_graph': batch(graphs[current_slice]), 'labels': labels[current_slice],
                 'paths': paths[current_slice]},
                pkl_file
            )

    return output_holdout_path
