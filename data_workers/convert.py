import os
import re
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pickle import dump, load
from typing import Dict

import numpy as np
import pandas as pd
from dgl import DGLGraph, batch, unbatch
from dgl.data.utils import save_graphs, load_graphs
from tqdm.auto import tqdm

from utils.common import create_folder, UNK, PAD, SOS, EOS


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


def _split_token(data: pd.Series, to_id: Dict, max_len: int, is_wrap: bool, delimiter: str = '|') -> pd.Series:
    unk_id = to_id[UNK]
    pad_id = to_id[PAD]

    def split_val(val):
        if is_wrap:
            return [SOS] + val.split(delimiter) + [EOS]
        else:
            return val.split(delimiter)

    data = data.apply(
        lambda _token:
        [to_id.get(t, unk_id) for t in split_val(_token)[:max_len]]
        if isinstance(_token, str) else []
    ).apply(
        lambda _l: _l + [pad_id] * (max_len - len(_l))
    )
    return data


def prepare_project_description(project_path: str, token_to_id: Dict, type_to_id: Dict, label_to_id: Dict,
                                is_split: bool, max_token_len: int = -1, max_label_len: int = -1,
                                wrap_tokens: bool = False, wrap_labels: bool = False, delimiter: str = '|'
                                ) -> pd.DataFrame:
    description = pd.read_csv(os.path.join(project_path, 'description.csv')).sort_values(['dot_file', 'node_id'])

    unk_type_id = type_to_id[UNK]
    description['type_feature'] = description['type'].apply(lambda _type: type_to_id.get(_type, unk_type_id))
    # convert token and label
    if is_split:
        description['token_feature'] = _split_token(description['token'], token_to_id, max_token_len, wrap_tokens,
                                                    delimiter)
        description['label_feature'] = _split_token(description['label'], label_to_id, max_label_len, wrap_labels,
                                                    delimiter)
    else:
        unk_token_id = token_to_id[UNK]
        unk_label_id = label_to_id[UNK]
        description['token_feature'] = description['token'].apply(lambda _token: token_to_id.get(_token, unk_token_id))
        description['label_feature'] = description['label'].apply(lambda _label: label_to_id.get(_label, unk_label_id))

    return description


def convert_project(project_path: str, token_to_id: Dict, type_to_id: Dict, label_to_id: Dict,
                    token_to_leaves: bool, is_split: bool, max_token_len: int = -1, max_label_len: int = -1,
                    wrap_tokens: bool = False, wrap_labels: bool = False, delimiter: str = '|', n_jobs: int = -1):
    print("node description preparation")
    description = prepare_project_description(
        project_path, token_to_id, type_to_id, label_to_id, is_split,
        max_token_len, max_label_len, wrap_tokens, wrap_labels, delimiter
    )

    asts = os.listdir(os.path.join(project_path, 'asts'))
    asts.sort(key=lambda _ast: int(re.findall(r'\d+', _ast)[0]))

    print("converting to dgl format...")
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    with Pool(n_jobs) as pool:
        results = pool.imap(convert_dot_to_dgl, [os.path.join(project_path, 'asts', ast) for ast in asts])
        graphs = [g for g in tqdm(results, total=len(asts))]

    description_ordered = description.set_index('dot_file').loc[asts]
    graphs = batch(graphs)
    graphs.ndata['token'] = np.vstack(description_ordered['token_feature'])
    graphs.ndata['type'] = description_ordered['type_feature'].to_numpy()
    graphs = unbatch(graphs)

    if token_to_leaves:
        print("move tokens to leaves...")
        graphs = [_move_tokens_to_leaves(g, token_to_id[PAD], type_to_id[PAD]) for g in tqdm(graphs)]

    mask_per_ast = description_ordered['node_id'] == 0
    labels = np.stack(description_ordered[mask_per_ast]['label_feature'].values)
    source_paths = np.stack(description_ordered[mask_per_ast]['source_file'].values)

    save_graphs(os.path.join(project_path, 'converted.dgl'), graphs)
    with open(os.path.join(project_path, 'converted.pkl'), 'wb') as pkl_file:
        dump({
            'labels': labels, 'source_paths': source_paths
        }, pkl_file)


def _convert_project_safe(project_path: str, log_file: str,  **kwargs):
    if os.path.exists(os.path.join(project_path, 'converted.dgl')):
        return
    try:
        convert_project(project_path, **kwargs)
    except Exception as err:
        with open(log_file, 'a') as log_file:
            log_file.write(f"can't convert {project_path} project, failed with\n{err}")


def convert_holdout(data_path: str, holdout_name: str, batch_size: int,
                    token_to_id: Dict, type_to_id: Dict, label_to_id: Dict,
                    tokens_to_leaves: bool = False, is_split: bool = False,
                    max_token_len: int = -1, max_label_len: int = -1, wrap_tokens: bool = False,
                    wrap_labels: bool = False, delimiter: str = '|', shuffle: bool = True, n_jobs: int = -1) -> str:
    log_file = os.path.join('logs', f"convert_{datetime.now().strftime('%Y_%m_%d_%H:%M:%S')}.txt")

    print(f"Convert asts for {holdout_name} data...")
    holdout_path = os.path.join(data_path, f'{holdout_name}_asts')
    output_holdout_path = os.path.join(data_path, f'{holdout_name}_preprocessed')
    create_folder(output_holdout_path)

    projects_paths = [os.path.join(holdout_path, project, 'java') for project in os.listdir(holdout_path)]

    print("converting projects...")
    for project_path in tqdm(projects_paths):
        print(f"converting {project_path}")
        _convert_project_safe(
            project_path, log_file, token_to_id=token_to_id, type_to_id=type_to_id, label_to_id=label_to_id,
            token_to_leaves=tokens_to_leaves, is_split=is_split, max_token_len=max_token_len,
            max_label_len=max_label_len, wrap_tokens=wrap_tokens, wrap_labels=wrap_labels, delimiter=delimiter,
            n_jobs=n_jobs
        )

    graphs = []
    labels = []
    source_paths = []
    print("load graphs to memory...")
    for project_path in tqdm(projects_paths):
        graph_path = os.path.join(project_path, 'converted.dgl')
        labels_path = os.path.join(project_path, 'converted.pkl')
        if not os.path.exists(graph_path) or not os.path.exists(labels_path):
            with open(log_file, 'a') as file:
                file.write(f"can't load graphs for {project_path} project\n")
            continue
        cur_graphs, _ = load_graphs(graph_path)
        with open(labels_path, 'rb') as pkl_file:
            pkl_data = load(pkl_file)
            cur_labels, cur_source_paths = pkl_data['labels'], pkl_data['source_paths']
        graphs += cur_graphs
        labels.append(cur_labels)
        source_paths.append(cur_source_paths)
    graphs = np.array(graphs)
    labels = np.vstack(labels)
    source_paths = np.concatenate(source_paths)

    assert len(graphs) == len(labels), "unequal lengths of graphs and labels"
    assert len(graphs) == len(source_paths), "unequal lengths of graphs and source paths"
    print(f"total number of graphs: {len(graphs)}")

    if shuffle:
        order = np.random.permutation(len(graphs))
        graphs = graphs[order]
        labels = labels[order]
        source_paths = source_paths[order]

    print(f"save batches...")
    n_batches = len(graphs) // batch_size + (1 if len(graphs) % batch_size > 0 else 0)
    for batch_num in tqdm(range(n_batches)):
        current_slice = slice(batch_num * batch_size, min((batch_num + 1) * batch_size, len(graphs)))
        output_graph_path = os.path.join(output_holdout_path, f'batch_{batch_num}.dgl')
        output_labels_path = os.path.join(output_holdout_path, f'batch_{batch_num}.pkl')
        save_graphs(output_graph_path, graphs[current_slice])
        with open(output_labels_path, 'wb') as pkl_file:
            dump({
                'labels': labels[current_slice], 'source_paths': source_paths[current_slice]
            }, pkl_file)

    return output_holdout_path
