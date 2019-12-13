import os
from multiprocessing import Pool, cpu_count
from pickle import dump as pkl_dump
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dgl import DGLGraph
from dgl import batch as dgl_batch
from networkx.drawing.nx_pydot import read_dot
from tqdm.auto import tqdm

from utils.common import create_folder


def _convert_dot_to_dgl(dot_path: str) -> DGLGraph:
    g_nx = read_dot(dot_path)
    g_dgl = DGLGraph(g_nx)
    return g_dgl


def _collect_ast_description(projects_paths: List[str], asts_batch: List[str]) -> pd.DataFrame:
    asts_per_project = {
        project_path: [ast for ast in asts_batch if ast.startswith(project_path)] for project_path in projects_paths
    }
    asts_description = pd.DataFrame()
    for project_path, asts in asts_per_project.items():
        if len(asts) == 0:
            continue
        project_description = pd.read_csv(os.path.join(project_path, 'description.csv'))
        ast_names = [os.path.basename(ast) for ast in asts]
        mask = project_description['dot_file'].isin(ast_names)
        project_description.loc[mask, 'dot_file'] = project_description.loc[mask, 'dot_file'].apply(
            lambda dot_file: os.path.join(project_path, 'asts', dot_file)
        )
        asts_description = pd.concat([asts_description, project_description[mask]], ignore_index=True)
    return asts_description


def _convert_full_project(project_path: str) -> Tuple[List[DGLGraph], List[str]]:
    asts_folder = os.path.join(project_path, 'asts')
    asts = [os.path.join(asts_folder, ast) for ast in os.listdir(asts_folder)][:200]
    graphs = [_convert_dot_to_dgl(ast) for ast in asts]
    return graphs, asts


def _collect_all_descriptions(project_paths: List[str]) -> pd.DataFrame:
    asts_description = pd.DataFrame()
    for project_path in project_paths:
        project_description = pd.read_csv(os.path.join(project_path, 'description.csv'))
        project_description['dot_file'] = project_description['dot_file'].apply(
            lambda dot_file: os.path.join(project_path, 'asts', dot_file)
        )
        asts_description = pd.concat([asts_description, project_description], ignore_index=True)
    return asts_description


def _async_transform_keys(df_chunk, token_to_id, type_to_id):
    keys = ['token', 'type']
    new_keys = ['token_id', 'type_id']
    funcs = [
        lambda token: token_to_id.get(token, 0),
        lambda cur_type: type_to_id.get(cur_type, 0)
    ]
    for key, new_key, func in zip(keys, new_keys, funcs):
        df_chunk[new_key] = df_chunk[key].apply(func)
    return df_chunk


def _convert_high_memory(project_paths: List[str], output_holdout_path, shuffled_asts: List[str],
                         token_to_id: Dict, type_to_id: Dict, n_jobs: int, batch_size: int) -> None:
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    print(f"Read all asts...")
    with Pool(n_jobs) as pool:
        results = pool.imap(_convert_dot_to_dgl, shuffled_asts)
        graphs = np.array([graph for graph in tqdm(results, total=len(shuffled_asts))])
    graphs_names = np.array(shuffled_asts)

    print(f"Prepare asts description...")
    full_descriptions = _collect_all_descriptions(project_paths)
    full_descriptions['token'].fillna(value='NAN', inplace=True)
    chunk_size = full_descriptions.shape[0] // n_jobs
    chunks = [
        (full_descriptions.iloc[i: min(i + chunk_size, full_descriptions.shape[0])], token_to_id, type_to_id)
        for i in range(0, full_descriptions.shape[0], chunk_size)
    ]
    with Pool(n_jobs) as pool:
        full_descriptions = pd.concat(
            pool.starmap(_async_transform_keys, chunks)
        )

    indexes = np.arange(len(graphs_names))
    np.random.shuffle(indexes)
    graphs = graphs[indexes]
    graphs_names = graphs_names[indexes]

    n_batches = len(graphs_names) // batch_size + (1 if len(graphs_names) % batch_size > 0 else 0)
    for batch_num in tqdm(range(n_batches)):
        current_slice = slice(batch_num * batch_size, min((batch_num + 1) * batch_size, len(graphs_names)))
        batched_graph = dgl_batch(graphs[current_slice])
        current_names = graphs_names[current_slice]

        desc_mask = full_descriptions['dot_file'].isin(current_names)
        current_description = full_descriptions[desc_mask]

        description_groups = current_description.groupby('dot_file')
        current_description = pd.concat(
            [description_groups.get_group(ast).sort_values('node_id') for ast in current_names],
            ignore_index=True
        )

        labels = description_groups.first().loc[current_names]['label'].to_list()
        paths = description_groups.first().loc[current_names]['source_file'].to_list()
        batched_graph.ndata['token_id'] = current_description['token_id'].to_numpy()
        batched_graph.ndata['type_id'] = current_description['type_id'].to_numpy()

        with open(os.path.join(output_holdout_path, f'batch_{batch_num}.pkl'), 'wb') as pkl_file:
            pkl_dump({'batched_graph': batched_graph, 'labels': labels, 'paths': paths}, pkl_file)


def _convert_small_memory(projects_paths: List[str], output_holdout_path: str, shuffled_asts: List[str],
                          token_to_id: Dict, type_to_id: Dict, n_jobs: int, batch_size: int) -> None:
    n_batches = len(shuffled_asts) // batch_size + (1 if len(shuffled_asts) % batch_size > 0 else 0)
    pool = Pool(cpu_count() if n_jobs == -1 else n_jobs)

    for batch_num in tqdm(range(n_batches)):
        current_asts = shuffled_asts[batch_num * batch_size: min((batch_num + 1) * batch_size, len(shuffled_asts))]
        async_batch = pool.map_async(_convert_dot_to_dgl, current_asts)

        current_description = _collect_ast_description(projects_paths, current_asts)
        current_description['token'].fillna(value='NAN', inplace=True)
        current_description['token_id'] = current_description['token'].apply(lambda token: token_to_id.get(token, 0))
        current_description['type_id'] = current_description['type'].apply(lambda cur_type: type_to_id.get(cur_type, 0))

        description_groups = current_description.groupby('dot_file')
        current_description = pd.concat(
            [description_groups.get_group(ast).sort_values('node_id') for ast in current_asts],
            ignore_index=True
        )

        labels = description_groups.first().loc[current_asts]['label'].to_list()
        paths = description_groups.first().loc[current_asts]['source_file'].to_list()
        batched_graph = dgl_batch(async_batch.get())
        batched_graph.ndata['token_id'] = current_description['token_id'].to_numpy()
        batched_graph.ndata['type_id'] = current_description['type_id'].to_numpy()

        with open(os.path.join(output_holdout_path, f'batch_{batch_num}.pkl'), 'wb') as pkl_file:
            pkl_dump({'batched_graph': batched_graph, 'labels': labels, 'paths': paths}, pkl_file)

    pool.close()


def convert_holdout(data_path: str, holdout_name: str, token_to_id: Dict,
                    type_to_id: Dict, n_jobs: int, batch_size: int, is_high_memory: bool) -> str:
    print(f"Convert asts for {holdout_name} data...")
    holdout_path = os.path.join(data_path, f'{holdout_name}_asts')
    output_holdout_path = os.path.join(data_path, f'{holdout_name}_preprocessed')
    create_folder(output_holdout_path)
    projects_paths = [os.path.join(holdout_path, project, 'java') for project in os.listdir(holdout_path)]
    asts = [os.path.join(project_path, 'asts', ast)
            for project_path in projects_paths
            for ast in os.listdir(os.path.join(project_path, 'asts'))
            ]
    np.random.shuffle(asts)

    _convert_func = _convert_small_memory
    if is_high_memory:
        _convert_func = _convert_high_memory
    _convert_func(projects_paths, output_holdout_path, asts[:200], token_to_id, type_to_id, n_jobs, batch_size)

    return output_holdout_path
