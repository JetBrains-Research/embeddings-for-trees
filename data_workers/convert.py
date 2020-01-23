import os
from multiprocessing import Pool, cpu_count
from pickle import dump as pkl_dump
from pickle import load as pkl_load
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from dgl import DGLGraph, BatchedDGLGraph, batch, unbatch
from networkx.drawing.nx_pydot import read_dot
from tqdm.auto import tqdm

from utils.common import create_folder


def convert_dot_to_dgl(dot_path: str) -> DGLGraph:
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


def _convert_full_project(project_path: str, n_jobs: int = -1) -> Tuple[List[DGLGraph], List[str]]:
    ast_folder = os.path.join(project_path, 'asts')
    ast_files = sorted(
        os.listdir(ast_folder),
        key=lambda name: int(name[4:-4])
    )
    asts = [os.path.join(ast_folder, ast) for ast in ast_files]

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    with Pool(n_jobs) as pool:
        results = pool.imap(convert_dot_to_dgl, asts)
        graphs = [graph for graph in tqdm(results, total=len(asts))]
    return graphs, asts


def _get_project_description(project_path: str) -> pd.DataFrame:
    project_description = pd.read_csv(os.path.join(project_path, 'description.csv'))
    project_description['dot_file'] = project_description['dot_file'].apply(
        lambda dot_file: os.path.join(project_path, 'asts', dot_file)
    )
    return project_description


def _collect_all_descriptions(project_paths: List[str]) -> pd.DataFrame:
    asts_description = pd.concat(
        [_get_project_description(pr_path) for pr_path in tqdm(project_paths)],
        ignore_index=True
    )
    return asts_description


def transform_keys(df_chunk, token_to_id, type_to_id):
    keys = ['token', 'type']
    new_keys = ['token_id', 'type_id']
    funcs = [
        lambda token: token_to_id.get(token, 0),
        lambda cur_type: type_to_id.get(cur_type, 0)
    ]
    for key, new_key, func in zip(keys, new_keys, funcs):
        df_chunk[new_key] = df_chunk[key].apply(func)
    return df_chunk


def prepare_batch(current_description: pd.DataFrame, current_names: List[str],
                  batched_graph_callback: Callable[[], List[DGLGraph]]) -> Tuple:
    description_groups = current_description.groupby('dot_file')
    current_description = pd.concat(
        [description_groups.get_group(ast).sort_values('node_id') for ast in current_names],
        ignore_index=True
    )

    labels = description_groups.first().loc[current_names]['label'].to_list()
    paths = description_groups.first().loc[current_names]['source_file'].to_list()
    batched_graph = batch(batched_graph_callback())
    batched_graph.ndata['token_id'] = current_description['token_id'].to_numpy()
    batched_graph.ndata['type_id'] = current_description['type_id'].to_numpy()
    return batched_graph, labels, paths


def save_batch(batched_graph: BatchedDGLGraph, labels: List[str], paths: List[str],
               output_path: str, batch_num: int) -> None:
    with open(os.path.join(output_path, f'batch_{batch_num}.pkl'), 'wb') as pkl_file:
        pkl_dump({'batched_graph': batched_graph, 'labels': labels, 'paths': paths}, pkl_file)


def _prepare_project(project_path: str, token_to_id: Dict, type_to_id: Dict, n_jobs: int):
    print(f"converting {project_path} asts into dgl format")
    project_graphs, project_asts = _convert_full_project(project_path, n_jobs)
    print(f"prepare description for {project_path} project")
    project_description = _get_project_description(project_path)
    project_description['token'].fillna(value='NAN', inplace=True)
    project_description = transform_keys(project_description, token_to_id, type_to_id)

    print(f"add features to graphs in {project_path} project")
    batched_graph, labels, paths = prepare_batch(project_description, project_asts, lambda: project_graphs)
    save_batch(batched_graph, labels, paths, project_path, 0)


def _convert_per_project(project_paths: List[str], asts: List[str], ast_order: np.array, output_path: str,
                         token_to_id: Dict, type_to_id: Dict, n_jobs: int, batch_size: int) -> None:
    print("convert all projects")
    converted_func = 0
    for project_path in tqdm(project_paths):
        converted_func += len(os.listdir(os.path.join(project_path, 'asts')))
        if not os.path.exists(os.path.join(project_path, 'batch_0.pkl')):
            _prepare_project(project_path, token_to_id, type_to_id, n_jobs)
        print(f"converted {converted_func} functions")

    print("load all graphs to memory")
    graphs = []
    labels = []
    paths = []
    for project_path in tqdm(project_paths):
        with open(os.path.join(project_path, 'batch_0.pkl'), 'rb') as pkl_file:
            project_data = pkl_load(pkl_file)
            graphs += unbatch(project_data['batched_graph'])
            labels += project_data['labels']
            paths += project_data['paths']

    assert len(graphs) == len(ast_order)
    graphs = np.array(graphs)[ast_order]
    labels = np.array(labels)[ast_order]
    paths = np.array(paths)[ast_order]

    print(f"save batches")
    n_batches = len(asts) // batch_size + (1 if len(asts) % batch_size > 0 else 0)
    for batch_num in tqdm(range(n_batches)):
        current_slice = slice(batch_num * batch_size, min((batch_num + 1) * batch_size, len(asts)))
        save_batch(
            batch(graphs[current_slice]), labels[current_slice], paths[current_slice], output_path, batch_num
        )


def _convert_high_memory(project_paths: List[str], asts: List[str], asts_order: np.array, output_path: str,
                         token_to_id: Dict, type_to_id: Dict, n_jobs: int, batch_size: int) -> None:
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    print(f"Read all asts...")
    with Pool(n_jobs) as pool:
        results = pool.imap(convert_dot_to_dgl, asts)
        graphs = np.array([graph for graph in tqdm(results, total=len(asts))])
    asts = np.array(asts)[asts_order]
    graphs = graphs[asts_order]

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
            pool.starmap(transform_keys, chunks)
        )

    n_batches = len(asts) // batch_size + (1 if len(asts) % batch_size > 0 else 0)
    for batch_num in tqdm(range(n_batches)):
        current_slice = slice(batch_num * batch_size, min((batch_num + 1) * batch_size, len(asts)))
        current_names = asts[current_slice]

        desc_mask = full_descriptions['dot_file'].isin(current_names)
        current_description = full_descriptions[desc_mask]

        batched_graph, labels, paths = prepare_batch(current_description, current_names, lambda: graphs[current_slice])
        save_batch(batched_graph, labels, paths, output_path, batch_num)


def _convert_small_memory(projects_paths: List[str], asts: List[str], asts_order: np.array, output_path: str,
                          token_to_id: Dict, type_to_id: Dict, n_jobs: int, batch_size: int) -> None:
    n_batches = len(asts) // batch_size + (1 if len(asts) % batch_size > 0 else 0)
    pool = Pool(cpu_count() if n_jobs == -1 else n_jobs)
    asts = np.array(asts)[asts_order]

    for batch_num in tqdm(range(n_batches)):
        current_asts = asts[batch_num * batch_size: min((batch_num + 1) * batch_size, len(asts))]
        async_batch = pool.map_async(convert_dot_to_dgl, current_asts)

        current_description = _collect_ast_description(projects_paths, current_asts)
        current_description['token'].fillna(value='NAN', inplace=True)
        current_description['token_id'] = current_description['token'].apply(lambda token: token_to_id.get(token, 0))
        current_description['type_id'] = current_description['type'].apply(lambda cur_type: type_to_id.get(cur_type, 0))

        batched_graph, labels, paths = prepare_batch(current_description, current_asts, lambda: async_batch.get())
        save_batch(batched_graph, labels, paths, output_path, batch_num)

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
    random_index_permutation = np.random.permutation(len(asts))

    # _convert_func = _convert_small_memory
    # if is_high_memory:
    #     _convert_func = _convert_high_memory
    _convert_func = _convert_per_project if is_high_memory else _convert_small_memory
    _convert_func(projects_paths, asts, random_index_permutation, output_holdout_path,
                  token_to_id, type_to_id, n_jobs, batch_size)

    return output_holdout_path
