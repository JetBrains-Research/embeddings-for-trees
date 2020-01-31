import os
import pickle
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from typing import Tuple

import dgl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _collect_node_type_statistic_per_batch(arg: Tuple[str, int]) -> pd.DataFrame:
    batch_path, number_of_types = arg
    with open(batch_path, 'rb') as pkl_file:
        graphs = dgl.unbatch(
            pickle.load(pkl_file)['batched_graph']
        )
    stats = pd.DataFrame(
        columns=['count'], dtype=np.int
    )
    for graph in graphs:
        for n_i in range(graph.number_of_nodes()):
            cur_type_id = graph.ndata['type_id'][n_i].item()
            children_mask_int = [0 for _ in range(number_of_types)]
            for child_type_id in graph.ndata['type_id'][graph.out_edges(n_i)[1]]:
                children_mask_int[child_type_id.item()] += 1
            children_mask = ''.join(map(str, children_mask_int))
            pandas_index = pd.MultiIndex.from_tuples([(cur_type_id, children_mask)])
            if (cur_type_id, children_mask) not in stats.index:
                new_row = pd.DataFrame({'count': [0]}, index=pandas_index)
                stats = stats.append(new_row)
            stats.loc[pandas_index, 'count'] += 1
    return stats


def collect_node_type_statistic(data_folder: str, holdout_name: str, n_jobs: int) -> pd.DataFrame:
    vocabulary = os.path.join(data_folder, 'vocabulary.pkl')
    batch_folder = os.path.join(data_folder, f'{holdout_name}_preprocessed')
    with open(vocabulary, 'rb') as pkl_file:
        type_to_id = pickle.load(pkl_file)['type_to_id']
    id_to_type = {v: k for k, v in type_to_id.items()}

    batches = os.listdir(batch_folder)
    if n_jobs == -1:
        n_jobs = cpu_count()
    with Pool(n_jobs) as pool:
        results = pool.imap(
            _collect_node_type_statistic_per_batch,
            map(lambda _p: (os.path.join(batch_folder, _p), len(type_to_id)), batches)
        )
        stats_per_batch = list(tqdm(results, total=len(batches)))
    stats = pd.concat(
        [_s.reset_index() for _s in stats_per_batch]
    )
    stats = stats.groupby('index').agg('sum')
    stats = stats.reset_index()
    stats['node_type_id'] = stats['index'].apply(lambda t: t[0])
    stats['node_type'] = stats['node_type_id'].apply(lambda t: id_to_type[t])
    for cur_type, cur_type_id in type_to_id.items():
        stats[cur_type] = stats['index'].apply(
            lambda t: int(t[1][cur_type_id])
        )
    stats.drop('index', inplace=True, axis=1)
    return stats


def _collect_tree_statistic_per_batch(batch_path: str) -> pd.DataFrame:
    with open(batch_path, 'rb') as pkl_file:
        batch = pickle.load(pkl_file)
    stats_data = {
        'number_of_nodes': map(lambda g: g.number_of_nodes(), dgl.unbatch(batch['batched_graph'])),
        'label': batch['labels'],
        'source_path': batch['paths']
    }
    return pd.DataFrame(stats_data)


def collect_tree_statistic(data_folder: str, holdout_name: str, n_jobs: int) -> pd.DataFrame:
    batch_folder = os.path.join(data_folder, f'{holdout_name}_preprocessed')
    batches = os.listdir(batch_folder)[:5]
    if n_jobs == -1:
        n_jobs = cpu_count()
    with Pool(n_jobs) as pool:
        results = pool.imap(
            _collect_tree_statistic_per_batch, map(lambda p: os.path.join(batch_folder, p), batches)
        )
        stats_per_batch = list(tqdm(results, total=len(batches)))
    stats = pd.concat(stats_per_batch, ignore_index=True)
    return stats


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--data', type=str, required=True)
    arg_parse.add_argument('--output', type=str, required=True)
    arg_parse.add_argument('--holdout', type=str, required=True)
    arg_parse.add_argument('--n_jobs', type=int, default=-1)
    subparsers = arg_parse.add_subparsers(help='type of statistic')

    node_type_parser = subparsers.add_parser('node_type')
    node_type_parser.set_defaults(statistic=collect_node_type_statistic)

    tree_parser = subparsers.add_parser('tree')
    tree_parser.set_defaults(statistic=collect_tree_statistic)

    args = arg_parse.parse_args()
    args.statistic(args.data, args.holdout, args.n_jobs).to_csv(os.path.join(args.data, args.output), index=False)
