import os
import pickle
from argparse import ArgumentParser
from os.path import join

import dgl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def collect_type_statistic(data_folder: str, vocabulary: str) -> pd.DataFrame:
    with open(vocabulary, 'rb') as pkl_file:
        type_to_id = pickle.load(pkl_file)['type_to_id']
    id_to_type = {v: k for k, v in type_to_id.items()}
    stats = pd.DataFrame(
        columns=['count'], dtype=np.int
    )
    batches = os.listdir(data_folder)
    for batch in tqdm(batches):
        with open(join(data_folder, batch), 'rb') as pkl_file:
            graphs = dgl.unbatch(
                pickle.load(pkl_file)['batched_graph']
            )
        for graph in tqdm(graphs):
            for n_i in range(graph.number_of_nodes()):
                cur_type_id = graph.ndata['type_id'][n_i].item()
                children_mask_int = [0 for _ in range(len(type_to_id))]
                for child_type_id in graph.ndata['type_id'][graph.out_edges(n_i)[1]]:
                    children_mask_int[child_type_id.item()] += 1
                children_mask = ''.join(map(str, children_mask_int))
                pandas_index = pd.MultiIndex.from_tuples([(cur_type_id, children_mask)])
                if (cur_type_id, children_mask) not in stats.index:
                    new_row = pd.DataFrame({'count': [0]}, index=pandas_index)
                    stats = stats.append(new_row)
                stats.loc[pandas_index, 'count'] += 1
    stats = stats.reset_index()
    stats['node_type_id'] = stats['index'].apply(lambda t: t[0])
    stats['node_type'] = stats['node_type_id'].apply(lambda t: id_to_type[t])
    for cur_type, cur_type_id in type_to_id.items():
        stats[cur_type] = stats['index'].apply(
            lambda t: int(t[1][cur_type_id])
        )
    stats.drop('index', inplace=True, axis=1)
    return stats


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('data_folder', type=str, help="path to the preprocessed data a.k.a batched graphs")
    arg_parse.add_argument('vocabulary', type=str, help="path to vocabulary pickle")
    arg_parse.add_argument('output', type=str)
    args = arg_parse.parse_args()

    collect_type_statistic(args.data_folder, args.vocabulary).to_csv(args.output, index=False)
