import json
import os
import pickle
from argparse import ArgumentParser
from os.path import join
from typing import Dict, List

import dgl
import numpy as np
from tqdm.auto import tqdm


def collect_type_statistic(data_folder: str, vocabulary: str) -> List:
    with open(vocabulary, 'rb') as pkl_file:
        type_to_id = pickle.load(pkl_file)['type_to_id']
    id_to_type = {v: k for k, v in type_to_id.items()}
    batches = os.listdir(data_folder)
    stats_per_type = {}
    for batch in tqdm(batches):
        with open(join(data_folder, batch), 'rb') as pkl_file:
            graphs = dgl.unbatch(
                pickle.load(pkl_file)['batched_graph']
            )
        for graph in tqdm(graphs):
            for n_i in range(graph.number_of_nodes()):
                cur_type_id = graph.ndata['type_id'][n_i].item()
                if cur_type_id not in stats_per_type:
                    stats_per_type[cur_type_id] = []
                children_types = graph.ndata['type_id'][graph.out_edges(n_i)[1]].tolist()
                children_types.sort()
                for prev_children, num in stats_per_type[cur_type_id]:
                    if len(children_types) == len(prev_children) and np.allclose(children_types, prev_children):
                        num += 1
                        continue
                stats_per_type[cur_type_id].append((tuple(children_types), 1))
    stats = []
    for type_id in stats_per_type:
        stats += [
            (count, id_to_type[type_id], [id_to_type[child_type] for child_type in children_types])
            for (children_types, count) in stats_per_type[type_id]
        ]
    stats = list(sorted(stats, key=lambda t: t[0], reverse=True))
    return stats


def save_stats(stats: List, output_path: str) -> None:
    with open(output_path, 'w') as output_file:
        output_file.write(f"count,type,children_types\n")
        for count, type_name, children_type_names in stats:
            output_file.write(
                f"{count},{type_name},{' '.join(children_type_names)}\n"
            )


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('data_folder', type=str, help="path to the preprocessed data a.k.a batched graphs")
    arg_parse.add_argument('vocabulary', type=str, help="path to vocabulary pickle")
    arg_parse.add_argument('output', type=str)
    args = arg_parse.parse_args()

    save_stats(collect_type_statistic(args.data_folder, args.vocabulary), args.output)
