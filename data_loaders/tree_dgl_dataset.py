import os
from pickle import load
from typing import Tuple, List

import torch
from dgl import DGLGraph, batch
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.tree_operations import get_tree_depth


def _get_batches(path: str, ext: str) -> List[str]:
    files = filter(lambda _f: _f.endswith(f'.{ext}'), os.listdir(path))
    # assuming filenames are "batch_xxx.{ext}"
    files = sorted(files, key=lambda name: int(name[6:-len(ext) - 1]))
    return [os.path.join(path, gf) for gf in files]


class TreeDGLDataset(Dataset):

    def __init__(
            self, dataset_path: str, batch_size: int, device: torch.device,
            invert_edges: bool = False, max_n_nodes: int = -1, max_depth: int = -1
    ) -> None:
        if not os.path.exists(dataset_path):
            raise ValueError(f"no dataset found on {dataset_path}")

        self.device = device
        self.invert_edges = invert_edges
        self.max_n_nodes = max_n_nodes
        self.max_depth = max_depth

        label_files = _get_batches(dataset_path, 'pkl')
        graph_files = _get_batches(dataset_path, 'dgl')
        self.batch_description = []

        # iterate over pkl files to aggregate information about batches
        print(f"prepare the {dataset_path} dataset...")
        for graph_file, label_file in tqdm(zip(graph_files, label_files), total=len(graph_files)):
            with open(label_file, 'rb') as pkl_file:
                pkl_data = load(pkl_file)
                labels = pkl_data['labels']
            n_graphs = len(labels)

            batches_per_file = n_graphs // batch_size + (1 if n_graphs % batch_size > 0 else 0)

            # collect information from the file
            for batch_id in range(batches_per_file):
                start_index = batch_id * batch_size
                end_index = min(n_graphs, (batch_id + 1) * batch_size)
                self.batch_description.append((graph_file, label_file, start_index, end_index))

        self.current_label_file = None
        self.labels = None

    def __len__(self) -> int:
        return len(self.batch_description)

    def _is_tree_suitable(self, tree: DGLGraph) -> bool:
        return (self.max_n_nodes == -1 or tree.number_of_nodes() < self.max_n_nodes) and\
               (self.max_depth == -1 or get_tree_depth(tree) < self.max_depth)

    def __getitem__(self, item) -> Tuple[DGLGraph, torch.Tensor]:
        graph_filename, label_filename, start_index, end_index = self.batch_description[item]

        if label_filename != self.current_label_file:
            self.current_label_file = label_filename
            with open(label_filename, 'rb') as pkl_file:
                pkl_data = load(pkl_file)
            self.labels = torch.tensor(pkl_data['labels'].T, device=self.device).detach()

        graphs, _ = load_graphs(graph_filename, list(range(start_index, end_index)))
        graphs, mask = zip(*[(g, i) for i, g in enumerate(graphs) if self._is_tree_suitable(g)])

        if self.invert_edges:
            graphs = [g.reverse(share_ndata=True) for g in graphs]

        graph = batch(graphs)
        graph.ndata['token'] = graph.ndata['token'].to(self.device).detach()
        graph.ndata['type'] = graph.ndata['type'].to(self.device).detach()
        # [sequence len, batch size]
        labels = self.labels[:, start_index:end_index][:, list(mask)]

        return graph, labels
