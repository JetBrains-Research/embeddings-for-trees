import os
from typing import Tuple

import torch
from dgl import DGLGraph, batch
from dgl.data.utils import load_labels, load_graphs
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.common import get_tree_depth


class JavaDataset(Dataset):

    def __init__(
            self, dataset_path: str, batch_size: int, device: torch.device, invert_edges: bool = False
    ) -> None:
        assert os.path.exists(dataset_path)

        self.device = device
        self.invert_edges = invert_edges

        graph_files = filter(lambda _f: _f.endswith('.dgl'), os.listdir(dataset_path))
        graph_files = sorted(graph_files, key=lambda name: int(name[6:-4]))
        graph_files = [os.path.join(dataset_path, gf) for gf in graph_files]

        self.batch_description = []

        # iterate over pkl files to aggregate information about batches
        print(f"prepare the {dataset_path} dataset...")
        for graph_file in tqdm(graph_files):
            labels = load_labels(graph_file)
            n_graphs = len(labels['labels'])

            batches_per_file = n_graphs // batch_size + (1 if n_graphs % batch_size > 0 else 0)

            # collect information from the file
            for batch_id in range(batches_per_file):
                start_index = batch_id * batch_size
                end_index = min(n_graphs, (batch_id + 1) * batch_size)
                self.batch_description.append((graph_file, start_index, end_index))

    def __len__(self) -> int:
        return len(self.batch_description)

    def __getitem__(self, item) -> Tuple[DGLGraph, torch.Tensor]:
        graph_filename, start_index, end_index = self.batch_description[item]

        graphs, label_dict = load_graphs(graph_filename, list(range(start_index, end_index)))
        graphs, mask = zip(*[
            (g, i) for i, g in enumerate(graphs)
            if g.number_of_nodes() < 10000 and get_tree_depth(g) < 100
        ])

        if self.invert_edges:
            graphs = [g.reverse(share_ndata=True) for g in graphs]

        graph = batch(graphs)
        graph.ndata['token'] = graph.ndata['token'].to(self.device).detach()
        graph.ndata['type'] = graph.ndata['type'].to(self.device).detach()
        # [sequence len, batch size]
        labels = label_dict['labels'][start_index:end_index, :][list(mask)]
        labels = labels.t().to(self.device).detach()

        return graph, labels
