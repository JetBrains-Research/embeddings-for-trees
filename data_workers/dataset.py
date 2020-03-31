import os
from typing import Tuple

import torch
from dgl import DGLGraph, batch
from dgl.data.utils import load_labels, load_graphs
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class JavaDataset(Dataset):

    def __init__(
            self, batched_graphs_path: str, batch_size: int, device: torch.device, invert_edges: bool = False
    ) -> None:
        self.device = device
        self.batched_graphs_path = batched_graphs_path
        self.batch_size = batch_size
        self.invert_edges = invert_edges
        assert os.path.exists(self.batched_graphs_path)

        self.batched_graph_files = sorted(list(filter(
            lambda filename: filename.endswith('.dgl'), os.listdir(self.batched_graphs_path)
        )), key=lambda name: int(name[6:-4]))

        self.batch_desc = {}
        self.n_batches = 0

        self.loaded_batch_basename = None
        self.loaded_graphs = None
        self.loaded_labels = None

        # iterate over pkl files to aggregate information about batches
        print(f"prepare the {batched_graphs_path} dataset...")
        for batched_graph_file in tqdm(self.batched_graph_files):
            labels = load_labels(os.path.join(batched_graphs_path, batched_graph_file))
            n_graphs = len(labels['labels'])
            batches_per_file = n_graphs // self.batch_size + (1 if n_graphs % self.batch_size > 0 else 0)

            # collect information from the file
            for batch_id in range(batches_per_file):
                batch_slice = slice(
                    batch_id * self.batch_size,
                    min((batch_id + 1) * self.batch_size, n_graphs)
                )
                self.batch_desc[self.n_batches + batch_id] = (
                    batched_graph_file, batch_slice
                )

            self.n_batches += batches_per_file

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, item) -> Tuple[DGLGraph, torch.Tensor]:
        batch_basename, batch_slice = self.batch_desc[item]

        # read file only if previous wasn't the same
        if self.loaded_batch_basename != batch_basename:
            self.loaded_graphs, labels = load_graphs(
                os.path.join(self.batched_graphs_path, batch_basename)
            )
            self.loaded_labels = labels['labels']
            self.loaded_batch_basename = batch_basename

        graphs_for_batch = self.loaded_graphs[batch_slice]
        if self.invert_edges:
            graphs_for_batch = list(map(lambda g: g.reverse(share_ndata=True), graphs_for_batch))

        graph = batch(graphs_for_batch)
        graph.ndata['token'] = graph.ndata['token'].to(self.device)
        graph.ndata['type'] = graph.ndata['type'].to(self.device)
        # [sequence len, batch size]
        labels = self.loaded_labels[batch_slice].t().contiguous().to(self.device)

        return graph, labels
