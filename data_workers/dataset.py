from os import listdir
from os.path import exists as path_exists
from os.path import join as path_join
from pickle import load as pkl_load
from typing import Tuple, List

from dgl import DGLGraph, unbatch, batch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class JavaDataset(Dataset):

    def __init__(self, batched_graphs_path: str, batch_size: int) -> None:
        self.batched_graphs_path = batched_graphs_path
        self.batch_size = batch_size
        assert path_exists(self.batched_graphs_path)

        self.batched_graph_files = sorted(list(filter(
            lambda filename: filename.endswith('.pkl'),
            listdir(self.batched_graphs_path)
        )), key=lambda name: int(name[6:-4]))
        self.batch_desc = {}
        self.n_batches = 0

        # iterate over pkl files to aggregate information about batches
        print(f"prepare the {batched_graphs_path} dataset...")
        for batched_graph_file in tqdm(self.batched_graph_files):
            with open(path_join(self.batched_graphs_path, batched_graph_file), 'rb') as pkl_file:
                batched_graph = pkl_load(pkl_file)
            n_graphs = len(batched_graph['batched_graph'].batch_num_nodes)
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

    def __getitem__(self, item) -> Tuple[DGLGraph, List[str]]:
        batch_basename, batch_slice = self.batch_desc[item]
        with open(path_join(self.batched_graphs_path, batch_basename), 'rb') as pkl_file:
            cur_batched_graph = pkl_load(pkl_file)
        graphs = unbatch(cur_batched_graph['batched_graph'])
        batched_graph = batch(graphs[batch_slice])
        labels = cur_batched_graph['labels'][batch_slice]
        return batched_graph, labels
