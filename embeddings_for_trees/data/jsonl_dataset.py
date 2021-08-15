import json
from json import JSONDecodeError
from os.path import exists
from typing import Optional, Tuple, List, Dict

import dgl
import torch
from commode_utils.filesystem import get_lines_offsets, get_line_by_offset
from omegaconf import DictConfig
from torch.utils.data import Dataset

from embeddings_for_trees.utils.common import LABEL, CHILDREN, TOKEN, SEPARATOR, NODE, TYPE, AST
from embeddings_for_trees.data.vocabulary import Vocabulary, TypedVocabulary


class JsonlASTDataset(Dataset):
    _log_file = "bad_samples.log"

    def __init__(self, data_file: str, vocabulary: Vocabulary, config: DictConfig, is_train: bool):
        if not exists(data_file):
            raise ValueError(f"Can't find file with data: {data_file}")
        self._data_file = data_file
        self._vocab = vocabulary
        self._config = config
        self._is_train = is_train

        self._token_unk = self._vocab.token_to_id[vocabulary.UNK]
        self._node_unk = self._vocab.node_to_id[vocabulary.UNK]
        self._label_unk = self._vocab.label_to_id[vocabulary.UNK]

        self._line_offsets = get_lines_offsets(data_file)
        self._n_samples = len(self._line_offsets)

    def __len__(self):
        return self._n_samples

    def _read_sample(self, index: int) -> Optional[Dict]:
        raw_sample = get_line_by_offset(self._data_file, self._line_offsets[index])
        try:
            sample = json.loads(raw_sample)
        except JSONDecodeError as e:
            with open(self._log_file, "a") as log_file:
                log_file.write(raw_sample + "\n")
            return None
        if sample[LABEL] == "":
            with open(self._log_file, "a") as log_file:
                log_file.write(raw_sample + "\n")
            return None
        return sample

    def _build_graph(self, ast: Dict) -> Tuple[dgl.DGLGraph, List[Dict]]:
        # iterate through nodes
        node_to_parent: Dict[int, int] = {}
        nodes: List[Tuple[Optional[int], Dict]] = []  # list of (subtoken, node, parent)
        for n_id, node in enumerate(ast):
            parent_id = node_to_parent.get(n_id, None)
            if CHILDREN in node and len(CHILDREN) > 0:
                for c in node[CHILDREN]:
                    node_to_parent[c] = len(nodes)
                del node[CHILDREN]
                nodes.append((parent_id, node))
            else:  # if token is leaf than split it into several
                for subtoken in node[TOKEN].split(SEPARATOR)[: self._config.max_token_parts]:
                    new_node = node.copy()
                    new_node[TOKEN] = subtoken
                    nodes.append((parent_id, new_node))

        # convert to dgl graph
        us, vs = zip(*[(child, parent) for child, (parent, _) in enumerate(nodes) if parent is not None])
        graph = dgl.graph((us, vs))
        return graph, [node for parent, node in nodes]

    def _get_label(self, str_label: str) -> torch.Tensor:
        label = torch.full((self._config.max_label_parts + 1, 1), self._vocab.label_to_id[self._vocab.PAD])
        label[0, 0] = self._vocab.label_to_id[self._vocab.SOS]
        sublabels = str_label.split(SEPARATOR)[: self._config.max_label_parts]
        label[1 : len(sublabels) + 1, 0] = torch.tensor(
            [self._vocab.label_to_id.get(sl, self._label_unk) for sl in sublabels]
        )
        if len(sublabels) < self._config.max_label_parts:
            label[len(sublabels) + 1, 0] = self._vocab.label_to_id[self._vocab.EOS]
        return label

    def _is_suitable_tree(self, tree: dgl.DGLGraph) -> bool:
        if self._config.max_tree_nodes is not None and tree.number_of_nodes() > self._config.max_tree_nodes:
            return False
        if (
            self._config.max_tree_depth is not None
            and len(dgl.topological_nodes_generator(tree)) > self._config.max_tree_depth
        ):
            return False
        return True

    def _set_graph_features(self, graph: dgl.DGLGraph, nodes: List[Dict]) -> dgl.DGLGraph:
        n_nodes = len(nodes)
        graph.ndata[TOKEN] = torch.full(
            (n_nodes, self._config.max_token_parts), self._vocab.token_to_id[self._vocab.PAD]
        )
        graph.ndata[NODE] = torch.empty((n_nodes,), dtype=torch.long)

        for n_id, node in enumerate(nodes):
            sub_tokens = node[TOKEN].split(SEPARATOR)[: self._config.max_token_parts]
            sub_tokens_ids = [self._vocab.token_to_id.get(st, self._token_unk) for st in sub_tokens]
            graph.ndata[TOKEN][n_id, : len(sub_tokens_ids)] = torch.tensor(sub_tokens_ids)

            graph.ndata[NODE][n_id] = self._vocab.node_to_id.get(node[NODE], self._node_unk)

        return graph

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, dgl.DGLGraph]]:
        sample = self._read_sample(index)
        if sample is None:
            return None

        label = self._get_label(sample[LABEL])
        graph, nodes = self._build_graph(sample[AST])
        # We should filter out big trees even for validation because of OOM error.
        if not self._is_suitable_tree(graph):
            return None
        graph = self._set_graph_features(graph, nodes)
        assert graph.num_edges() > 0, index
        assert graph.num_nodes() > 0, index
        assert (
            graph.num_nodes() == graph.num_edges() + 1
        ), f"index: {index}, # edges: {graph.num_edges()}, # nodes: {graph.num_nodes()}"
        return label, graph

    def _print_tree(self, tree: dgl.DGLGraph, symbol: str = ".."):
        id_to_subtoken = {v: k for k, v in self._vocab.token_to_id.items()}
        id_to_node = {v: k for k, v in self._vocab.node_to_id.items()}
        node_depth = {0: 0}
        print(f"{id_to_subtoken[tree.ndata[TOKEN][0].item()]}/{id_to_node[tree.ndata[NODE][0].item()]}")

        edges = tree.edges()
        for edge_id in dgl.dfs_edges_generator(tree, 0, True):
            edge_id = edge_id.item()
            v, u = edges[0][edge_id].item(), edges[1][edge_id].item()
            cur_depth = node_depth[u] + 1
            node_depth[v] = cur_depth
            print(
                f"{symbol * cur_depth}"
                f"{id_to_subtoken[tree.ndata[TOKEN][v].item()]}/{id_to_node[tree.ndata[NODE][v].item()]}"
            )


class JsonlTypedASTDataset(JsonlASTDataset):
    _vocab: TypedVocabulary

    def __init__(self, data_file: str, vocabulary: TypedVocabulary, config: DictConfig, is_train: bool):
        super().__init__(data_file, vocabulary, config, is_train)
        self._type_unk = self._vocab.type_to_id[vocabulary.UNK]

    def _set_graph_features(self, graph: dgl.DGLGraph, nodes: List[Dict]) -> dgl.DGLGraph:
        n_nodes = len(nodes)
        graph.ndata[TOKEN] = torch.full(
            (n_nodes, self._config.max_token_parts), self._vocab.token_to_id[self._vocab.PAD]
        )
        graph.ndata[TYPE] = torch.full((n_nodes, self._config.max_type_parts), self._vocab.type_to_id[self._vocab.PAD])
        graph.ndata[NODE] = torch.empty((n_nodes,), dtype=torch.long)

        for n_id, node in enumerate(nodes):
            sub_tokens = node[TOKEN].split(SEPARATOR)[: self._config.max_token_parts]
            sub_tokens_ids = [self._vocab.token_to_id.get(st, self._token_unk) for st in sub_tokens]
            graph.ndata[TOKEN][n_id, : len(sub_tokens_ids)] = torch.tensor(sub_tokens_ids)

            sub_types = node[TYPE].split(SEPARATOR)[: self._config.max_type_parts]
            sub_types_ids = [self._vocab.type_to_id.get(st, self._type_unk) for st in sub_types]
            graph.ndata[TYPE][n_id, : len(sub_types_ids)] = torch.tensor(sub_types_ids)

            graph.ndata[NODE][n_id] = self._vocab.node_to_id.get(node[NODE], self._node_unk)

        return graph
