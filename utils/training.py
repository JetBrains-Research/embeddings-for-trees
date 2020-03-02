from typing import List, Dict, Tuple

import dgl
import torch
import torch.nn as nn
from numpy import cumsum
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from model.tree2seq import Tree2Seq
from utils.common import EOS, UNK, PAD
from utils.learning_info import LearningInfo
from utils.metrics import calculate_batch_statistics


def get_root_indexes(graph: dgl.BatchedDGLGraph) -> torch.Tensor:
    """Get indexes of roots in given graph

    :param graph: batched dgl graph
    :return: tensor with indexes of roots [batch size]
    """
    mask = torch.zeros(graph.number_of_nodes())
    idx_of_roots = cumsum([0] + graph.batch_num_nodes)[:-1]
    mask[idx_of_roots] = 1
    root_indexes = torch.nonzero(mask).squeeze(1)
    return root_indexes


def _process_data(
        model: Tree2Seq, graph: dgl.BatchedDGLGraph, labels: List[str],
        criterion: nn.modules.loss, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make model step

    :param model: Tree2Seq model
    :param graph: batched dgl graph
    :param labels: [batch size] list of string labels
    :param criterion: criterion to optimize
    :param device: torch device
    :return: Tuple[
        loss [1] torch tensor with loss information
        prediction [the longest sequence, batch size]
        ground truth [the longest sequence, batch size]
    ]
    """
    root_indexes = get_root_indexes(graph).to(device)

    root_logits, ground_truth = model(graph, root_indexes, labels, device)
    root_logits = root_logits[1:]
    ground_truth = ground_truth[1:]

    loss = criterion(root_logits.view(-1, root_logits.shape[-1]), ground_truth.view(-1))
    prediction = model.predict(root_logits)

    return loss, prediction, ground_truth


def train_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
        graph: dgl.BatchedDGLGraph, labels: List[str],
        params: Dict, device: torch.device
) -> Dict:
    model.train()

    # Model step
    model.zero_grad()
    loss, prediction, ground_truth = _process_data(model, graph, labels, criterion, device)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), params['clip_norm'])
    optimizer.step()
    scheduler.step()

    # Calculate metrics
    batch_train_info = {
        'loss': loss.item(),
        'learning_rate': scheduler.get_lr()[0],
        'statistics':
            calculate_batch_statistics(
                ground_truth.t(), prediction.t(), [model.decoder.label_to_id[token] for token in [PAD, UNK, EOS]]
            )
    }
    return batch_train_info


def eval_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, graph: dgl.BatchedDGLGraph, labels: List[str], device: torch.device
) -> Tuple[Dict, torch.Tensor]:
    model.eval()
    # Model step
    with torch.no_grad():
        loss, prediction, ground_truth = _process_data(model, graph, labels, criterion, device)

    # Calculate metrics
    batch_eval_info = {
        'loss': loss.item(),
        'statistics':
            calculate_batch_statistics(
                ground_truth.t(), prediction.t(), [model.decoder.label_to_id[token] for token in [PAD, UNK, EOS]]
            )
    }
    return batch_eval_info, prediction


def evaluate_dataset(
        dataset: Dataset, model: Tree2Seq, criterion: nn.modules.loss, device: torch.device
) -> LearningInfo:
    eval_epoch_info = LearningInfo()

    for batch_id in tqdm(range(len(dataset))):
        graph, labels = dataset[batch_id]
        graph.ndata['token_id'] = graph.ndata['token_id'].to(device)
        graph.ndata['type_id'] = graph.ndata['type_id'].to(device)
        batch_info, _ = eval_on_batch(
            model, criterion, graph, labels, device
        )
        eval_epoch_info.accumulate_info(batch_info)

    return eval_epoch_info
