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


def train_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
        graph: dgl.BatchedDGLGraph, labels: List[str],
        params: Dict, device: torch.device
) -> Dict:
    model.train()
    root_indexes = get_root_indexes(graph).to(device)

    # Model step
    model.zero_grad()
    root_logits, ground_truth = model(graph, root_indexes, labels, params['teacher_force'], device)
    root_logits = root_logits[1:]
    ground_truth = ground_truth[1:]
    loss = criterion(root_logits.view(-1, root_logits.shape[-1]), ground_truth.view(-1))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), params['clip_norm'])
    optimizer.step()
    scheduler.step()

    # Calculate metrics
    prediction = model.predict(root_logits)
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

    root_indexes = get_root_indexes(graph).to(device)

    # Model step
    with torch.no_grad():
        root_logits, ground_truth = model(graph, root_indexes, labels, 0.0, device)
        root_logits = root_logits[1:]
        ground_truth = ground_truth[1:]
        loss = criterion(root_logits.view(-1, root_logits.shape[-1]), ground_truth.view(-1))
        prediction = model.predict(root_logits)

    # Calculate metrics
    batch_eval_info = {
        'loss': loss.item(),
        'statistics':
            calculate_batch_statistics(
                ground_truth.t(), prediction.t(), [model.decoder.label_to_id[token] for token in [PAD, UNK, EOS]]
            )
    }
    del root_logits, ground_truth, loss
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
