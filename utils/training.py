from typing import Dict, Tuple

import dgl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from model.tree2seq import Tree2Seq
from utils.common import EOS, UNK, PAD, get_root_indexes, is_step_match
from utils.learning_info import LearningInfo
from utils.logger import Logger
from utils.metrics import calculate_batch_statistics


def _forward_pass(
        model: Tree2Seq, graph: dgl.DGLGraph, labels: torch.Tensor,
        criterion: nn.modules.loss, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Make model step

    :param model: Tree2Seq model
    :param graph: batched dgl graph
    :param labels: [sequence len, batch size] ground truth labels
    :param criterion: criterion to optimize
    :param device: torch device
    :return: Tuple[
        loss [1] torch tensor with loss information
        prediction [the longest sequence, batch size]
        batch info [Dict] dict with statistics
    ]
    """
    root_indexes = torch.tensor(get_root_indexes(graph.batch_num_nodes), dtype=torch.long, device=device)
    root_logits = model(graph, root_indexes, labels, device)
    # remove <SOS> token
    # [the longest sequence, batch size, vocab size]
    root_logits = root_logits[1:]
    # [the longest sequence, batch size]
    labels = labels[1:]

    loss = criterion(root_logits.reshape(-1, root_logits.shape[-1]), labels.reshape(-1))
    # [the longest sequence, batch size]
    prediction = model.predict(root_logits)

    # Calculate metrics
    batch_info = {
        'loss': loss.item(),
        'statistics':
            calculate_batch_statistics(
                labels.t(), prediction.t(), [model.decoder.label_to_id[token] for token in [PAD, UNK, EOS]]
            )
    }

    return loss, prediction, batch_info


def train_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
        graph: dgl.DGLGraph, labels: torch.Tensor, clip_norm: int, device: torch.device
) -> Dict:
    model.train()

    # Model step
    model.zero_grad()
    loss, _, batch_info = _forward_pass(model, graph, labels, criterion, device)
    batch_info['learning_rate'] = scheduler.get_last_lr()[0]
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    scheduler.step()

    return batch_info


def eval_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, graph: dgl.DGLGraph,
        labels: torch.Tensor, device: torch.device
) -> Tuple[Dict, torch.Tensor]:
    model.eval()
    # Model step
    with torch.no_grad():
        _, prediction, batch_info = _forward_pass(model, graph, labels, criterion, device)

    return batch_info, prediction


def evaluate_on_dataset(
        dataset: Dataset, model: Tree2Seq, criterion: nn.modules.loss, device: torch.device
) -> LearningInfo:
    eval_epoch_info = LearningInfo()

    for batch_id in tqdm(range(len(dataset))):
        graph, labels = dataset[batch_id]
        batch_info, _ = eval_on_batch(
            model, criterion, graph, labels, device
        )
        eval_epoch_info.accumulate_info(batch_info)

    return eval_epoch_info


def train_on_dataset(
        train_dataset: Dataset, val_dataset, model: Tree2Seq, criterion: nn.modules.loss, optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler, clip_norm: int, logger: Logger, device: torch.device,
        start_batch_id: int = 0, log_step: int = -1, eval_step: int = -1, save_step: int = -1
):
    train_epoch_info = LearningInfo()

    batch_iterator_pb = tqdm(range(start_batch_id, len(train_dataset)), total=len(train_dataset))
    batch_iterator_pb.update(start_batch_id)
    batch_iterator_pb.refresh()

    for batch_id in batch_iterator_pb:
        graph, labels = train_dataset[batch_id]
        batch_info = train_on_batch(model, criterion, optimizer, scheduler, graph, labels, clip_norm, device)
        train_epoch_info.accumulate_info(batch_info)

        if is_step_match(batch_id, log_step):
            logger.log(train_epoch_info.get_state_dict(), batch_id, is_train=True)
            train_epoch_info = LearningInfo()

        if is_step_match(batch_id, save_step):
            train_dump = {
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'batch_id': batch_id
            }
            logger.save_model(f'batch_{batch_id}.pt', train_dump)

        if is_step_match(batch_id, eval_step):
            eval_info = evaluate_on_dataset(val_dataset, model, criterion, device)
            logger.log(eval_info.get_state_dict(), batch_id, is_train=False)

    logger.log(train_epoch_info.get_state_dict(), len(train_dataset), is_train=True)
