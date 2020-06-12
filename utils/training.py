from typing import Dict, Tuple

import dgl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from logger import AbstractLogger
from model.tree2seq import Tree2Seq
from utils.common import EOS, UNK, PAD, is_step_match
from utils.learning_info import LearningInfo
from utils.metrics import calculate_batch_statistics


def _forward_pass(
        model: Tree2Seq, graph: dgl.DGLGraph, labels: torch.Tensor, criterion: nn.modules.loss
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Make model step

    :param model: Tree2Seq model
    :param graph: batched dgl graph
    :param labels: [seq len; batch size] ground truth labels
    :param criterion: criterion to optimize
    :return: Tuple[
        loss [1] torch tensor with loss information
        prediction [the longest sequence, batch size]
        batch info [Dict] dict with statistics
    ]
    """
    # [seq len; batch size; vocab size]
    root_logits = model(graph, labels)

    # if seq len in labels equal to 1, then model solve classification task
    # for longer sequences we should remove <SOS> token, since it's always on the first place
    if labels.shape[0] > 1:
        # [seq len - 1; batch size; vocab size]
        root_logits = root_logits[1:]
        # [seq len - 1; batch size]
        labels = labels[1:]

    loss = criterion(root_logits.reshape(-1, root_logits.shape[-1]), labels.reshape(-1))
    # [the longest sequence, batch size]
    prediction = model.predict(root_logits)

    # Calculate metrics
    skipping_tokens = [model.decoder.label_to_id[token]
                       for token in [PAD, UNK, EOS]
                       if token in model.decoder.label_to_id]
    batch_info = {
        'loss': loss.item(),
        'statistics':
            calculate_batch_statistics(
                labels.t(), prediction.t(), skipping_tokens
            )
    }

    return loss, prediction, batch_info


def train_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, optimizer: torch.optim, scheduler: torch.optim.lr_scheduler,
        graph: dgl.DGLGraph, labels: torch.Tensor, clip_norm: int
) -> Dict:
    model.train()

    # Model step
    model.zero_grad()
    loss, prediction, batch_info = _forward_pass(model, graph, labels, criterion)
    batch_info['learning_rate'] = scheduler.get_last_lr()[0]
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), clip_norm)
    optimizer.step()
    scheduler.step()
    del loss
    del prediction
    torch.cuda.empty_cache()

    return batch_info


def eval_on_batch(
        model: Tree2Seq, criterion: nn.modules.loss, graph: dgl.DGLGraph,
        labels: torch.Tensor
) -> Tuple[Dict, torch.Tensor]:
    model.eval()
    # Model step
    with torch.no_grad():
        loss, prediction, batch_info = _forward_pass(model, graph, labels, criterion)
        del loss

    return batch_info, prediction


def evaluate_on_dataset(
        dataset: Dataset, model: Tree2Seq, criterion: nn.modules.loss
) -> LearningInfo:
    eval_epoch_info = LearningInfo()

    for batch_id in tqdm(range(len(dataset))):
        graph, labels = dataset[batch_id]
        batch_info, prediction = eval_on_batch(
            model, criterion, graph, labels
        )
        eval_epoch_info.accumulate_info(batch_info)
        del prediction

    return eval_epoch_info


def train_on_dataset(
        train_dataset: Dataset, val_dataset, model: Tree2Seq, criterion: nn.modules.loss, optimizer: torch.optim,
        scheduler: torch.optim.lr_scheduler, clip_norm: int, logger: AbstractLogger, start_batch_id: int = 0,
        log_step: int = -1, eval_step: int = -1, save_step: int = -1
):
    train_epoch_info = LearningInfo()

    batch_iterator_pb = tqdm(range(start_batch_id, len(train_dataset)), total=len(train_dataset))
    batch_iterator_pb.update(start_batch_id)
    batch_iterator_pb.refresh()

    for batch_id in batch_iterator_pb:
        graph, labels = train_dataset[batch_id]
        batch_info = train_on_batch(model, criterion, optimizer, scheduler, graph, labels, clip_norm)
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
            eval_info = evaluate_on_dataset(val_dataset, model, criterion)
            logger.log(eval_info.get_state_dict(), batch_id, is_train=False)

    if train_epoch_info.batch_processed > 0:
        logger.log(train_epoch_info.get_state_dict(), len(train_dataset) - 1, is_train=True)
