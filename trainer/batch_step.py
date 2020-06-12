from typing import Dict, Tuple

import dgl
import torch
from torch import nn

from model.tree2seq import Tree2Seq
from utils.common import PAD, UNK, EOS
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
