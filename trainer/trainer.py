import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from logger import AbstractLogger
from model.tree2seq import Tree2Seq
from trainer.batch_step import eval_on_batch, train_on_batch
from utils.common import is_step_match
from utils.learning_info import LearningInfo


def evaluate_on_dataset(dataset: Dataset, model: Tree2Seq, criterion: nn.modules.loss) -> LearningInfo:
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
