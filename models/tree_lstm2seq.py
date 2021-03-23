from typing import Tuple, List, Dict

import dgl
import torch
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from models.parts import NodeEmbedding, LSTMDecoder, TreeLSTM
from utils.common import PAD, UNK, EOS, SOS
from utils.vocabulary import Vocabulary


class TreeLSTM2Seq(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self._vocabulary = vocabulary

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")

        pad_idx = vocabulary.label_to_id[PAD]
        ignore_idx = [vocabulary.label_to_id[i] for i in [UNK, EOS, SOS] if i in vocabulary.label_to_id]
        self._train_metrics = SequentialF1Score(mask_after_pad=True, pad_idx=pad_idx, ignore_idx=ignore_idx)
        self._val_metrics = SequentialF1Score(
            mask_after_pad=True, pad_idx=pad_idx, ignore_idx=ignore_idx, compute_on_step=False
        )
        self._test_metrics = SequentialF1Score(
            mask_after_pad=True, pad_idx=pad_idx, ignore_idx=ignore_idx, compute_on_step=False
        )

        self._embedding = self._get_embedding()
        self._encoder = TreeLSTM(config.model)
        self._decoder = LSTMDecoder(config.model, vocabulary)

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _get_embedding(self) -> torch.nn.Module:
        return NodeEmbedding(self._config.model, self._vocabulary)

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(
            self.parameters(),
            lr=self._config.trainer.learning_rate,
            weight_decay=self._config.trainer.weight_decay,
        )

        def scheduler_lambda(epoch: int) -> int:
            return self._config.trainer.lr_decay_gamma ** epoch

        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_lambda)
        return [optimizer], [scheduler]

    def forward(  # type: ignore
        self,
        batched_trees: dgl.DGLGraph,
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        batched_trees.ndata["x"] = self._embedding(batched_trees)
        encoded_nodes = self._encoder(batched_trees)
        output_logits = self._decoder(encoded_nodes, batched_trees.batch_num_nodes(), output_length, target_sequence)
        return output_logits

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate cross entropy with ignoring PAD index
        :param logits: [seq length; batch size; vocab size]
        :param labels: [seq length; batch size]
        :return: [1]
        """
        batch_size = labels.shape[-1]
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = labels.permute(1, 0)
        # [batch size; seq length]
        loss = torch.nn.functional.cross_entropy(_logits, _labels, reduction="none")
        # [batch size; seq length]
        mask = _labels != self._vocabulary.label_to_id[PAD]
        # [batch size; seq length]
        loss = loss * mask
        # [1]
        loss = loss.sum() / batch_size
        return loss

    # ========== Model step ==========

    def training_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        labels, graph = batch
        # [seq length; batch size; vocab size]
        logits = self(graph, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        batch_metrics: ClassificationMetrics = self._train_metrics(prediction, labels)
        log = {
            "train/loss": loss,
            "train/f1": batch_metrics.f1_score,
            "train/precision": batch_metrics.precision,
            "train/recall": batch_metrics.recall,
        }
        self.log_dict(log)
        self.log("f1", batch_metrics.f1_score, prog_bar=True, logger=False)

        return {"loss": loss}

    def validation_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        labels, graph = batch
        # [seq length; batch size; vocab size]
        logits = self(graph, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        self._val_metrics(prediction, labels)
        return {"loss": loss}

    def test_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        labels, graph = batch
        # [seq length; batch size; vocab size]
        logits = self(graph, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        self._test_metrics(prediction, labels)
        return {"loss": loss}

    # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str, metric: SequentialF1Score):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            epoch_metric = metric.compute()
            log = {
                f"{group}/loss": mean_loss,
                f"{group}/f1": epoch_metric.f1_score,
                f"{group}/precision": epoch_metric.precision,
                f"{group}/recall": epoch_metric.recall,
            }
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train", self._train_metrics)

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val", self._val_metrics)

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test", self._test_metrics)
