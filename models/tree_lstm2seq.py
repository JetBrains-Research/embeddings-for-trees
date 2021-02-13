from typing import Tuple, List, Dict, Union

import dgl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from models.parts import NodeEmbedding, LSTMDecoder, TreeLSTM
from utils.common import PAD, UNK, EOS, SOS
from utils.metrics import PredictionStatistic
from utils.vocabulary import Vocabulary


class TreeLSTM2Seq(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self._vocabulary = vocabulary

        if SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")
        self._label_pad_id = vocabulary.label_to_id[PAD]
        self._metric_skip_tokens = [
            vocabulary.label_to_id[i] for i in [PAD, UNK, EOS, SOS] if i in vocabulary.label_to_id
        ]

        self._embedding = self._get_embedding()
        self._encoder = TreeLSTM(config)
        self._decoder = LSTMDecoder(config, vocabulary)

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _get_embedding(self) -> torch.nn.Module:
        return NodeEmbedding(self._config, self._vocabulary)

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(self.parameters(), lr=self._config.learning_rate, weight_decay=self._config.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: self._config.lr_decay_gamma ** epoch)
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

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        batch_metric = statistic.update_statistic(labels, prediction)

        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}
        for key, value in batch_metric.items():
            log[f"train/{key}"] = value
        self.log_dict(log)
        self.log("f1", batch_metric["f1"], prog_bar=True, logger=False)

        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        labels, graph = batch
        # [seq length; batch size; vocab size]
        logits = self(graph, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)
        prediction = logits.argmax(-1)

        statistic = PredictionStatistic(True, self._label_pad_id, self._metric_skip_tokens)
        statistic.update_statistic(labels, prediction)

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        return self.validation_step(batch, batch_idx)

    # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            statistic = PredictionStatistic.create_from_list([out["statistic"] for out in outputs])
            epoch_metrics = statistic.get_metric()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}
            for key, value in epoch_metrics.items():
                log[f"{group}/{key}"] = value
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")
