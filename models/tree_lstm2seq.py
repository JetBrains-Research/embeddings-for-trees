from typing import Tuple, List, Dict, Union

import dgl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from models.parts import NodeFeaturesEmbedding, TreeLSTM, LSTMDecoder
from utils.common import PAD
from utils.vocabulary import Vocabulary


class TreeLSTM2Seq(LightningModule):
    def __init__(self, config: DictConfig, vocabulary: Vocabulary):
        super().__init__()
        self.save_hyperparameters()
        self._config = config
        self._vocabulary = vocabulary

        self._embedding = NodeFeaturesEmbedding(config, vocabulary)
        self._encoder = TreeLSTM(config)
        self._decoder = LSTMDecoder(config, vocabulary)

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

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

    def _shared_step(self, graph: dgl.DGLGraph, labels: torch.Tensor, group: str) -> Dict:
        # [seq length; batch size; vocab size]
        logits = self(graph, labels.shape[0], labels)
        loss = self._calculate_loss(logits, labels)

        log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": loss}
        self.log_dict(log)
        return {"loss": loss}

    def training_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        return self._shared_step(batch[1], batch[0], "train")

    def validation_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        return self._shared_step(batch[1], batch[0], "val")

    def test_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        return self._shared_step(batch[1], batch[0], "test")

    # ========== On epoch end ==========

    def _shared_epoch_end(self, outputs: List[Dict], group: str):
        with torch.no_grad():
            mean_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
            log: Dict[str, Union[float, torch.Tensor]] = {f"{group}/loss": mean_loss}
            self.log_dict(log)
            self.log(f"{group}_loss", mean_loss)

    def training_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict]):
        self._shared_epoch_end(outputs, "test")
