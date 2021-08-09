from typing import Tuple, List, Dict

import dgl
import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torchmetrics import MetricCollection, Metric

from embeddings_for_trees.data.vocabulary import Vocabulary
from embeddings_for_trees.models.modules import NodeEmbedding, TreeLSTM


class TreeLSTM2Seq(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: Vocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._model_config = model_config
        self.__optim_config = optimizer_config
        self._vocabulary = vocabulary

        if vocabulary.SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")

        self.__pad_idx = vocabulary.label_to_id[vocabulary.PAD]
        eos_idx = vocabulary.label_to_id[vocabulary.EOS]
        ignore_idx = [vocabulary.label_to_id[vocabulary.SOS]]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=self.__pad_idx, eos_idx=eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }
        self.__metrics = MetricCollection(metrics)

        self.__embedding = self._get_embedding()
        self.__encoder = TreeLSTM(model_config)
        decoder_step = LSTMDecoderStep(model_config, len(vocabulary.label_to_id), self.__pad_idx)
        self.__decoder = Decoder(
            decoder_step, len(vocabulary.label_to_id), vocabulary.label_to_id[vocabulary.SOS], teacher_forcing
        )

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, reduction="batch-mean")

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _get_embedding(self) -> torch.nn.Module:
        return NodeEmbedding(self._model_config, self._vocabulary)

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.__optim_config.lr,
            weight_decay=self.__optim_config.weight_decay,
        )

        def scheduler_lambda(epoch: int) -> float:
            return self.__optim_config.decay_gamma ** epoch

        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_lambda)
        return [optimizer], [scheduler]

    def forward(  # type: ignore
        self,
        batched_trees: dgl.DGLGraph,
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        batched_trees.ndata["x"] = self.__embedding(batched_trees)
        encoded_nodes = self.__encoder(batched_trees)
        output_logits = self.__decoder(encoded_nodes, batched_trees.batch_num_nodes(), output_length, target_sequence)
        return output_logits

    # ========== Model step ==========

    def _shared_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], step: str) -> Dict:
        labels, graph = batch
        # [seq length; batch size; vocab size]
        logits = self(graph, labels.shape[0], labels) if step == "train" else self(graph, labels.shape[0])
        loss = self.__loss(logits[1:], labels[1:])

        with torch.no_grad():
            prediction = logits.argmax(-1)
            metric: ClassificationMetrics = self.__metrics[f"{step}_f1"](prediction, labels)

        return {
            f"{step}/loss": loss,
            f"{step}/f1": metric.f1_score,
            f"{step}/precision": metric.precision,
            f"{step}/recall": metric.recall,
        }

    def training_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "train")
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("f1", result["train/f1"], prog_bar=True, logger=False)
        return result["train/loss"]

    def validation_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "val")
        return result["val/loss"]

    def test_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "test")
        return result["test/loss"]

    # ========== On epoch end ==========

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, step: str):
        with torch.no_grad():
            losses = [so if isinstance(so, torch.Tensor) else so["loss"] for so in step_outputs]
            mean_loss = torch.stack(losses).mean()
            metric = self.__metrics[f"{step}_f1"].compute()
            log = {
                f"{step}/loss": mean_loss,
                f"{step}/f1": metric.f1_score,
                f"{step}/precision": metric.precision,
                f"{step}/recall": metric.recall,
            }
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "val")

    def test_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "test")
