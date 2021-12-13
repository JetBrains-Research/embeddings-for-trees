from typing import Tuple, List, Dict, Optional

import dgl
import torch
from commode_utils.metrics import SequentialF1Score, ChrF, ClassificationMetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torchmetrics import Metric, MetricCollection

from embeddings_for_trees.data.vocabulary import Vocabulary
from embeddings_for_trees.models.modules import TreeLSTM, NodeEmbedding
from embeddings_for_trees.models.modules.pointer_decoder import PointerDecoder


class TreeLSTM2SeqPointers(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: Vocabulary,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._model_config = model_config
        self._optim_config = optimizer_config
        self._vocabulary = vocabulary

        self._embedding = NodeEmbedding(model_config, vocabulary)
        self._encoder = TreeLSTM(model_config)
        self._decoder = PointerDecoder(model_config, vocabulary.token_to_id)

        token2id = vocabulary.token_to_id
        pad_idx = token2id[vocabulary.PAD]
        self._loss = torch.nn.NLLLoss(ignore_index=pad_idx)

        eos_idx = token2id[vocabulary.EOS]
        ignore_idx = [token2id[vocabulary.SOS], token2id[vocabulary.UNK]]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=pad_idx, eos_idx=eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }
        id2token = {v: k for k, v in token2id.items()}
        metrics.update(
            {f"{holdout}_chrf": ChrF(id2token, ignore_idx + [pad_idx, eos_idx]) for holdout in ["val", "test"]}
        )
        self._metrics = MetricCollection(metrics)

    def forward(  # type: ignore
        self, batched_trees: dgl.DGLGraph, output_length: int, target_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batched_trees.ndata["x"] = self._embedding(batched_trees)
        encoded_nodes = self._encoder(batched_trees)
        output_logits = self._decoder(batched_trees, encoded_nodes, output_length, target_sequence)
        return output_logits

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self._optim_config.lr,
            weight_decay=self._optim_config.weight_decay,
        )

        def scheduler_lambda(epoch: int) -> float:
            return self._optim_config.decay_gamma ** epoch

        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_lambda)
        return [optimizer], [scheduler]

    # ========== Model step ==========

    def _shared_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], step: str) -> Dict:
        labels, graph = batch
        # [seq length; batch size; vocab size]
        pass_labels = labels if step == "train" else None
        logits = self(graph, labels.shape[0], pass_labels)

        log_prob = (logits + 1e-9).log()
        loss = self._loss(log_prob[1:].permute(1, 2, 0), labels[1:].permute(1, 0))

        result = {f"{step}/loss": loss}

        with torch.no_grad():
            prediction = logits.argmax(-1)
            metric: ClassificationMetrics = self._metrics[f"{step}_f1"](prediction, labels)
            result.update(
                {f"{step}/f1": metric.f1_score, f"{step}/precision": metric.precision, f"{step}/recall": metric.recall}
            )
            if step != "train":
                result[f"{step}/chrf"] = self._metrics[f"{step}_chrf"](prediction, labels)

        return result

    def training_step(self, batch: Tuple[torch.Tensor, dgl.DGLGraph], batch_idx: int) -> Dict:  # type: ignore
        result = self._shared_step(batch, "train")
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("f1", result["train/f1"], prog_bar=True, logger=False, batch_size=batch[0].shape[1])
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
            metric = self._metrics[f"{step}_f1"].compute()
            log = {
                f"{step}/loss": mean_loss,
                f"{step}/f1": metric.f1_score,
                f"{step}/precision": metric.precision,
                f"{step}/recall": metric.recall,
            }
            self._metrics[f"{step}_f1"].reset()
            if step != "train":
                log[f"{step}/chrf"] = self._metrics[f"{step}_chrf"].compute()
                self._metrics[f"{step}_chrf"].reset()
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "train")

    def validation_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "val")

    def test_epoch_end(self, step_outputs: EPOCH_OUTPUT):
        self._shared_epoch_end(step_outputs, "test")
