from typing import Dict

import torch
import torch.nn as nn


class _IDecoder(nn.Module):
    """Decode a given batch of encoded vectors.
    Forward method return tensor [batch, max_output_length, k],
    where k corresponding to the size of prediction
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: torch.Tensor, max_output_length: int) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, batch: torch.Tensor, pad_token: int) -> torch.Tensor:
        """Predict labels based on given logits"""
        raise NotImplementedError


class LinearDecoder(_IDecoder):

    def __init__(self, in_size: int, label_to_id: Dict, prediction_threshold: float = 0.0) -> None:
        super().__init__()
        self.prediction_threshold = prediction_threshold
        self.label_to_id = label_to_id
        out_size = len(label_to_id)
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.bias = nn.Parameter(torch.zeros((1, out_size)), requires_grad=True)

    def forward(self, batch: torch.Tensor, max_output_length: int) -> torch.Tensor:
        """returns tensor with shape [N, C, max_output_length]"""
        logits = self.linear(batch) + self.bias
        expanded_logits = logits.unsqueeze(1).expand(-1, max_output_length, -1)
        return expanded_logits.permute(0, 2, 1)

    def predict(self, logits: torch.Tensor, pad_token: int) -> torch.Tensor:
        """ Prediction based on the threshold
        :param logits: tensor [N, C, max_output_length]
        :param pad_token: a token id for fill empty places in prediction
        :return predicted_indexes: [N, max_predicted_sequence]
        """
        probabilities = nn.functional.softmax(logits[:, :, 0], dim=1)
        topk_probs, topk_labels = probabilities.topk(logits.shape[2])
        mask = (topk_probs >= self.prediction_threshold)
        topk_labels[~mask] = pad_token
        return topk_labels
