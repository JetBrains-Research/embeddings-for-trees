import torch
import torch.nn as nn


class _IDecoder(nn.Module):
    """Decode a given batch of encoded vectors.
    Forward method return tensor [batch, max_output_length, k],
    where k corresponding to the size of prediction
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        pass


class LinearDecoder(_IDecoder):

    def __init__(self, in_size: int, out_size: int, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.linear(batch)
        return logits
