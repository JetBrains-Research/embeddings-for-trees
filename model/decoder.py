import torch
import torch.nn as nn


class LinearDecoder(nn.Module):

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        logits = self.linear(batch)
        return logits
