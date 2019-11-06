import torch.nn as nn
from torch import device
from dgl import BatchedDGLGraph


class _IEmbedding(nn.Module):
    """Interface of embedding module.
    Forward method takes batched graph and applies embedding to its features.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        return graph


class TokenEmbedding(_IEmbedding):
    def __init__(self, token_vocab_size: int, out_size: int, using_device: device, **kwargs) -> None:
        super().__init__()
        self.device = using_device
        self.token_embedding = nn.Embedding(token_vocab_size, out_size)

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        graph.ndata['token_embeds'] = self.token_embedding(graph.ndata['token_id']).to(self.device)
        return graph
