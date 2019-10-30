import torch.nn as nn
from dgl import BatchedDGLGraph


class TokenEmbedding(nn.Module):
    def __init__(self, token_vocab_size: int, out_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(token_vocab_size, out_size)

    def forward(self, graph: BatchedDGLGraph) -> BatchedDGLGraph:
        graph.ndata['token_embeds'] = self.embedding(graph.ndata['token_id'])
        return graph
