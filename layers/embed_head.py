import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor):
        y = F.embedding(x, self.weight)
        return y