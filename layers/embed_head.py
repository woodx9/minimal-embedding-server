import math
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
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.num_embeddings = num_embeddings
        self.padded_num_embeddings = int(
            math.ceil(num_embeddings / self.tp_size) * self.tp_size
        )
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = min(
            self.vocab_start_index + self.num_embeddings_per_partition,
            self.num_embeddings,
        )

        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        param_data.zero_()
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        end_idx = min(start_idx + shard_size, self.num_embeddings)
        if end_idx <= start_idx:
            return
        loaded_slice = loaded_weight.narrow(0, start_idx, end_idx - start_idx)
        param_data[: end_idx - start_idx].copy_(loaded_slice)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)
            x = x - self.vocab_start_index
            x = x.masked_fill(~mask, 0)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y
