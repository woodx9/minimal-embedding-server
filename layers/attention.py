import torch
from torch import nn
from flash_attn import flash_attn_func, flash_attn_varlen_func


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, positions: torch.Tensor):
        o: torch.Tensor

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        seq_boundaries = [0]
        if len(positions) > 1:
            jumps = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0]
            for jump_idx in jumps:
                seq_boundaries.append(jump_idx.item() + 1)
        seq_boundaries.append(len(positions))
        
        cu_seqlens = torch.tensor(seq_boundaries, dtype=torch.int32, device=q.device)
        
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()
        
        o = flash_attn_varlen_func(
            q, k, v, 
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True
        )

        o = o.reshape(-1, self.num_heads * self.head_dim)
        return o
