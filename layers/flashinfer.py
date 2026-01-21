from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
import torch
from torch import nn

# 全局 workspace buffer，只创建一次，所有层共享
global_workspace_buffer = None

def get_global_workspace_buffer(workspace_size=16 * 1024 * 1024, device="cuda"):
    """获取全局 workspace buffer，如果不存在则创建"""
    global global_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, 
            dtype=torch.int64, 
            device=device
        )
    return global_workspace_buffer

class FlashInferAttention(nn.Module):
    def __init__(self, num_heads,
        head_dim,
        scale,
        num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.workspace_size = 512 * 1024 * 1024

        # 使用全局共享的 workspace buffer
        self.workspace_buffer = get_global_workspace_buffer(
            self.workspace_size,
            "cuda:1"
        )
        
        # 创建 wrapper（推荐在初始化时创建，避免重复创建开销）
        self.prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",  # 数据布局
            backend="fa2"
        )
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        positions: torch.Tensor
    ):
        # Reshape to [total_tokens, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # 计算序列边界（与你原来的逻辑相同）
        seq_boundaries = [0]
        if len(positions) > 1:
            jumps = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0]
            for jump_idx in jumps:
                seq_boundaries.append(jump_idx.item() + 1)
        seq_boundaries.append(len(positions))
        
        # 构建 cu_seqlens（累积序列长度）
        cu_seqlens = torch.tensor(
            seq_boundaries, 
            dtype=torch.int64, 
            device=q.device
        )
        
        # 调用 begin_forward 设置元数据
        self.prefill_wrapper.begin_forward(
            cu_seqlens,  # qo_indptr
            cu_seqlens,  # kv_indptr（对于 embedding，Q 和 KV 长度相同）
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            q_data_type=q.dtype
        )
        
        # 执行 forward
        o = self.prefill_wrapper.forward(
            q, k, v,
            causal=True,
            sm_scale=self.scale, 
        )
        
        # Reshape 回原来的形状
        o = o.reshape(-1, self.num_heads * self.head_dim)
        return o