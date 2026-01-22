"""
Base Attention class and factory for creating different attention implementations
"""
import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseAttention(nn.Module, ABC):
    """Base class for all attention implementations"""
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
    
    @abstractmethod
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for attention
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            positions: Position indices
            
        Returns:
            Output tensor after attention
        """
        pass


def create_attention(
    num_heads: int,
    head_dim: int,
    scale: float,
    num_kv_heads: int,
    attn_backend: str = "flashAttention",
    device: str = "cuda:0",
) -> BaseAttention:
    """
    Factory function to create attention based on backend type
    
    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        scale: Scaling factor for attention scores
        num_kv_heads: Number of key-value heads
        attn_backend: Backend type ("flashAttention" or "flashInfer")
        device: Device to use
        
    Returns:
        Attention instance
    """
    if attn_backend == "flash_attention":
        from layers.attention.flashattention import FlashAttention
        return FlashAttention(num_heads, head_dim, scale, num_kv_heads)
    elif attn_backend == "flash_infer":
        from layers.attention.flashinfer import FlashInferAttention
        return FlashInferAttention(num_heads, head_dim, scale, num_kv_heads, device)
    else:
        raise ValueError(f"Unknown attention backend: {attn_backend}")
