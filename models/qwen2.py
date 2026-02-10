import torch
from torch import nn
from transformers import Qwen2Config
import torch.distributed as dist

from layers.activation import SiluAndMul
from layers.embed_head import VocabParallelEmbedding
from layers.attention.base import create_attention
from layers.layernorm import RMSNorm
from layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from layers.rotary_embedding import get_rope
from schemas.config import MESConfig
from models.registry import register_model


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        mes_config: MESConfig,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.mes_config = mes_config
        
        # Get quantization config if available
        quant_config = mes_config.quantization_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # Use factory function to create attention based on config
        self.attn = create_attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            attn_backend=mes_config.attn_backend,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, positions)
        output = self.o_proj(o)
        return output


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mes_config: MESConfig,
    ) -> None:
        super().__init__()
        self.mes_config = mes_config
        
        # Get quantization config if available
        quant_config = mes_config.quantization_config
        
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        mes_config: MESConfig,
    ) -> None:
        super().__init__()
        self.mes_config = mes_config
        model_config = mes_config.model_config  # Qwen2Config的代称
        
        self.self_attn = Qwen2Attention(
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_attention_heads,
            num_kv_heads=model_config.num_key_value_heads,
            mes_config=mes_config,
            max_position=model_config.max_position_embeddings,
            rms_norm_eps=model_config.rms_norm_eps,
            qkv_bias=True,
            head_dim=getattr(model_config, 'head_dim', None),
            rope_theta=getattr(model_config, "rope_theta", 1000000),
            rope_scaling=getattr(model_config, "rope_scaling", None),
        )
        self.mlp = Qwen2MLP(
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            hidden_act=model_config.hidden_act,
            mes_config=mes_config,
        )
        self.input_layernorm = RMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        mes_config: MESConfig,
    ) -> None:
        super().__init__()
        self.mes_config = mes_config
        model_config = mes_config.model_config  # Qwen2Config的代称
        
        self.embed_tokens = VocabParallelEmbedding(model_config.vocab_size, model_config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(mes_config) for _ in range(model_config.num_hidden_layers)])
        self.norm = RMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_model("Qwen2ForCausalLM")
class Qwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        mes_config: MESConfig = None,
    ) -> None:
        super().__init__()
        if mes_config is None:
            mes_config = MESConfig()
        self.mes_config = mes_config
        self.model = Qwen2Model(mes_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states