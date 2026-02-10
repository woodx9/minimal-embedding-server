import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional
from layers.quantization.base_config import QuantizationConfig
    


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """Linear层基类，支持可选的量化"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_config = quant_config
        
        # 根据配置初始化量化方法
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self)
        else:
            self.quant_method = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """副本Linear层（无张量并行）"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(input_size, output_size, quant_config=quant_config)
        self.input_size = input_size
        self.output_size = output_size
        
        # 根据量化配置创建参数
        if self.quant_method is not None:
            self.quant_method.create_weights(
                layer=self,
                input_size_per_partition=input_size,
                output_partition_sizes=[output_size],
                input_size=input_size,
                output_size=output_size,
                params_dtype=torch.float16,
            )
        else:
            self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
            self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行Linear层，输出维度按TP切分"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(input_size, output_size, 0, quant_config=quant_config)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        
        if self.quant_method is not None:
            self.quant_method.create_weights(
                layer=self,
                input_size_per_partition=self.input_size_per_partition,
                output_partition_sizes=[self.output_size_per_partition],
                input_size=input_size,
                output_size=output_size,
                params_dtype=torch.float16,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(self.output_size_per_partition, self.input_size)
            )
            self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_weight) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(self, x, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
    
    

class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并的列并行Linear层（如gate_up_proj）"""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV并行Linear层，融合Q/K/V投影"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行Linear层，输入维度按TP切分，需要all_reduce"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(input_size, output_size, 1, quant_config=quant_config)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        if self.quant_method is not None:
            self.quant_method.create_weights(
                layer=self,
                input_size_per_partition=self.input_size_per_partition,
                output_partition_sizes=[self.output_size_per_partition],
                input_size=input_size,
                output_size=output_size,
                params_dtype=torch.float16,
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(self.output_size, self.input_size_per_partition)
            )
            self.weight.weight_loader = self.weight_loader
        
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_method is not None:
            y = self.quant_method.apply(self, x, self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
