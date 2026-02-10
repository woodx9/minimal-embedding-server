"""AWQ Marlin量化实现"""
from typing import Optional
import torch
from torch import nn
from sgl_kernel import (
    awq_marlin_repack,
    gptq_marlin_gemm,
)
from layers.quantization.base_config import QuantizationConfig, LinearMethodBase
from layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_make_empty_g_idx,
    verify_marlin_supports_shape,
    marlin_permute_scales,
    awq_to_marlin_zero_points,
)
from layers.quantization.quant_utils import ScalarType, get_scalar_types


_, scalar_types = get_scalar_types()


class AWQMarlinConfig(QuantizationConfig):
    """AWQ Marlin量化配置"""
    
    def __init__(
        self,
        weight_bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
    ):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // self.weight_bits  # 8 for 4-bit
        
        if self.weight_bits != 4:
            raise ValueError(
                f"目前AWQ Marlin仅支持4-bit量化，但收到{self.weight_bits} bits"
            )
        
        self.quant_type = scalar_types.uint4
    
    def get_name(self) -> str:
        return "awq_marlin"
    
    def get_quant_method(self, layer: nn.Module) -> LinearMethodBase:
        return AWQMarlinLinearMethod(self)
    
    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
        )


class AWQMarlinLinearMethod(LinearMethodBase):
    """AWQ Marlin量化方法"""
    
    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config
    
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """创建量化权重参数"""
        output_size_per_partition = sum(output_partition_sizes)
        
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        
        # 验证shape是否与Marlin兼容
        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size,
        )
        
        num_groups = input_size_per_partition // group_size
        
        # qweight: INT4权重打包到INT32
        qweight = nn.Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        
        # qzeros: per-group zero-points
        qzeros = nn.Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        
        # scales: per-group缩放因子
        scales = nn.Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)
        
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups
    
    def process_weights_after_loading(self, layer: nn.Module):
        """加载后将权重从AWQ格式重打包为Marlin格式"""
        device = layer.qweight.device
        
        # 创建Marlin workspace（kernel临时缓冲区）
        layer.workspace = marlin_make_workspace(device)
        
        # 重打包qweight: AWQ → Marlin
        marlin_qweight = awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        layer.qweight = nn.Parameter(marlin_qweight, requires_grad=False)
        
        # 重排scales
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size,
        )
        layer.scales = nn.Parameter(marlin_scales, requires_grad=False)
        
        # 重排zero-points
        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        layer.qzeros = nn.Parameter(marlin_zp, requires_grad=False)
        
        # 创建空g_idx（AWQ不用，但kernel需要）
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)
    
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用Marlin kernel执行量化矩阵乘法"""
        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        
        output = gptq_marlin_gemm(
            reshaped_x,
            None,
            layer.qweight,
            layer.scales,
            None,
            layer.qzeros,
            layer.g_idx,
            layer.g_idx_sort_indices,
            layer.workspace,
            self.quant_config.quant_type,
            size_m=reshaped_x.shape[0],
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        
        if bias is not None:
            output.add_(bias)
        
        return output.reshape(out_shape)
