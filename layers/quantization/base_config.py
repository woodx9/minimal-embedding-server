"""量化配置和方法的基类"""
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import nn


class QuantizationConfig(ABC):
    """量化配置基类"""
    
    @abstractmethod
    def get_name(self) -> str:
        """返回量化方法名称"""
        pass
    
    @abstractmethod
    def get_quant_method(self, layer: nn.Module) -> Optional['LinearMethodBase']:
        """返回对应的量化方法"""
        pass


class LinearMethodBase(ABC):
    """Linear层量化方法基类"""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def process_weights_after_loading(self, layer: nn.Module):
        """加载后处理权重（如重打包为Marlin格式）"""
        pass
    
    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """应用量化线性变换"""
        pass
