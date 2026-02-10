"""Quantization layers package"""
from layers.quantization.base_config import (
    QuantizationConfig,
    LinearMethodBase,
)
from layers.quantization.awq_marlin import (
    AWQMarlinConfig,
    AWQMarlinLinearMethod,
)

__all__ = [
    'QuantizationConfig',
    'LinearMethodBase',
    'AWQMarlinConfig',
    'AWQMarlinLinearMethod',
]
