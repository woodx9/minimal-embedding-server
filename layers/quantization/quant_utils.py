"""量化工具函数和类型定义"""
from dataclasses import dataclass


@dataclass
class ScalarType:
    """量化标量类型"""
    id: int
    size_bits: int
    name: str


def get_scalar_types():
    """获取可用的量化标量类型"""
    scalar_types_dict = {
        'uint4': ScalarType(id=0, size_bits=4, name='uint4'),
        'uint4b8': ScalarType(id=1, size_bits=4, name='uint4b8'),
        'uint8b128': ScalarType(id=2, size_bits=8, name='uint8b128'),
        'float8_e4m3fn': ScalarType(id=3, size_bits=8, name='float8_e4m3fn'),
        'float4_e2m1f': ScalarType(id=4, size_bits=4, name='float4_e2m1f'),
    }
    
    class ScalarTypes:
        def __init__(self, types_dict):
            for name, stype in types_dict.items():
                setattr(self, name, stype)
    
    return ScalarType, ScalarTypes(scalar_types_dict)
