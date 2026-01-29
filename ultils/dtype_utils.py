"""
数据类型工具函数
"""
import torch


def get_torch_dtype(dtype_str: str, config=None) -> torch.dtype:
    """根据配置字符串返回torch dtype"""
    if dtype_str == "auto":
        return _get_auto_dtype(config)
    
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16, 
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    
    if dtype_str.lower() in dtype_map:
        return dtype_map[dtype_str.lower()]
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


def _get_auto_dtype(config=None) -> torch.dtype:
    """自动确定数据类型"""
    # 从模型config读取torch_dtype
    if config is not None:
        model_dtype = getattr(config, 'torch_dtype', None)
        if model_dtype is not None and model_dtype != torch.float32:
            return model_dtype
    
    # 默认选择最佳dtype
    return _get_best_dtype()


def _get_best_dtype() -> torch.dtype:
    """根据GPU能力选择最佳dtype"""
    if _gpu_supports_bf16():
        return torch.bfloat16
    else:
        return torch.float16


def check_dtype_compatibility(dtype: torch.dtype, attn_backend: str) -> tuple[bool, torch.dtype, str]:
    """检查dtype与attention backend的兼容性"""
    # 检查attention backend是否支持float32
    if dtype == torch.float32 and attn_backend in ["flash_infer", "flash_attention"]:
        best_dtype = _get_best_dtype()
        return False, best_dtype, f"{attn_backend} doesn't support float32, using {dtype_to_string(best_dtype)}"
    
    return True, dtype, ""


def _gpu_supports_bf16() -> bool:
    """检查GPU是否支持bfloat16"""
    return hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()


def dtype_to_string(dtype: torch.dtype) -> str:
    """将torch.dtype转换为字符串"""
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    return dtype_map.get(dtype, str(dtype))


def validate_dtype_compatibility(dtype: torch.dtype) -> bool:
    """验证数据类型与GPU的兼容性"""
    if dtype == torch.bfloat16:
        return _gpu_supports_bf16()
    return True