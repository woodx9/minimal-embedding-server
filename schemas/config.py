from typing import Optional
from layers.quantization.awq_marlin import AWQMarlinConfig

class MESConfig:
    """Minimal Embedding Server配置"""
    def __init__(
        self,
        attn_backend: str = "flash_attn",
        model_name: str | None = None,
        max_tokens_per_batch: int | None = None,
        enable_monitoring: bool | None = None,
        dtype: str = "auto",
        model_config = None,
        quantization: Optional[str] = None,  # 用户指定的量化方法
    ):
        self.attn_backend = attn_backend
        self.model_name = model_name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.enable_monitoring = enable_monitoring
        self.dtype = dtype
        self.model_config = model_config
        
        # 从 model_config 加载量化配置
        self.quantization_config = self._load_quantization_config(model_config, quantization)
    
    def _load_quantization_config(self, model_config, user_method: Optional[str]):
        """从模型配置加载量化配置"""
        if not user_method:
            return None
        
        # 从 config 对象获取量化配置
        quant_config = getattr(model_config, 'quantization_config', None)
        
        if not quant_config:
            raise ValueError(
                f"指定了量化方法 '{user_method}'，"
                f"但模型配置中没有 quantization_config 字段。"
                f"请确认模型是量化模型。"
            )
        
        # quantization_config 可能是字典或对象
        if hasattr(quant_config, 'to_dict'):
            quant_config = quant_config.to_dict()
        
        # 获取模型配置中的量化方法
        file_method = quant_config.get("quant_method", "").lower()
        user_method_lower = user_method.lower()
        
        # 验证兼容性
        if user_method_lower == "awq_marlin" and file_method == "awq":
            print(f"[MESConfig] 检测到 AWQ 模型，使用 Marlin kernel 加速")
        elif user_method_lower != file_method:
            raise ValueError(
                f"量化方法不匹配：命令行指定 '{user_method_lower}'，"
                f"但模型配置是 '{file_method}'"
            )
        
        # 创建量化配置对象
        if user_method_lower in ["awq", "awq_marlin"]:
            
            bits = quant_config.get("bits", 4)
            group_size = quant_config.get("group_size", 128)
            zero_point = quant_config.get("zero_point", True)
            
            print(f"[MESConfig] 量化方法: awq_marlin")
            print(f"[MESConfig] 量化参数: bits={bits}, group_size={group_size}, zero_point={zero_point}")
            
            return AWQMarlinConfig(
                weight_bits=bits,
                group_size=group_size,
                zero_point=zero_point,
            )
        else:
            raise ValueError(f"不支持的量化方法: {user_method_lower}，目前仅支持 awq/awq_marlin")

