import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, quant_config=None):
    """加载模型权重，支持量化模型"""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 1. 加载所有权重
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if not weight_name.startswith("model."):
                    param_name = f"model.{weight_name}"
                else:
                    param_name = weight_name

                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = param_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    
    # 2. 权重加载完成后，处理量化权重（AWQ→Marlin格式转换）
    if quant_config is not None:
        print(f"[Loader] 开始处理量化权重: {quant_config.get_name()}")
        for name, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                print(f"[Loader] 处理量化层: {name}")
                quant_method.process_weights_after_loading(module)
        print(f"[Loader] 量化权重处理完成")
