from typing import Dict, Type

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(architecture: str):
    def decorator(model_class):
        MODEL_REGISTRY[architecture] = model_class
        return model_class
    return decorator


def get_model_class(config):
    if not hasattr(config, 'architectures') or not config.architectures:
        raise ValueError(f"Model config does not have 'architectures' field")
    
    architecture = config.architectures[0]
    
    if architecture not in MODEL_REGISTRY:
        available_archs = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model architecture: {architecture}. "
            f"Available architectures: {available_archs}"
        )
    
    return MODEL_REGISTRY[architecture]