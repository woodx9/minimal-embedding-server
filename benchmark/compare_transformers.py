# start benchmark cli
# python benchmark/compare_transformers.py

import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.config import MSEConfig
import time
from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from models.qwen3 import Qwen3ForCausalLM
from ultils.loader import load_model

def main():
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_run_time = 100
    input_text_length = 100  # 输入文本长度倍增因子

    # 使用第一个可见的 GPU（受 CUDA_VISIBLE_DEVICES 控制）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    mse_config = MSEConfig(attn_backend="flash_infer")
    config = AutoConfig.from_pretrained(model_name)

    model_runner = Qwen3ForCausalLM(config, mse_config)
    model_path = snapshot_download(model_name)
    load_model(model_runner, model_path)
    
    # 将模型移到 GPU 并转换为 bf16
    model_runner = model_runner.to(device).to(torch.bfloat16)
    model_runner.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 测试文本
    text = "Hello, how are you?" * input_text_length
    print(f"测试文本长度: {len(text)} 字符")
    
    # ========== 使用自定义模型 ==========
    print("\n" + "=" * 60)
    print("使用自定义模型 (model_runner)")
    print("=" * 60)
    
    # 编码
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded['input_ids'].squeeze(0).tolist()
    seq_len = len(input_ids)
    positions = list(range(seq_len))
    
    # 转换为 tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)
    positions = torch.tensor(positions, dtype=torch.int64, device=device)

    # Warmup run
    with torch.no_grad():
        outputs = model_runner(input_ids=input_ids, positions=positions)

    start_time = time.perf_counter()
    for _ in range(model_run_time):
        with torch.no_grad():
            outputs = model_runner(input_ids=input_ids, positions=positions)
    end_time = time.perf_counter()
    print(f"{model_run_time}次推理时间: {end_time - start_time:.4f} 秒")
    
    # ========== 使用 Transformers 库 ==========
    print("\n" + "=" * 60)
    print("使用 Transformers 库")
    print("=" * 60)
    
    model_tf = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to(device)
    model_tf.eval()
    
    # 复用 tokenizer 和 text
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Warmup run
    with torch.no_grad():
        outputs_tf = model_tf(**encoded)

    start_time = time.perf_counter()
    for _ in range(model_run_time):
        with torch.no_grad():
            outputs_tf = model_tf(**encoded)
    end_time = time.perf_counter()
    print(f"{model_run_time}次推理时间: {end_time - start_time:.4f} 秒")


if __name__ == "__main__":
    main()
