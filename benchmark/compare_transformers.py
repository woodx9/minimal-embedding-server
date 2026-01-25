# start benchmark cli
# python benchmark/compare_transformers.py

import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.config import MSEConfig
import time
import asyncio
from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch.nn.functional as F
from core.engine import Engine
from ultils.loader import load_model

def main():
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_run_time = 100
    input_text_length = 50  # 输入文本长度倍增因子

    # 使用第一个可见的 GPU（受 CUDA_VISIBLE_DEVICES 控制）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    config = AutoConfig.from_pretrained(model_name)
    mes_engine = Engine(attn_backend="flash_attention", tensor_parallel_size=1)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 测试文本
    text = "Hello, how are you?" * input_text_length
    print(f"测试文本长度: {len(text)} 字符")
    
    # ========== 使用自定义模型 ==========
    print("\n" + "=" * 60)
    print("使用自定义模型 (mes Engine)")
    print("=" * 60)
    
    # 预热运行
    with torch.no_grad():
        embeddings_list, seq_lengths = asyncio.run(mes_engine.v1_embeddings(input=[text]))
        embedding_custom = torch.tensor(embeddings_list[0], device=device, dtype=torch.bfloat16)

    # 性能测试 - 多次运行
    start_time = time.perf_counter()
    for _ in range(model_run_time):
        with torch.no_grad():
            embeddings_list, seq_lengths = asyncio.run(mes_engine.v1_embeddings(input=[text]))
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
        embedding_tf = outputs_tf.last_hidden_state[:, -1, :]
        embedding_tf = F.normalize(embedding_tf, p=2, dim=1)

    start_time = time.perf_counter()
    for _ in range(model_run_time):
        with torch.no_grad():
            outputs_tf = model_tf(**encoded)
    end_time = time.perf_counter()
    print(f"{model_run_time}次推理时间: {end_time - start_time:.4f} 秒")
    
    # ========== 计算准确性比较 ==========
    print("\n" + "=" * 60)
    print("余弦相似度 (自定义模型 vs Transformers)")
    print("=" * 60)
    # 确保两个tensor使用相同的dtype进行计算
    embedding_custom = embedding_custom.to(dtype=embedding_tf.dtype)
    sim = torch.matmul(embedding_custom, embedding_tf.T).item()
    difference = 1 - sim
    print(f"相似度: {sim:.6f} (差异: {difference:.6f})")


if __name__ == "__main__":
    main()
