import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch.nn.functional as F
import asyncio

from core.engine import Engine


def test_qwen2_accuracy():
    model_name = "Qwen/Qwen2-0.5B"
    mes_engine = Engine(model_name=model_name, attn_backend="flash_attn", tensor_parallel_size=1)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    text = "Hello, how are you?"
    print(f"测试文本: {text}")
    print(f"测试文本长度: {len(text)} 字符")
    
    print("\n" + "=" * 60)
    print("使用自定义模型 (MES Engine - Qwen2)")
    print("=" * 60)
    
    with torch.no_grad():
        embeddings_list, seq_lengths = asyncio.run(mes_engine.v1_embeddings(input=[text]))
        embedding_custom = torch.tensor(embeddings_list[0], device="cuda", dtype=torch.bfloat16)

    print("\n" + "=" * 60)
    print("使用 Transformers 库 (Qwen2)")
    print("=" * 60)
    
    model_tf = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).cuda()
    model_tf.eval()
    
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {k: v.cuda() for k, v in encoded.items()}

    with torch.no_grad():
        outputs_tf = model_tf(**encoded)
        embedding_tf = outputs_tf.last_hidden_state[:, -1, :]
        embedding_tf = F.normalize(embedding_tf, p=2, dim=1)

    print("\n" + "=" * 60)
    print("余弦相似度 (自定义模型 vs Transformers)")
    print("=" * 60)
    
    embedding_custom = embedding_custom.to(dtype=embedding_tf.dtype)
    sim = torch.matmul(embedding_custom, embedding_tf.T).item()
    difference = 1 - sim
    max_tolerance = 1e-5
    
    print(f"相似度: {sim:.6f} (差异: {difference:.6f}, 阈值: {max_tolerance})")
    
    assert difference < max_tolerance, f"Qwen2模型输出差异过大: 余弦相似度 {sim:.6f}，差异 {difference:.6f} 超过阈值 {max_tolerance}"
    print("Qwen2 精度测试通过!")


if __name__ == "__main__":
    test_qwen2_accuracy()