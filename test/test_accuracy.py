from huggingface_hub import snapshot_download
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch.nn.functional as F

from models.qwen3 import Qwen3ForCausalLM
from schemas.config import MSEConfig
from ultils.loader import load_model
from ultils.pool import last_token_pool


def test_accuracy_bf16():
     model_name = "Qwen/Qwen3-Embedding-0.6B"
     config = AutoConfig.from_pretrained(model_name)
     mse_config = MSEConfig(attn_backend="flash_infer")
     model_runner = Qwen3ForCausalLM(config, mse_config)

     model_path = snapshot_download(model_name)
     load_model(model_runner, model_path)
     
     # 将模型移到 GPU 并转换为 bf16
     model_runner = model_runner.cuda().to(torch.bfloat16)
     model_runner.eval()
     
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     
     # 测试文本
     text = "Hello, how are you?" 
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
     input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
     positions = torch.tensor(positions, dtype=torch.int64, device="cuda")

     with torch.no_grad():
          outputs = model_runner(input_ids=input_ids, positions=positions)
          embedding_custom = last_token_pool(outputs, seq_len)
          embedding_custom = embedding_custom.unsqueeze(0) 
          embedding_custom = F.normalize(embedding_custom, p=2, dim=1)

     # ========== 使用 Transformers 库 ==========
     print("\n" + "=" * 60)
     print("使用 Transformers 库")
     print("=" * 60)
     
     model_tf = AutoModel.from_pretrained(
          model_name,
          dtype=torch.bfloat16,
     ).cuda()
     model_tf.eval()
     
     # 复用 tokenizer 和 text
     encoded = tokenizer(text, return_tensors="pt")
     encoded = {k: v.cuda() for k, v in encoded.items()}

     with torch.no_grad():
          outputs_tf = model_tf(**encoded)
          embedding_tf = outputs_tf.last_hidden_state[:, -1, :]
          embedding_tf = F.normalize(embedding_tf, p=2, dim=1)

     
     # 计算两个嵌入之间的余弦相似度
     print("\n" + "=" * 60)
     print("余弦相似度 (自定义模型 vs Transformers)")
     print("=" * 60)
     sim = torch.matmul(embedding_custom, embedding_tf.T).item()
     difference = 1 - sim
     max_tolerance = 1e-5
     assert difference < max_tolerance, f"模型输出差异过大: 余弦相似度 {sim:.6f}，差异 {difference:.6f} 超过阈值 {max_tolerance}。自定义模型与 Transformers 参考实现不一致。"
     print(f"相似度: {sim:.6f} (差异: {difference:.6f}, 阈值: {max_tolerance})")
