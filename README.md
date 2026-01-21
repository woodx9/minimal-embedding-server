# Minimal Embedding Server - 高性能多进程推理框架

一个基于多进程架构的高性能 Embedding 服务器，专门为解决 CPU tokenizer 瓶颈和最大化 GPU 利用率而设计。

**核心特性**：
-  支持 Flash Attention 和 FlashInfer 加速注意力计算
-  多进程架构完全突破 Python GIL 限制
-  专为 Embedding 场景优化的轻量级推理引擎
-  智能动态 batch 聚合，最大化 GPU 吞吐

## 性能表现

**压测对比（10 并发客户端，每批 20 个文本，每文本 1000 tokens）**

| 框架 | QPS | 性能提升 |
|------|-----|----------|
| vLLM | 1.04 | 基准 |
| **本框架** | **1.10** | **快 5.8%** |

**测试命令**：
```bash
cd benchmark
python3 stress_test.py \
    --concurrent-clients 10 \
    --batch-size 20 \
    --token-length 1000 \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-Embedding-0.6B
```

> 测试脚本和部署脚本位于 `benchmark/` 目录下：
> - `stress_test.py` - 性能压测脚本
> - `vllm.sh` - vLLM 部署脚本
> - `compare_transformers.py` - 精度对比脚本

**为什么更快？**

本框架专为 Embedding 场景设计，更加精简高效：
- 去除了 vLLM 中复杂的通用 LLM 推理逻辑（采样、解码、KV Cache 等）
- 针对 Embedding 任务优化的轻量级架构
- 多进程隔离，CPU tokenizer 和 GPU 推理完全并行
- 智能动态 batch 聚合，最大化 GPU 吞吐
- 向量化后处理，单次 GPU 同步代替多次同步

---

## 核心设计目标

在传统的单进程推理服务中，经常遇到以下问题：
- **CPU 利用率暴涨至 400%**（多线程 tokenizer 受 GIL 限制）
- **GPU 利用率下降**（tokenizer 阻塞导致 GPU 饥饿）
- **推理延迟增加**（CPU 和 GPU 无法并行工作）

本框架通过**多进程架构**彻底解决这些问题，实现 CPU 和 GPU 的完全并行。

---

## 架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                        (uvicorn + asyncio)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engine (主协调器)                           │
│  - 创建 MPQueue 进行进程间通信                                    │
│  - 启动 Tokenizer Manager 进程                                   │
│  - 启动 GPU Worker 进程                                          │
│  - 结果分发线程（Result Dispatcher）                              │
└──────────┬──────────────────────────────────────┬───────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│  Tokenizer Manager 进程   │      │      GPU Worker 进程          │
│  (CPU 密集型)             │      │      (GPU 密集型)             │
├──────────────────────────┤      ├──────────────────────────────┤
│ • 10 个 Tokenizer 线程    │      │ • 模型加载到 GPU              │
│ • 1 个 Batch Prepare 线程 │──────▶│ • 1 个 Inference 线程        │
│ • CPU 上完成所有 tokenize │      │ • 4 个 Callback 线程          │
│ • 动态 batch 聚合         │      │ • 向量化后处理                │
│ • numpy 序列化传输        │      │ • 批量归一化                  │
└──────────────────────────┘      └──────────────────────────────┘
```

---

## 三大核心优化

### 1. 多进程隔离：彻底突破 GIL

**问题：** Python GIL 导致多线程 tokenizer 无法真正并行，CPU 飙升但效率低下。

**解决方案：**
```python
# Tokenizer Manager - 独立进程
_prepare_process = Process(target=tokenizer_manager_main)

# GPU Worker - 独立进程  
_inference_process = Process(target=gpu_worker_main)
```

**效果：**
- Tokenizer 和 GPU 推理在**不同进程**中运行
- 完全避开 GIL 限制
- CPU 和 GPU **真正并行工作**

---

### 2. 智能 Batch 聚合：最大化 GPU 吞吐

**核心策略：**
```python
# 动态等待策略
max_wait_rounds = 1 if ready_queue.qsize() < 3 else 10

while total_tokens < max_tokens_per_batch:
    # 1. 快速收集队列中所有等待请求
    while not tokenized_queue.empty():
        batch.append(tokenized_queue.get_nowait())
    
    # 2. 根据 GPU 负载动态调整等待时间
    # GPU 空闲时快速发送，GPU 忙时激进聚合
```

**优势：**
- **GPU 空闲时**：立即发送小 batch，降低延迟
- **GPU 繁忙时**：等待更多请求，聚合成大 batch
- **Token 上限**：`max_tokens_per_batch = 120,000`，充分利用 GPU 显存

---

### 3. 向量化后处理：消除 GPU 同步开销

**传统方法的问题：**
```python
#  旧代码：多次 GPU 同步，性能差
for seq_len in seq_lengths:
    embedding = outputs[start:start+seq_len][-1]  # GPU 操作
    embedding = F.normalize(embedding)            # GPU 操作
    embeddings.append(embedding.cpu())            # GPU→CPU 同步
    start += seq_len
```

**优化后的向量化处理：**
```python
# 新代码：单次 GPU 同步，性能提升 10 倍
# 1. 预计算所有 last token 索引（CPU 上完成）
last_token_indices = [idx + seq_len - 1 for idx, seq_len in ...]

# 2. 一次性提取所有 embeddings（GPU 向量化操作）
last_token_indices_tensor = torch.tensor(last_token_indices, device='cuda')
all_embeddings = outputs[last_token_indices_tensor]  # [N, hidden_dim]

# 3. 批量归一化（GPU 向量化操作）
all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

# 4. 单次转 CPU（只有一次 GPU 同步！）
all_embeddings_cpu = all_embeddings.cpu()
```

---

## 数据流详解

### 请求处理全流程

```
1. 用户请求
   POST /v1/embeddings {"input": ["text1", "text2"]}
                │
                ▼
2. Engine.v1_embeddings (主进程)
   - 生成 UUID 作为 future_id
   - 存储到 _future_map: {uuid: (future, num_texts)}
   - 发送到 raw_request_queue: (texts, future_id)
                │
                ▼
3. Tokenizer Manager 进程
   ┌─────────────────────────────────────┐
   │ Tokenizer 线程池 (10 线程)           │
   │  - 并行 tokenize 多个请求            │
   │  - CPU 密集型操作，完全并行          │
   │  → tokenized_queue                  │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │ Batch Prepare 线程 (1 线程)          │
   │  - 激进聚合：收集多个请求             │
   │  - 动态等待策略                      │
   │  - 预计算 last_token_indices         │
   │  - 转 numpy 准备跨进程传输           │
   │  → ready_inference_queue            │
   └──────────────┬──────────────────────┘
                  │
4. GPU Worker 进程
   ┌──────────────▼──────────────────────┐
   │ Inference 线程 (1 线程)              │
   │  - numpy → tensor → GPU             │
   │  - 模型推理                          │
   │  - 向量化后处理 (单次 GPU 同步)       │
   │  → callback_queue                   │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │ Callback 线程池 (4 线程)             │
   │  - 异步发送结果                      │
   │  → result_queue (跨进程)            │
   └──────────────┬──────────────────────┘
                  │
5. Engine 结果分发线程 (主进程)
   - 从 result_queue 接收结果
   - 根据 num_texts 正确分割 embeddings
   - 通过 future.set_result() 返回给对应请求
                │
                ▼
6. 返回给用户
   {"data": [{"embedding": [...], "index": 0}, ...]}
```

---

## 关键技术细节

### 1. 无锁并发设计

```python
# UUID 保证唯一性，无需锁保护
future_id = str(uuid.uuid4())

# GIL 保证单个赋值的原子性
self._future_map[future_id] = (future, len(input))

# 只在组合操作（check + read + delete）时加锁
with self._future_lock:
    if future_id in self._future_map:
        future, num_texts = self._future_map[future_id]
        del self._future_map[future_id]
```

### 2. 跨进程通信优化

```python
# 使用 multiprocessing.Queue 进行进程间通信
raw_request_queue = MPQueue(maxsize=1000)
ready_inference_queue = MPQueue(maxsize=100)
result_queue = MPQueue(maxsize=1000)

# tensor 转 numpy 方便序列化传输
merged_input_ids.numpy()  # 在 Tokenizer Manager
torch.from_numpy(input_ids_np).to('cuda')  # 在 GPU Worker
```

### 3. 动态 Batch 聚合算法

```python
# 根据 GPU 队列深度动态调整等待策略
if ready_queue.qsize() < 3:
    max_wait_rounds = 1  # GPU 空闲，快速发送
else:
    max_wait_rounds = 10  # GPU 繁忙，激进聚合

# 持续收集直到 token 上限或超时
while total_tokens < 120000 and wait_rounds < max_wait_rounds:
    # 快速收集 + 超时等待
```

---


## 安装指南

### 快速安装

#### 1. 创建 Conda 环境
```bash
conda create -n flashattn5 python=3.10
conda activate flashattn5
```

#### 2. 运行安装脚本
```bash
./build.sh
```

安装脚本会自动完成：
- 安装 PyTorch 2.4.1 (CUDA 12.1)
- 下载并安装 Flash Attention 和 FlashInfer
- 安装所有依赖包

#### 3. 启动服务
```bash
uvicorn openai_server.fast_api:app --host 0.0.0.0 --port 8000
```

---

### 手动安装（如果脚本失败）

#### 步骤 1: 安装 PyTorch
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 步骤 2: 下载 Flash Attention
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 步骤 3: 安装 Flash Attention
```bash
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 步骤 4: 下载并安装 FlashInfer（可选）

根据你的 CUDA 版本选择对应的包：


**CUDA 12.9:**
```bash
wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.1/flashinfer_python-0.6.1-py3-none-any.whl
wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.1/flashinfer_cubin-0.6.1-py3-none-any.whl
wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.1/flashinfer_jit_cache-0.6.1+cu129-cp39-abi3-manylinux_2_28_x86_64.whl

pip install flashinfer_python-0.6.1-py3-none-any.whl
pip install flashinfer_cubin-0.6.1-py3-none-any.whl
pip install flashinfer_jit_cache-0.6.1+cu129-cp39-abi3-manylinux_2_28_x86_64.whl
```

#### 步骤 5: 安装其他依赖
```bash
pip install -r requirements.txt
```

---

### 版本说明

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 | 必须 |
| PyTorch | 2.4.1 | CUDA 12.1 |
| Flash Attention | 2.8.3 | 支持 |
| FlashInfer | 0.6.1 | 支持 |
| Transformers | 最新兼容版 | 兼容 torch 2.4 |
| FastAPI | 最新版 | Web 框架 |
| Uvicorn | 最新版 | ASGI 服务器 |

---

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
python -c "import flashinfer; print('FlashInfer: 0.6.1')"
python -c "import transformers; print('Transformers OK')"
python -c "import fastapi; print('FastAPI OK')"
```

---

### 故障排查

#### 问题 1: Flash Attention 下载失败
**解决方案**: 手动从 GitHub Release 下载
```bash
# 使用代理或镜像
export https_proxy=http://your-proxy:port
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 问题 2: CUDA 版本不匹配
**解决方案**: 检查 CUDA 版本
```bash
nvidia-smi  # 查看系统 CUDA 版本
nvcc --version  # 查看 CUDA toolkit 版本
```

如果系统 CUDA 不是 12.1，需要：
- 安装对应版本的 PyTorch
- 下载对应版本的 Flash Attention

#### 问题 3: Python 版本错误
**解决方案**: 确保使用 Python 3.10
```bash
python --version  # 必须是 3.10.x
conda create -n flashattn5 python=3.10  # 重新创建环境
```

---

### 其他 Flash Attention 版本

如果需要其他版本，访问：
https://github.com/Dao-AILab/flash-attention/releases

选择适合你的：
- CUDA 版本 (cu118, cu121, etc.)
- PyTorch 版本 (torch2.0, torch2.1, torch2.2, etc.)
- Python 版本 (cp310, cp311, etc.)

---

## 快速开始

### 启动服务
```bash
uvicorn openai_server.fast_api:app --host 0.0.0.0 --port 8000
```

### 测试请求
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "How are you"], "model": "Qwen3-Embedding-0.6B"}'
```

---

## 项目结构

```
minimal-embedding-server/
├── core/
│   ├── engine.py              # 主协调器（Engine 类）
│   ├── tokenizer_manager.py   # CPU 密集型 Tokenizer 进程
│   ├── gpu_worker.py          # GPU 密集型推理进程
│   └── scheduler.py
├── models/
│   └── qwen3.py               # Qwen3 模型定义
├── layers/
│   ├── attention.py
│   ├── linear.py
│   └── rotary_embedding.py
├── ultils/
│   ├── loader.py              # 模型加载
│   └── pool.py
├── schemas/
│   └── http.py                # API 数据模型
├── open_server/
│   └── fast_api.py            # FastAPI 服务器
└── test/
    └── test_accuracy.py
```

---

## 设计理念

### 为什么选择多进程而不是多线程？

1. **GIL 限制**：Python 的 GIL 导致多线程无法真正并行执行 CPU 密集型任务
2. **资源隔离**：进程级别隔离，tokenizer 不会影响 GPU 推理
3. **可扩展性**：未来可以轻松扩展到多机器分布式架构

### 为什么需要向量化后处理？

传统的循环式 embedding 提取每次都会触发 GPU 同步，而 GPU 同步是非常昂贵的操作（~1ms/次）。通过向量化：
- 将 N 次 GPU 同步减少到 1 次
- 利用 PyTorch 的向量化算子（更快）
- 后处理时间从 100ms 降低到 5ms

### 为什么使用动态 Batch 聚合？

- **低延迟优先**：请求少时快速响应
- **高吞吐优先**：请求多时聚合成大 batch
- **自适应调整**：根据实时负载动态调整策略

---

## 监控日志示例

```
[TokenizerManager] Loaded tokenizer, starting 10 tokenizer threads...
[GPUWorker] Loading model on cuda:1...
[GPUWorker] Model loaded successfully on cuda:1
[Engine] All processes started successfully

[BatchPrepare-0] Batch:5 TotalTokens:8192 AvgTokens:1638 Collect:45.23ms Merge:2.15ms Total:47.38ms ready:2
[Inference] Requests:5 TotalTokens:8192 Wait:1.23ms Transfer:3.45ms Infer:15.67ms Post:4.89ms Callback:0.12ms Total:25.36ms ready:1
```

---

## 注意事项

1. **内存管理**：每个进程独立加载模型，需要足够的 GPU 显存
2. **进程启动时间**：首次启动需要加载模型（~5-10 秒）
3. **队列大小**：根据实际负载调整 `MPQueue(maxsize=...)`

---

## 许可证

MIT License

---

## 致谢

本项目专为解决实际生产环境中的 GPU 利用率问题而设计，采用了多项业界最佳实践。

