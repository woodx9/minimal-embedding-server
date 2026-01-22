# Minimal Embedding Server (MES) - 高性能多进程推理框架

一个基于多进程架构的高性能 Embedding 服务器，专门为解决 CPU tokenizer 瓶颈和最大化 GPU 利用率而设计。

**核心特性**：
-  支持 Flash Attention 和 FlashInfer 加速注意力计算
-  多进程架构完全突破 Python GIL 限制
-  专为 Embedding 场景优化的轻量级推理引擎
-  智能动态 batch 聚合，最大化 GPU 吞吐

> **注意**：当前版本仅支持 **Qwen3 Embedding** 系列模型（如 `Qwen/Qwen3-Embedding-0.6B`）

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/woodx9/minimal-embedding-server.git
cd minimal-embedding-server

pip install -e .
```

> **注意**：安装过程会自动下载并安装：
> - PyTorch 2.4.1 (CUDA 12.1)
> - Flash Attention 2.8.3
> - FlashInfer 0.6.1
> - 其他依赖包

### 使用方式

#### 方式 1: 命令行启动（推荐）

```bash
# 基本启动（默认使用 flash_attention）
mes

# 指定端口和注意力后端
mes --port 8000 --attn-backend flash_attention

# 查看更多选项
mes --help
```

**命令行参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务器监听地址 |
| `--port` | `8000` | 服务器监听端口 |
| `--attn-backend` | `flash_attention` | 注意力后端（flash_attention/flash_infer） |


### 测试 API

```bash
# 健康检查
curl http://localhost:8000/health

# 获取 embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["你好，世界！", "Hello, world!"]
  }'

```

## 性能表现

**压测对比（10 并发客户端，每批 20 个文本，每文本 1000 tokens）**

| 框架 | QPS | 性能提升 |
|------|-----|----------|
| vLLM | 1.04 | 基准 |
| **本框架** | **1.10** | **快 5.8%** |

**测试命令**：
```bash

python3  benchmark/stress_test.py \
    --concurrent-clients 10 \
    --batch-size 20 \
    --token-length 1000 \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-Embedding-0.6B
```

> 测试脚本和部署脚本位于 `benchmark/` 目录下：
> - `stress_test.py` - 性能压测脚本
> - `vllm.sh` - vLLM 部署脚本
> - `compare_transformers.py` - 对比 transformer 速度脚本

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
│  - 创建 MPQueue 进行进程间通信                                     │
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
│ • 10 个 Tokenizer 线程    │      │ • 模型加载到 GPU               │
│ • 1 个 Batch Prepare 线程 │──────▶│ • 1 个 Inference 线程        │
│ • CPU 上完成所有 tokenize  │      │ • 4 个 Callback 线程          │
│ • 动态 batch 聚合          │      │ • 向量化后处理                │
│ • numpy 序列化传输         │      │ • 批量归一化                  │
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



## 许可证

MIT License

---

## 致谢

本项目专为解决实际生产环境中的 GPU 利用率问题而设计，采用了多项业界最佳实践。

