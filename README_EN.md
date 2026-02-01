# Minimal Embedding Server (MES) - High-Performance Multi-Process Inference Framework

English | [中文](README.md)

A high-performance Embedding server based on multi-process architecture, specifically designed to solve CPU tokenizer bottlenecks and maximize GPU utilization.

**Core Features**:
- Support for Flash Attention and FlashInfer accelerated attention computation
- Support for tensor parallel
- Multi-process architecture completely breaks through Python GIL limitations
- Lightweight inference engine optimized for Embedding scenarios
- Intelligent dynamic batch aggregation to maximize GPU throughput
- Automatic model architecture recognition, supporting Qwen2 and Qwen3 series models

> **Supported Model Architectures**:
> - **Qwen2ForCausalLM**: Qwen2 series models (e.g., `Qwen/Qwen2-0.5B`, `Qwen/Qwen2-1.5B`, etc.)
> - **Qwen3ForCausalLM**: Qwen3 series models (e.g., `Qwen/Qwen3-Embedding-0.6B`, etc.)
> 
> The system automatically selects the corresponding model implementation based on the `architectures` field in the model configuration

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/woodx9/minimal-embedding-server.git
cd minimal-embedding-server

pip install -e .
```

> **Note**: The installation process will automatically download and install:
> - PyTorch 2.9.1 (CUDA 12.8)
> - SGL-Kernel 0.3.21
> - FlashInfer 0.6.2
> - Other dependencies

### Usage

#### Method 1: Command Line Startup (Recommended)

```bash
# Start Qwen3 Embedding model
mes --model "Qwen/Qwen3-Embedding-0.6B"

# Start Qwen2 model
mes --model "Qwen/Qwen2-0.5B"

# Specify port and attention backend
mes --model "Qwen/Qwen3-Embedding-0.6B" --port 8000 --attn-backend flash_attn

# Use different data types
mes --model "Qwen/Qwen2-1.5B" --dtype bfloat16

# Multi-GPU parallel inference
mes --model "Qwen/Qwen3-Embedding-0.6B" --tensor_parallel_size 2 --dtype auto

# View more options
mes --help
```

**Command Line Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | **Required** | **Model name or path (required parameter)** |
| `--host` | `0.0.0.0` | Server listening address |
| `--port` | `8000` | Server listening port |
| `--attn-backend` | `flash_attn` | Attention backend (flash_attn/flash_infer) |
| `--tensor_parallel_size` | `1` | Tensor parallel size |
| `--dtype` | `auto` | Model data type (auto/float32/float16/bfloat16) |

**Supported Model Examples:**

| Model Architecture | Model Name Example | Description |
|-------------------|-------------------|-------------|
| Qwen2ForCausalLM | `Qwen/Qwen2-0.5B` | Qwen2 series base model |
| Qwen2ForCausalLM | `Qwen/Qwen2-1.5B` | Qwen2 series large model |
| Qwen3ForCausalLM | `Qwen/Qwen3-Embedding-0.6B` | Qwen3 dedicated Embedding model |

> The system automatically selects the corresponding implementation based on the `architectures` field in the model configuration, no need to manually specify the model type.

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Get embeddings
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": ["Hello, world!", "How are you?"]
  }'
```

## Performance

**Stress Test Comparison (10 concurrent clients, 20 texts per batch, 1000 tokens per text)**

| Framework | QPS | Performance Improvement |
|-----------|-----|------------------------|
| vLLM | 1.04 | Baseline |
| **This Framework** | **1.10** | **5.8% faster** |

**Test Command**:
```bash
python3 benchmark/stress_test.py \
    --concurrent-clients 10 \
    --batch-size 20 \
    --token-length 1000 \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-Embedding-0.6B
```

> Test scripts and deployment scripts are located in the `benchmark/` directory:
> - `stress_test.py` - Performance stress test script
> - `vllm.sh` - vLLM deployment script
> - `compare_transformers.py` - Compare transformer speed script

**Why is it faster?**

This framework is specifically designed for Embedding scenarios, making it more streamlined and efficient:
- Removes complex general LLM inference logic from vLLM (sampling, decoding, KV Cache, etc.)
- Lightweight architecture optimized for Embedding tasks
- Multi-process isolation, CPU tokenizer and GPU inference completely parallel
- Intelligent dynamic batch aggregation to maximize GPU throughput
- Vectorized post-processing, single GPU synchronization instead of multiple synchronizations

---

## Core Design Goals

In traditional single-process inference services, the following problems are often encountered:
- **CPU utilization spikes to 400%** (multi-threaded tokenizer limited by GIL)
- **GPU utilization drops** (tokenizer blocking causes GPU starvation)
- **Inference latency increases** (CPU and GPU cannot work in parallel)

This framework completely solves these problems through **multi-process architecture**, achieving complete parallelism between CPU and GPU.

---

## Architecture Design

### Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                        (uvicorn + asyncio)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engine (Main Coordinator)                   │
│  - Create MPQueue for inter-process communication                │
│  - Start Tokenizer Manager process                               │
│  - Start GPU Worker process                                      │
│  - Result Dispatcher thread                                      │
└──────────┬──────────────────────────────────────┬───────────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│  Tokenizer Manager Process│      │      GPU Worker Process       │
│  (CPU Intensive)          │      │      (GPU Intensive)          │
├──────────────────────────┤      ├──────────────────────────────┤
│ • 10 Tokenizer threads    │      │ • Model loading to GPU        │
│ • 1 Batch Prepare thread  │──────▶│ • 1 Inference thread         │
│ • Complete tokenize on CPU│      │ • 4 Callback threads          │
│ • Dynamic batch aggregation│      │ • Vectorized post-processing │
│ • numpy serialization     │      │ • Batch normalization        │
└──────────────────────────┘      └──────────────────────────────┘
```

---

## Three Core Optimizations

### 1. Multi-Process Isolation: Complete GIL Breakthrough

**Problem:** Python GIL causes multi-threaded tokenizer to not truly parallel, CPU spikes but low efficiency.

**Solution:**
```python
# Tokenizer Manager - Independent process
_prepare_process = Process(target=tokenizer_manager_main)

# GPU Worker - Independent process  
_inference_process = Process(target=gpu_worker_main)
```

**Effect:**
- Tokenizer and GPU inference run in **different processes**
- Completely avoid GIL limitations
- CPU and GPU **truly work in parallel**

---

### 2. Intelligent Batch Aggregation: Maximize GPU Throughput

**Core Strategy:**
```python
# Dynamic waiting strategy
max_wait_rounds = 1 if ready_queue.qsize() < 3 else 10

while total_tokens < max_tokens_per_batch:
    # 1. Quickly collect all waiting requests in queue
    while not tokenized_queue.empty():
        batch.append(tokenized_queue.get_nowait())
    
    # 2. Dynamically adjust waiting time based on GPU load
    # Send quickly when GPU is idle, aggressive aggregation when GPU is busy
```

**Advantages:**
- **When GPU is idle**: Send small batches immediately, reduce latency
- **When GPU is busy**: Wait for more requests, aggregate into large batches
- **Token limit**: `max_tokens_per_batch = 120,000`, fully utilize GPU memory

---

### 3. Vectorized Post-Processing: Eliminate GPU Synchronization Overhead

**Traditional Method Problems:**
```python
# Old code: Multiple GPU synchronizations, poor performance
for seq_len in seq_lengths:
    embedding = outputs[start:start+seq_len][-1]  # GPU operation
    embedding = F.normalize(embedding)            # GPU operation
    embeddings.append(embedding.cpu())            # GPU→CPU synchronization
    start += seq_len
```

**Optimized Vectorized Processing:**
```python
# New code: Single GPU synchronization, 10x performance improvement
# 1. Pre-calculate all last token indices (completed on CPU)
last_token_indices = [idx + seq_len - 1 for idx, seq_len in ...]

# 2. Extract all embeddings at once (GPU vectorized operation)
last_token_indices_tensor = torch.tensor(last_token_indices, device='cuda')
all_embeddings = outputs[last_token_indices_tensor]  # [N, hidden_dim]

# 3. Batch normalization (GPU vectorized operation)
all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

# 4. Single CPU transfer (only one GPU synchronization!)
all_embeddings_cpu = all_embeddings.cpu()
```

---

## Data Flow Details

### Complete Request Processing Flow

```
1. User Request
   POST /v1/embeddings {"input": ["text1", "text2"]}
                │
                ▼
2. Engine.v1_embeddings (main process)
   - Generate UUID as future_id
   - Store in _future_map: {uuid: (future, num_texts)}
   - Send to raw_request_queue: (texts, future_id)
                │
                ▼
3. Tokenizer Manager Process
   ┌─────────────────────────────────────┐
   │ Tokenizer Thread Pool (10 threads)  │
   │  - Parallel tokenize multiple requests│
   │  - CPU intensive operations, fully parallel│
   │  → tokenized_queue                  │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │ Batch Prepare Thread (1 thread)     │
   │  - Aggressive aggregation: collect multiple requests│
   │  - Dynamic waiting strategy          │
   │  - Pre-calculate last_token_indices  │
   │  - Convert to numpy for cross-process transfer│
   │  → ready_inference_queue            │
   └──────────────┬──────────────────────┘
                  │
4. GPU Worker Process
   ┌──────────────▼──────────────────────┐
   │ Inference Thread (1 thread)         │
   │  - numpy → tensor → GPU             │
   │  - Model inference                  │
   │  - Vectorized post-processing (single GPU sync)│
   │  → callback_queue                   │
   └──────────────┬──────────────────────┘
                  │
   ┌──────────────▼──────────────────────┐
   │ Callback Thread Pool (4 threads)    │
   │  - Asynchronous result sending      │
   │  → result_queue (cross-process)     │
   └──────────────┬──────────────────────┘
                  │
5. Engine Result Dispatcher Thread (main process)
   - Receive results from result_queue
   - Correctly split embeddings based on num_texts
   - Return to corresponding request via future.set_result()
                │
                ▼
6. Return to User
   {"data": [{"embedding": [...], "index": 0}, ...]}
```

---

## Key Technical Details

### 1. Lock-Free Concurrent Design

```python
# UUID ensures uniqueness, no lock protection needed
future_id = str(uuid.uuid4())

# GIL ensures atomicity of single assignment
self._future_map[future_id] = (future, len(input))

# Only lock during composite operations (check + read + delete)
with self._future_lock:
    if future_id in self._future_map:
        future, num_texts = self._future_map[future_id]
        del self._future_map[future_id]
```

### 2. Cross-Process Communication Optimization

```python
# Use multiprocessing.Queue for inter-process communication
raw_request_queue = MPQueue(maxsize=1000)
ready_inference_queue = MPQueue(maxsize=100)
result_queue = MPQueue(maxsize=1000)

# tensor to numpy for easy serialization
merged_input_ids.numpy()  # In Tokenizer Manager
torch.from_numpy(input_ids_np).to('cuda')  # In GPU Worker
```

### 3. Dynamic Batch Aggregation Algorithm

```python
# Dynamically adjust waiting strategy based on GPU queue depth
if ready_queue.qsize() < 3:
    max_wait_rounds = 1  # GPU idle, send quickly
else:
    max_wait_rounds = 10  # GPU busy, aggressive aggregation

# Continue collecting until token limit or timeout
while total_tokens < 120000 and wait_rounds < max_wait_rounds:
    # Quick collection + timeout waiting
```

---

## License

MIT License

---

## Acknowledgments

This project is specifically designed to solve GPU utilization problems in actual production environments, adopting multiple industry best practices.