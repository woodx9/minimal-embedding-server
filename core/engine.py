import torch
from schemas import http
from schemas.config import MSEConfig
import threading
import asyncio
from queue import Empty
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue
import uuid

# 设置多进程启动方法为 spawn（CUDA 要求）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 导入新的进程模块
from core.tokenizer_manager import TokenizerManager
from core.gpu_worker import GPUWorker


class Engine:
     _instance = None
     _lock = threading.Lock()
     _has_init = False
     _model_name = ""
     _device = None
     _attn_backend = ""
     _tensor_parallel_size = 1
     
     # 多进程架构
     _prepare_processes = None        # Prepare 进程
     _inference_process = []      # Inference 进程
     _inference_events = []         # Inference 事件列表
     _raw_request_queue = None      # 原始请求队列（多进程）
     _ready_inference_queue = None  # 准备好的batch队列（多进程）
     _result_queue = None           # 结果队列（多进程）
     _future_map = {}               # future_id -> (future, num_texts)
     _future_lock = threading.Lock()  # 保护 _future_map 的读写
     _result_dispatcher_thread = None  # 结果分发线程
     
     # 配置
     _num_tokenize_threads = 5     # tokenizer 线程数
     _max_batch_size = 64
     _batch_timeout = 0.05
     _max_tokens_per_batch = 120000
     _enable_monitoring = True

     def __new__(cls, *args, **kwargs):
          if cls._instance is None:
               cls._instance = super().__new__(cls)
          return cls._instance

     def __init__(self, attn_backend="flash_attention", tensor_parallel_size=1):
          if self._has_init:
               return
          self._has_init = True
          self._model_name = "Qwen/Qwen3-Embedding-0.6B"
          self._attn_backend = attn_backend  
          self._tensor_parallel_size = tensor_parallel_size 

          # 创建多进程队列
          self._raw_request_queue = MPQueue(maxsize=1000)
          self._ready_inference_queue = MPQueue(maxsize=100)
          self._result_queue = MPQueue(maxsize=1000)
          
          print("[Engine] Starting Tokenizer Manager Process...")
          # 启动 Tokenizer Manager 进程（CPU密集型）
          mse_config = MSEConfig(
               attn_backend=self._attn_backend,
               model_name=self._model_name,
               max_tokens_per_batch=self._max_tokens_per_batch,
               enable_monitoring=self._enable_monitoring,
          )
          self._prepare_process = Process(
               target=TokenizerManager,
               args=(
                    self._raw_request_queue,
                    self._ready_inference_queue,
                    self._num_tokenize_threads,
                    self._batch_timeout,
                    mse_config,
               ),
               name="TokenizerManager",
          )
          self._prepare_process.start()
         
          print(f"[Engine] Starting GPU Worker Process (attn_backend={self._attn_backend})...")

          ctx = mp.get_context("spawn")
          for i in range(1, self._tensor_parallel_size):
               print(f"[Engine] Starting GPU Worker Process Rank {i}...")
               event = ctx.Event()
               process = ctx.Process(
                    target=GPUWorker,
                    args=(
                         i,
                         self._tensor_parallel_size,
                         event,
                         self._ready_inference_queue,
                         self._result_queue,
                         mse_config,
                    )
               )
               process.start()
               self._inference_process.append(process)
               self._inference_events.append(event)
          
     
          event = ctx.Event()
          process = ctx.Process(
               target=GPUWorker,
               args=(
                    0,
                    self._tensor_parallel_size,
                    self._inference_events,
                    self._ready_inference_queue,
                    self._result_queue,
                    mse_config,
               )
          )
          process.start()
          self._inference_process.append(process)

          # 启动结果分发线程
          self._result_dispatcher_thread = threading.Thread(
              target=self._result_dispatcher_worker,
              name="ResultDispatcher"
          )
          self._result_dispatcher_thread.start()
          
          print("[Engine] All processes started successfully")

     def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self._prepare_processes:
            p.join()

     
     def _result_dispatcher_worker(self):
          """结果分发线程 - 从 result_queue 取结果并分发给对应的 future"""
          while True:
               try:
                    # 从结果队列获取结果
                    result = self._result_queue.get(timeout=0.1)
                    if result is None:  # 终止信号
                        break
                    
                    all_embeddings_list, all_seq_lengths, future_ids = result
                    
                    # 正确分发：根据每个请求包含的texts数量来分割
                    embedding_idx = 0  # 当前处理的embedding索引
                    
                    for future_id in future_ids:
                        with self._future_lock:
                            if future_id not in self._future_map:
                                print(f"[ResultDispatcher Warning] future_id {future_id} not found in map")
                                continue
                            
                            future, num_texts = self._future_map[future_id]
                            
                            # 提取这个请求对应的 embeddings 和 seq_lengths
                            request_embeddings = all_embeddings_list[embedding_idx:embedding_idx + num_texts]
                            request_seq_lengths = all_seq_lengths[embedding_idx:embedding_idx + num_texts]
                            embedding_idx += num_texts
                            
                            # 设置结果（线程安全地回调到asyncio）
                            future.get_loop().call_soon_threadsafe(
                                future.set_result, 
                                (request_embeddings, request_seq_lengths)
                            )
                            
                            # 清理
                            del self._future_map[future_id]
                    
               except Empty:
                    continue
               except Exception as e:
                    print(f"[ResultDispatcher Error] {e}")
                    import traceback
                    traceback.print_exc()
                    continue
     
     async def v1_embeddings(self, request: http.EmbeddingRequest):
          input = request.input
          
          # 确保 input 是列表
          if isinstance(input, str):
               input = [input]
          
          # 创建 Future
          loop = asyncio.get_event_loop()
          future = loop.create_future()
          
          # 生成唯一的 future_id（UUID 保证唯一，无需锁）
          future_id = str(uuid.uuid4())
          
          # 存储 future（GIL 保证单个赋值的原子性，无需额外加锁）
          self._future_map[future_id] = (future, len(input))
          
          # 发送到 Tokenizer Manager 进程
          await asyncio.to_thread(self._raw_request_queue.put, (input, future_id))
          
          # 等待结果
          embeddings_list, seq_lengths = await future
          
          # 构建响应数据
          data = [
               http.EmbeddingData(
                    object="embedding",
                    embedding=emb,
                    index=idx,
               )
               for idx, emb in enumerate(embeddings_list)
          ]
          
          total_tokens = sum(seq_lengths)
          
          return http.EmbeddingResponse(
               object="list",
               data=data,
               model=request.model,
               usage=http.EmbeddingUsage(
                    prompt_tokens=total_tokens,
                    total_tokens=total_tokens
               )
          )
