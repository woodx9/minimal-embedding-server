"""
GPU Worker - GPU密集型的推理和后处理进程
"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from models.qwen3 import Qwen3ForCausalLM
from ultils.loader import load_model
from huggingface_hub import snapshot_download
import time
import threading
from queue import Queue, Empty


def gpu_worker_main(model_name, device, ready_queue, result_queue, max_tokens_per_batch, enable_monitoring):
    """
    GPU Worker 进程入口
    负责：
    1. 加载模型到GPU
    2. GPU推理（向量化）
    3. 后处理（批量归一化）
    4. 异步回调
    """
    # 加载模型到GPU
    print(f"[GPUWorker] Loading model on {device}...")
    config = AutoConfig.from_pretrained(model_name)
    model = Qwen3ForCausalLM(config).to(device).to(torch.bfloat16)
    model.eval()
    model_path = snapshot_download(model_name)
    load_model(model, model_path)
    print(f"[GPUWorker] Model loaded successfully on {device}")
    
    # 回调队列（内部线程间通信）
    callback_queue = Queue(maxsize=1000)
    
    def callback_worker():
        """异步回调线程 - 将结果发送回主进程"""
        while True:
            try:
                embeddings_list, seq_lengths, future_ids = callback_queue.get(timeout=0.1)
                
                # 发送结果回主进程
                result_queue.put((embeddings_list, seq_lengths, future_ids))
            except Empty:
                continue
            except Exception as e:
                print(f"[Callback] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 启动回调线程池（4个线程异步处理回调）
    callback_threads = []
    for i in range(4):
        t = threading.Thread(target=callback_worker, daemon=True, name=f"Callback-{i}")
        t.start()
        callback_threads.append(t)
    
    def inference_worker():
        """推理线程 - GPU密集型操作"""
        while True:
            try:
                wait_start = time.time()
                # 获取准备好的 batch
                batch_data = ready_queue.get(timeout=0.1)
                if batch_data is None:  # 终止信号
                    break
                
                merged_input_ids_np, merged_positions_np, all_seq_lengths, last_token_indices, future_ids = batch_data
                
                wait_time = (time.time() - wait_start) * 1000
                
                # numpy -> tensor -> GPU（单次数据传输）
                transfer_start = time.time()
                merged_input_ids = torch.from_numpy(merged_input_ids_np).to(device)
                merged_positions = torch.from_numpy(merged_positions_np).to(device)
                transfer_time = (time.time() - transfer_start) * 1000
                
                total_tokens = merged_input_ids.shape[0]
                
                # GPU 推理
                inference_start = time.time()
                with torch.no_grad():
                    outputs = model(input_ids=merged_input_ids, positions=merged_positions)
                inference_time = (time.time() - inference_start) * 1000
                
                # 后处理（向量化操作，无循环）
                postprocess_start = time.time()
                
                # 1. 预计算的indices直接转GPU tensor
                last_token_indices_tensor = torch.tensor(last_token_indices, device=device)
                
                # 2. 一次性提取所有last token embeddings
                all_embeddings = outputs[last_token_indices_tensor]  # [num_seqs, hidden_dim]
                
                # 3. 批量归一化（GPU向量化操作）
                all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
                
                # 4. 一次性转CPU（单次GPU同步）
                all_embeddings_cpu = all_embeddings.cpu()
                
                # 5. 转list（在CPU上，无GPU阻塞）
                all_embeddings_list = all_embeddings_cpu.tolist()
                postprocess_time = (time.time() - postprocess_start) * 1000
                
                # 异步回调（放入回调队列，不阻塞GPU推理）
                callback_start = time.time()
                callback_queue.put((all_embeddings_list, all_seq_lengths, future_ids))
                callback_time = (time.time() - callback_start) * 1000
                
                if enable_monitoring:
                    total_time = wait_time + transfer_time + inference_time + postprocess_time + callback_time
                    num_requests = len(future_ids)
                    print(f"[Inference] Requests:{num_requests} TotalTokens:{total_tokens} "
                          f"Wait:{wait_time:.2f}ms Transfer:{transfer_time:.2f}ms "
                          f"Infer:{inference_time:.2f}ms Post:{postprocess_time:.2f}ms Callback:{callback_time:.2f}ms "
                          f"Total:{total_time:.2f}ms ready:{ready_queue.qsize()}")
                
            except Empty:
                continue
            except Exception as e:
                print(f"[Inference] Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 启动推理线程
    inference_thread = threading.Thread(target=inference_worker, daemon=True, name="Inference")
    inference_thread.start()
    
    # 等待线程
    inference_thread.join()
    for t in callback_threads:
        t.join()
    
    print("[GPUWorker] Shutting down...")
