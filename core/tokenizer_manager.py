"""
Tokenizer Manager - CPU密集型的tokenizer和batch准备进程
"""
import torch
from transformers import AutoTokenizer
import threading
from queue import Queue, Empty
import time


def tokenizer_manager_main(model_name, device, raw_queue, ready_queue, num_tokenizer_threads, 
                          batch_timeout, max_tokens_per_batch, enable_monitoring):
    """
    Tokenizer Manager 进程入口
    负责：
    1. 多线程 Tokenizer（CPU密集型）
    2. 单线程 Batch 聚合
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[TokenizerManager] Loaded tokenizer, starting {num_tokenizer_threads} tokenizer threads...")
    
    # 内部队列：tokenized_queue
    tokenized_queue = Queue(maxsize=500)
    
    def tokenize_worker():
        """Tokenizer 线程 - CPU密集型tokenization"""
        while True:
            try:
                item = raw_queue.get(timeout=0.1)
                if item is None:  # 终止信号
                    break
                
                texts, future_id = item
                
                try:
                    # Tokenize（CPU密集型操作）
                    all_input_ids = []
                    all_positions = []
                    seq_lengths = []
                    
                    for text in texts:
                        encoded = tokenizer(text, return_tensors="pt")
                        ids = encoded['input_ids'].squeeze(0).tolist()
                        seq_len = len(ids)
                        seq_lengths.append(seq_len)
                        
                        all_input_ids.extend(ids)
                        all_positions.extend(list(range(seq_len)))
                    
                    # 转为 CPU tensor
                    input_ids = torch.tensor(all_input_ids, dtype=torch.int64)
                    positions = torch.tensor(all_positions, dtype=torch.int64)
                    
                    # 放入内部队列
                    tokenized_queue.put((input_ids, positions, seq_lengths, future_id))
                except Exception as e:
                    print(f"[Tokenizer] Error: {e}")
                    import traceback
                    traceback.print_exc()
            except Empty:
                continue
    
    def batch_prepare_worker():
        """BatchPrepare 线程 - 单线程聚合，激进收集"""
        thread_id = "BatchPrepare-0"
        
        while True:
            batch = []
            total_tokens = 0
            start_time = time.time()
            collect_start = time.time()
            
            # 第一步：阻塞等待第一个请求
            try:
                item = tokenized_queue.get(timeout=0.1)
                input_ids, positions, seq_lengths, future_id = item
                batch.append(item)
                total_tokens += len(input_ids)
            except Empty:
                continue
            
            # 第二步：激进聚合 - 根据GPU负载动态调整等待策略
            max_wait_rounds = 1 if ready_queue.qsize() < 3 else 10
            wait_rounds = 0
            
            while total_tokens < max_tokens_per_batch and wait_rounds < max_wait_rounds:
                collected = False
                
                # 快速收集队列里所有等待的请求
                while total_tokens < max_tokens_per_batch:
                    try:
                        item = tokenized_queue.get_nowait()
                        input_ids, positions, seq_lengths, future_id = item
                        item_tokens = len(input_ids)
                        
                        if total_tokens + item_tokens > max_tokens_per_batch:
                            tokenized_queue.put(item)
                            break
                        
                        batch.append(item)
                        total_tokens += item_tokens
                        collected = True
                        wait_rounds = 0
                    except Empty:
                        break
                
                if collected:
                    continue
                
                # 等待新请求
                try:
                    item = tokenized_queue.get(timeout=batch_timeout)
                    input_ids, positions, seq_lengths, future_id = item
                    item_tokens = len(input_ids)
                    
                    if total_tokens + item_tokens <= max_tokens_per_batch:
                        batch.append(item)
                        total_tokens += item_tokens
                        wait_rounds = 0
                    else:
                        tokenized_queue.put(item)
                        break
                except Empty:
                    wait_rounds += 1
            
            if batch:
                collect_time = (time.time() - collect_start) * 1000
                
                # Merge - 在CPU上完成
                merge_start = time.time()
                all_input_ids = []
                all_positions = []
                all_seq_lengths = []
                future_ids = []
                
                for input_ids, positions, seq_lengths, future_id in batch:
                    all_input_ids.append(input_ids)
                    all_positions.append(positions)
                    all_seq_lengths.extend(seq_lengths)
                    future_ids.append(future_id)
                
                # 预计算 last_token_indices（避免GPU阶段的循环）
                last_token_indices = []
                global_idx = 0
                for seq_len in all_seq_lengths:
                    last_token_indices.append(global_idx + seq_len - 1)
                    global_idx += seq_len
                
                # CPU 合并
                merged_input_ids = torch.cat(all_input_ids, dim=0)
                merged_positions = torch.cat(all_positions, dim=0)
                merge_time = (time.time() - merge_start) * 1000
                
                if enable_monitoring:
                    total_time = (time.time() - start_time) * 1000
                    avg_tokens = total_tokens / len(batch)
                    print(f"[{thread_id}] Batch:{len(batch)} TotalTokens:{total_tokens} AvgTokens:{avg_tokens:.0f} "
                          f"Collect:{collect_time:.2f}ms Merge:{merge_time:.2f}ms Total:{total_time:.2f}ms "
                          f"ready:{ready_queue.qsize()}")
                
                # 发送到GPU Worker（转numpy方便跨进程序列化）
                ready_queue.put((
                    merged_input_ids.numpy(),
                    merged_positions.numpy(),
                    all_seq_lengths,
                    last_token_indices,
                    future_ids
                ))
    
    # 启动 tokenizer 线程池
    tokenizer_threads = []
    for i in range(num_tokenizer_threads):
        t = threading.Thread(target=tokenize_worker, daemon=True, name=f"Tokenizer-{i}")
        t.start()
        tokenizer_threads.append(t)
    
    # 启动 batch_prepare 线程
    batch_thread = threading.Thread(target=batch_prepare_worker, daemon=True, name="BatchPrepare")
    batch_thread.start()
    
    # 等待线程
    for t in tokenizer_threads:
        t.join()
    batch_thread.join()
    
    print("[TokenizerManager] Shutting down...")
