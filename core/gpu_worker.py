"""
GPU Worker - GPU密集型的推理和后处理进程
"""
import torch
import torch.nn.functional as F
import models
from models.registry import get_model_class
from ultils.loader import load_model
from ultils.dtype_utils import get_torch_dtype, dtype_to_string
from huggingface_hub import snapshot_download
import time
import threading
from queue import Queue, Empty
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
import pickle


class GPUWorker:
    """
    GPU Worker 进程入口
    负责：
    1. 加载模型到GPU
    2. GPU推理（向量化）
    3. 后处理（批量归一化）
    4. 异步回调
    """

    def __init__(
        self,
        rank,
        world_size,
        event,
        ready_queue,
        result_queue,
        mes_config,
        nccl_port,
    ):
        self.rank = rank
        self.ready_queue = ready_queue
        self.result_queue = result_queue
        self.mes_config = mes_config
        self.model_name = mes_config.model_name
        self.max_tokens_per_batch = mes_config.max_tokens_per_batch
        self.enable_monitoring = mes_config.enable_monitoring
        self.callback_queue = Queue(maxsize=1000)
        self.rank = rank
        self.world_size = world_size
        self.event = event 
        self.nccl_port = nccl_port
        self.model = None
        self.run()

    def run(self):
        # 初始化分布式环境
        dist.init_process_group(
            "nccl", f"tcp://localhost:{self.nccl_port}", world_size=self.world_size, rank=self.rank
        )
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f"cuda:{self.rank}")

        print(f"[GPUWorker] Loading model on rank:{self.rank}...")
        
        torch_dtype = get_torch_dtype(self.mes_config.dtype, self.mes_config.model_config)
        print(f"[GPUWorker] Using dtype: {dtype_to_string(torch_dtype)}")
        
        model_class = get_model_class(self.mes_config.model_config)
        print(f"[GPUWorker] Using model class: {model_class.__name__}")
        
        self.model = (
            model_class(self.mes_config)
            .to(self.device)
            .to(torch_dtype)
        )
        self.model.eval()
        model_path = snapshot_download(self.model_name)
        load_model(self.model, model_path)
        print(f"[GPUWorker] Model loaded successfully on {self.device} with dtype {dtype_to_string(torch_dtype)}")


        callback_threads = []
        if self.rank == 0:
            # 启动回调线程池（4个线程异步处理回调）
            for i in range(4):
                t = threading.Thread(
                    target=self._callback_worker, daemon=True, name=f"Callback-{i}"
                )
                t.start()
                callback_threads.append(t)
        
            # 启动推理线程
            inference_thread = threading.Thread(
                target=self._inference_worker, daemon=True, name=f"Inference:{self.rank}"
            )
            print(f"[GPUWorker] Starting inference thread on rank:{self.rank}...")
            inference_thread.start()

        if self.world_size > 1:
            if self.rank == 0:
                self.shm = SharedMemory(name="minimal-embedding-server", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="minimal-embedding-server")
                self.loop()
        
        # 等待线程
        inference_thread.join()
        for t in callback_threads:
            t.join()
        print("[GPUWorker] Shutting down...")

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def _callback_worker(self):
        """异步回调线程 - 将结果发送回主进程"""
        while True:
            try:
                embeddings_list, seq_lengths, future_ids = self.callback_queue.get(timeout=0.1)

                # 发送结果回主进程
                self.result_queue.put((embeddings_list, seq_lengths, future_ids))
            except Empty:
                continue
            except Exception as e:
                print(f"[Callback] Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def _inference(self, input_ids, positions):
        input_ids = torch.from_numpy(input_ids).to(self.device)
        positions = torch.from_numpy(positions).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, positions=positions)
        return outputs

    def _inference_worker(self):
        """推理线程 - GPU密集型操作"""
        while True:
            try:
                wait_start = time.time()
                # 获取准备好的 batch
                batch_data = self.ready_queue.get(timeout=0.1)
                if batch_data is None:  # 终止信号
                    break

                (
                    merged_input_ids_np,
                    merged_positions_np,
                    all_seq_lengths,
                    last_token_indices,
                    future_ids,
                ) = batch_data
                wait_time = (time.time() - wait_start) * 1000
                
                # GPU 推理
                inference_start = time.time()
                with torch.no_grad():
                    outputs = self.call("_inference", merged_input_ids_np, merged_positions_np)
                inference_time = (time.time() - inference_start) * 1000
                
                # 后处理（向量化操作，无循环）
                postprocess_start = time.time()

                # 1. 预计算的indices直接转GPU tensor
                last_token_indices_tensor = torch.tensor(
                    last_token_indices, device=self.device
                )

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
                self.callback_queue.put(
                    (all_embeddings_list, all_seq_lengths, future_ids)
                )
                callback_time = (time.time() - callback_start) * 1000

                if self.enable_monitoring:
                    total_time = (
                        wait_time
                        + inference_time
                        + postprocess_time
                        + callback_time
                    )
                    num_requests = len(future_ids)
                    print(
                        f"[Inference] Requests:{num_requests}"
                        f"Infer:{inference_time:.2f}ms Post:{postprocess_time:.2f}ms Callback:{callback_time:.2f}ms "
                        f"Total:{total_time:.2f}ms ready:{self.ready_queue.qsize()}"
                    )

            except Empty:
                continue
            except Exception as e:
                print(f"[Inference] Error: {e}")
                import traceback
                traceback.print_exc()
                continue