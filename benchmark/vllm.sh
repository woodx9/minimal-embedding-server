export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=0
export VLLM_NO_USAGE_STATS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

vllm serve \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port 8000 \
    --max-num-seqs 10 \
    --max-model-len 1024 \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 10240 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    > vllm_server.log 2>&1

    