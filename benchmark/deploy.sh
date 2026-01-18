# 临时取消代理设置，避免 SOCKS 代理导致的错误
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY
export export HF_HUB_OFFLINE=0

python3 stress_test.py \
    --concurrent-clients 10 \
    --batch-size 20 \
    --token-length 1000 \
    --base-url http://localhost:8000 \
    --model Qwen/Qwen3-Embedding-0.6B