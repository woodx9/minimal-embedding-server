# start fast_api cli
# uvicorn openai_server.fast_api:app --reload --host 0.0.0.0 --port 8000

import argparse
import os
from fastapi import FastAPI

from core.engine import Engine
from schemas import http


app = FastAPI(title="MES - Minimal Embedding Server", version="0.1.0")
engine_instance = None  # 延迟初始化


@app.post("/v1/embeddings")
async def v1_embeddings(request: http.EmbeddingRequest): 
    input = request.input

    # 确保 input 是列表
    if isinstance(input, str):
        input = [input]

    embeddings_list, seq_lengths = await engine_instance.v1_embeddings(input)
    
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


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok", "version": "0.1.0"}


def start_server():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(
        description="MES - Minimal Embedding Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动服务器（使用默认配置）
  mes
  
  # 指定端口和主机
  mes --host 0.0.0.0 --port 8000
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("MES_HOST", "0.0.0.0"),
        help="服务器监听地址 (默认: 0.0.0.0，可通过 MES_HOST 环境变量设置)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MES_PORT", "8000")),
        help="服务器监听端口 (默认: 8000，可通过 MES_PORT 环境变量设置)"
    )

    parser.add_argument(
        "--attn-backend",
        type=str,
        default=os.getenv("MES_ATTENTION_BACKEND", "flash_attention"),
        choices=["flash_attention", "flash_infer"],
        help="注意力机制后端选择 (默认: flash_attention)"
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=int(os.getenv("MES_TENSOR_PARALLEL_SIZE", "1")),
        help="张量并行度 (默认: 1)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=os.getenv("MES_DTYPE", "auto"),
        choices=["auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"],
        help="模型权重和激活的数据类型 (默认: auto)"
    )
    
    args = parser.parse_args()
    
    # 初始化 Engine（在启动服务器之前）
    global engine_instance
    attn_backend = args.attn_backend if hasattr(args, 'attn_backend') else args.__dict__.get('attn-backend', 'flashattention')
    
    # 打印启动信息
    print("=" * 60)
    print("MES - Minimal Embedding Server")
    print("=" * 60)
    print(f"服务器地址: http://{args.host}:{args.port}")
    print(f"健康检查: http://{args.host}:{args.port}/health")
    print(f"注意力后端: {attn_backend}")
    print(f"数据类型: {args.dtype}")
    print(f"张量并行度: {args.tensor_parallel_size}")
    
    # 初始化 Engine
    print("正在初始化 Engine...")
    engine_instance = Engine(
        attn_backend=attn_backend, 
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype
    )
    print("Engine 初始化完成！")
    print()
    
    # 启动服务器
    import uvicorn
    uvicorn.run(
        "openai_server.fast_api:app",
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
