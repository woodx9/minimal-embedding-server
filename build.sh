#!/bin/bash
set -e

echo "=========================================="
echo "Minimal Embedding Server - 环境安装脚本"
echo "=========================================="

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

if [[ ! "$python_version" =~ ^3\.10 ]]; then
    echo "错误: 需要 Python 3.10，当前版本为 $python_version"
    echo "请使用: conda create -n flashattn5 python=3.10"
    exit 1
fi

echo ""
echo "步骤 1/5: 安装 PyTorch 2.4.1 (CUDA 12.1)..."
echo "----------------------------------------"
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "步骤 2/5: 下载 Flash Attention 2.8.3 预编译包..."
echo "----------------------------------------"
FLASH_ATTN_WHL="flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_ATTN_WHL}"

if [ -f "$FLASH_ATTN_WHL" ]; then
    echo "Flash Attention 已下载，跳过..."
else
    echo "正在下载 Flash Attention..."
    wget "$FLASH_ATTN_URL"
fi

echo ""
echo "步骤 3/5: 安装 Flash Attention..."
echo "----------------------------------------"
pip install "$FLASH_ATTN_WHL"
echo "----------------------------------------"
pip install -r requirements.txt

echo ""
echo "步骤 4/5: 验证安装..."
echo "----------------------------------------"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "启动服务："
echo "  uvicorn openai_server.fast_api:app --host 0.0.0.0 --port 8000"
echo ""
