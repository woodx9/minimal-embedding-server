#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal Embedding Server (MES) - 高性能多进程 Embedding 推理框架
"""
import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_flash_dependencies():
    """安装 Flash Attention 和 FlashInfer"""
    print("\n" + "="*60)
    print("正在安装 Flash Attention 和 FlashInfer...")
    print("="*60 + "\n")
    
    # Flash Attention
    flash_attn_whl = "flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    flash_attn_url = f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/{flash_attn_whl}"
    
    if not os.path.exists(flash_attn_whl):
        print(f"下载 Flash Attention...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", flash_attn_url])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", flash_attn_whl])
    
    # FlashInfer
    flashinfer_urls = [
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.1/flashinfer_python-0.6.1-py3-none-any.whl",
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.1/flashinfer_cubin-0.6.1-py3-none-any.whl",
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.1/flashinfer_jit_cache-0.6.1+cu129-cp39-abi3-manylinux_2_28_x86_64.whl",
    ]
    
    for url in flashinfer_urls:
        print(f"安装 {url.split('/')[-1]}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", url])
    
    print("\n" + "="*60)
    print("Flash 依赖安装完成！")
    print("="*60 + "\n")


class PostDevelopCommand(develop):
    """开发模式安装后钩子"""
    def run(self):
        develop.run(self)
        install_flash_dependencies()
            


class PostInstallCommand(install):
    """安装后钩子"""
    def run(self):
        install.run(self)
        install_flash_dependencies()


# 读取 README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# 读取基础依赖
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# PyTorch 依赖（从 CUDA 12.1 源安装）
torch_requirements = [
    "torch==2.4.1",
    "torchvision==0.19.1",
]

setup(
    name="mes",
    version="0.1.0",
    author="woodx9",
    author_email="",
    description="高性能多进程 Embedding 推理框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/woodx9/minimal-embedding-server",
    packages=find_packages(exclude=["test", "benchmark"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements + torch_requirements,
    dependency_links=[
        "https://download.pytorch.org/whl/cu121",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mes=openai_server.fast_api:start_server",
        ],
    },
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    include_package_data=True,
    zip_safe=False,
)
