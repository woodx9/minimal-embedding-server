#!/usr/bin/env python3
"""
OpenAI Compatible Embedding API 压力测试工具
支持并发请求、批量embedding、性能指标统计
支持 vLLM、SGLang 等 OpenAI compatible 框架
使用 transformers tokenizer 精确生成指定 token 长度的文本
持续压力测试模式 - 实时监控和日志记录
"""

import asyncio
import argparse
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import os
import aiohttp
import sys
import json
from transformers import AutoTokenizer


@dataclass
class RequestResult:
    """单个请求的结果"""
    success: bool
    latency: float  # 毫秒
    error_message: str = ""
    timestamp: float = 0.0


@dataclass
class TestMetrics:
    """测试指标统计"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    # 用于实时统计的滑动窗口（最近1000个请求）
    recent_latencies: List[float] = field(default_factory=list)
    recent_window_size: int = 1000
    
    def add_result(self, result: RequestResult):
        """添加一个请求结果"""
        self.total_requests += 1
        if result.success:
            self.successful_requests += 1
            self.latencies.append(result.latency)
            self.recent_latencies.append(result.latency)
            
            # 保持滑动窗口大小
            if len(self.recent_latencies) > self.recent_window_size:
                self.recent_latencies.pop(0)
        else:
            self.failed_requests += 1
            self.errors.append(result.error_message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """计算统计指标"""
        current_time = time.time()
        total_time = current_time - self.start_time
        
        stats = {
            "总请求数": self.total_requests,
            "成功请求数": self.successful_requests,
            "失败请求数": self.failed_requests,
            "成功率": f"{(self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0:.2f}%",
            "运行时长(秒)": f"{total_time:.2f}",
        }
        
        if self.latencies:
            stats.update({
                "平均延迟(ms)": f"{statistics.mean(self.latencies):.2f}",
                "中位数延迟(ms)": f"{statistics.median(self.latencies):.2f}",
                "最小延迟(ms)": f"{min(self.latencies):.2f}",
                "最大延迟(ms)": f"{max(self.latencies):.2f}",
                "标准差(ms)": f"{statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0:.2f}",
            })
            
            # 计算百分位数
            sorted_latencies = sorted(self.latencies)
            p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
            p90 = sorted_latencies[int(len(sorted_latencies) * 0.90)]
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            
            stats.update({
                "P50延迟(ms)": f"{p50:.2f}",
                "P90延迟(ms)": f"{p90:.2f}",
                "P95延迟(ms)": f"{p95:.2f}",
                "P99延迟(ms)": f"{p99:.2f}",
            })
        
        # 计算QPS
        if total_time > 0:
            current_qps = self.total_requests / total_time
            avg_qps = self.successful_requests / total_time
            stats.update({
                "当前QPS": f"{current_qps:.2f}",
                "平均QPS(成功)": f"{avg_qps:.2f}",
            })
        
        # 最近窗口的统计（用于实时监控）
        if self.recent_latencies:
            stats.update({
                "最近平均延迟(ms)": f"{statistics.mean(self.recent_latencies):.2f}",
                "最近中位数延迟(ms)": f"{statistics.median(self.recent_latencies):.2f}",
            })
        
        return stats
    
    def get_raw_statistics(self) -> Dict[str, float]:
        """获取原始数值统计（用于文件记录）"""
        current_time = time.time()
        total_time = current_time - self.start_time
        
        stats = {
            "timestamp": current_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
            "runtime_seconds": total_time,
        }
        
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            stats.update({
                "avg_latency_ms": statistics.mean(self.latencies),
                "median_latency_ms": statistics.median(self.latencies),
                "min_latency_ms": min(self.latencies),
                "max_latency_ms": max(self.latencies),
                "std_latency_ms": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
                "p50_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.50)],
                "p90_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.90)],
                "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)],
                "p99_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.99)],
            })
        
        if total_time > 0:
            stats.update({
                "current_qps": self.total_requests / total_time,
                "avg_qps_success": self.successful_requests / total_time,
            })
        
        if self.recent_latencies:
            stats.update({
                "recent_avg_latency_ms": statistics.mean(self.recent_latencies),
                "recent_median_latency_ms": statistics.median(self.recent_latencies),
            })
        
        # 添加错误统计信息
        if self.errors:
            # 统计错误类型
            error_types = {}
            for error in self.errors:
                error_type = error.split(':')[0] if ':' in error else error[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            stats.update({
                "error_types": error_types,
                "recent_errors": self.errors[-10:]  # 最近10个错误的详细信息
            })
        
        return stats


def generate_random_text_with_tokenizer(tokenizer, token_length: int) -> str:
    """
    使用 tokenizer 生成精确 token 长度的随机文本
    
    Args:
        tokenizer: transformers tokenizer
        token_length: 精确的 token 长度
        
    Returns:
        str: 生成的文本，token 数量精确等于 token_length
    """
    # 生成随机数字序列作为 tokens
    # 使用数字是因为大部分 tokenizer 对数字的 tokenization 比较稳定
    random_tokens = [str(random.randint(0, 9999)) for _ in range(token_length * 2)]
    text = ' '.join(random_tokens)
    
    # 编码并截断到精确长度
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # 如果 tokens 不够，继续添加
    while len(tokens) < token_length:
        additional_text = ' '.join([str(random.randint(0, 9999)) for _ in range(10)])
        additional_tokens = tokenizer.encode(additional_text, add_special_tokens=False)
        tokens.extend(additional_tokens)
    
    # 截断到精确长度
    tokens = tokens[:token_length]
    
    # 解码回文本
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # 验证 token 数量
    verify_tokens = tokenizer.encode(text, add_special_tokens=False)
    assert len(verify_tokens) == token_length, f"生成的文本 token 数量 {len(verify_tokens)} 不等于目标 {token_length}"
    
    return text


class EmbeddingStressTester:
    """Embedding API 压力测试器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        tokenizer_name: str,
        log_file: str = "stress_test_metrics.jsonl"
    ):
        """
        初始化测试器
        
        Args:
            api_key: API密钥（OpenAI compatible）
            base_url: API基础URL
            model: embedding模型名称
            tokenizer_name: tokenizer 名称（用于精确生成 token）
            log_file: 日志文件路径
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')  # 移除末尾斜杠
        self.model = model
        self.metrics = TestMetrics()
        self.session: Optional[aiohttp.ClientSession] = None
        self.log_file = log_file
        self.log_interval = 2  # 每2秒记录一次
        self.display_interval = 1  # 每1秒更新一次屏幕
        self.running = True
        
        # 加载 tokenizer
        print(f"正在加载 tokenizer: {tokenizer_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        print(f"✓ Tokenizer 加载成功\n")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """关闭 session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def single_embedding_request(
        self,
        texts: List[str],
        client_id: int
    ) -> RequestResult:
        """
        执行单个embedding请求
        
        Args:
            texts: 要embedding的文本列表
            client_id: 客户端ID
            
        Returns:
            RequestResult: 请求结果
        """
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # 构造 OpenAI 兼容的请求
            url = f"{self.base_url}/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model,
                "input": texts
            }
            
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                # 验证响应
                result = await response.json()
                if "data" not in result:
                    raise Exception(f"Invalid response format: {result}")
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            
            return RequestResult(
                success=True,
                latency=latency,
                timestamp=end_time
            )
            
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            error_msg = f"Client {client_id}: {type(e).__name__} - {str(e)}"
            
            return RequestResult(
                success=False,
                latency=latency,
                error_message=error_msg,
                timestamp=end_time
            )
    
    async def client_worker(
        self,
        client_id: int,
        batch_size: int,
        token_length: int,
        continuous: bool = True
    ):
        """
        单个客户端工作器，持续执行请求
        
        Args:
            client_id: 客户端ID
            batch_size: 每个请求的batch大小
            token_length: 每个文本的token长度
            continuous: 是否持续运行
        """
        req_num = 0
        while self.running:
            # 使用 tokenizer 生成精确 token 长度的随机文本
            texts = [
                generate_random_text_with_tokenizer(self.tokenizer, token_length) 
                for _ in range(batch_size)
            ]
            
            # 执行请求
            result = await self.single_embedding_request(texts, client_id)
            
            # 记录结果
            self.metrics.add_result(result)
            
            req_num += 1
            
            if not continuous:
                break
    
    async def log_metrics_worker(self):
        """定期将指标写入日志文件"""
        while self.running:
            await asyncio.sleep(self.log_interval)
            
            if self.metrics.total_requests > 0:
                stats = self.metrics.get_raw_statistics()
                stats['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 写入 JSONL 格式
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(stats, ensure_ascii=False) + '\n')
    
    async def display_metrics_worker(self):
        """实时显示指标到屏幕"""
        while self.running:
            await asyncio.sleep(self.display_interval)
            
            if self.metrics.total_requests > 0:
                self.display_live_metrics()
    
    def display_live_metrics(self):
        """在屏幕上显示实时指标"""
        # 移动光标到屏幕顶部并清除之后的内容
        # \033[H 移动到顶部，\033[J 清除光标之后的内容
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()
        
        stats = self.metrics.get_statistics()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"{'='*80}")
        print(f"🔥 持续压力测试实时监控 - {current_time}")
        print(f"{'='*80}")
        print(f"📋 模型: {self.model}")
        print(f"📊 日志文件: {self.log_file}")
        print(f"{'='*80}\n")
        
        # 基本统计
        print(f"📈 请求统计:")
        print(f"  ├─ 总请求数:     {stats['总请求数']:>12,}")
        print(f"  ├─ 成功请求:     {stats['成功请求数']:>12,}")
        print(f"  ├─ 失败请求:     {stats['失败请求数']:>12,}")
        print(f"  ├─ 成功率:       {stats['成功率']:>12}")
        print(f"  └─ 运行时长:     {stats['运行时长(秒)']:>12} 秒\n")
        
        # QPS
        if '当前QPS' in stats:
            print(f"🚀 吞吐量:")
            print(f"  ├─ 当前QPS:      {stats['当前QPS']:>12}")
            print(f"  └─ 平均QPS:      {stats['平均QPS(成功)']:>12}\n")
        
        # 延迟统计
        if '平均延迟(ms)' in stats:
            print(f"⏱  延迟统计:")
            print(f"  ├─ 平均延迟:     {stats['平均延迟(ms)']:>12} ms")
            print(f"  ├─ 中位数延迟:   {stats['中位数延迟(ms)']:>12} ms")
            print(f"  ├─ 最小延迟:     {stats['最小延迟(ms)']:>12} ms")
            print(f"  ├─ 最大延迟:     {stats['最大延迟(ms)']:>12} ms")
            print(f"  └─ 标准差:       {stats['标准差(ms)']:>12} ms\n")
            
            print(f"📊 延迟百分位:")
            print(f"  ├─ P50:          {stats['P50延迟(ms)']:>12} ms")
            print(f"  ├─ P90:          {stats['P90延迟(ms)']:>12} ms")
            print(f"  ├─ P95:          {stats['P95延迟(ms)']:>12} ms")
            print(f"  └─ P99:          {stats['P99延迟(ms)']:>12} ms\n")
        
        # 最近窗口统计
        if '最近平均延迟(ms)' in stats:
            print(f"🔄 最近{self.metrics.recent_window_size}个请求:")
            print(f"  ├─ 平均延迟:     {stats['最近平均延迟(ms)']:>12} ms")
            print(f"  └─ 中位数延迟:   {stats['最近中位数延迟(ms)']:>12} ms\n")
        
        # 错误统计
        if self.metrics.failed_requests > 0:
            error_types = {}
            for error in self.metrics.errors[-100:]:  # 只统计最近100个错误
                error_type = error.split(':')[0] if ':' in error else error[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            print(f"❌ 错误统计 (最近100个):")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  ├─ {error_type}: {count}")
            print()
        
        print(f"{'='*80}")
        print(f"💡 提示: 按 Ctrl+C 停止测试")
        print(f"{'='*80}")
    
    async def run_continuous_stress_test(
        self,
        concurrent_clients: int,
        batch_size: int,
        token_length: int
    ):
        """
        运行持续压力测试
        
        Args:
            concurrent_clients: 并发客户端数量
            batch_size: 每个请求的batch大小
            token_length: token长度
        """
        # 清屏并显示启动信息
        print("\033[2J\033[H", end='', flush=True)
        
        print(f"\n{'='*80}")
        print(f"🚀 启动持续压力测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"配置:")
        print(f"  ├─ 并发客户端数:   {concurrent_clients}")
        print(f"  ├─ 每请求Batch数:  {batch_size}")
        print(f"  ├─ Token长度:      {token_length}")
        print(f"  ├─ 模型:           {self.model}")
        print(f"  ├─ Tokenizer:      {self.tokenizer.name_or_path}")
        print(f"  ├─ 日志文件:       {self.log_file}")
        print(f"  ├─ 日志间隔:       {self.log_interval}秒")
        print(f"  └─ 显示刷新:       {self.display_interval}秒")
        print(f"{'='*80}\n")
        print("⏳ 准备启动...", flush=True)
        
        # 创建日志文件头
        with open(self.log_file, 'w', encoding='utf-8') as f:
            header = {
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "model": self.model,
                "concurrent_clients": concurrent_clients,
                "batch_size": batch_size,
                "token_length": token_length
            }
            f.write(json.dumps(header, ensure_ascii=False) + '\n')
        
        print(f"✓ 日志文件已创建: {self.log_file}\n")
        await asyncio.sleep(2)
        
        self.metrics.start_time = time.time()
        self.running = True
        
        # 创建并发任务
        tasks = []
        
        # 客户端工作器
        for i in range(concurrent_clients):
            task = asyncio.create_task(
                self.client_worker(
                    client_id=i,
                    batch_size=batch_size,
                    token_length=token_length,
                    continuous=True
                )
            )
            tasks.append(task)
        
        # 日志记录工作器
        log_task = asyncio.create_task(self.log_metrics_worker())
        tasks.append(log_task)
        
        # 显示工作器
        display_task = asyncio.create_task(self.display_metrics_worker())
        tasks.append(display_task)
        
        try:
            # 等待所有任务（实际上会一直运行直到被中断）
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.close()
    
    async def run_stress_test(
        self,
        concurrent_clients: int,
        batch_size: int,
        token_length: int,
        requests_per_client: int = 10
    ):
        """
        运行压力测试（非持续模式，保留用于向后兼容）
        
        Args:
            concurrent_clients: 并发客户端数量
            batch_size: 每个请求的batch大小
            token_length: token长度
            requests_per_client: 每个客户端的请求数
        """
        print(f"\n{'='*60}")
        print(f"开始压力测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"配置:")
        print(f"  - 并发客户端数: {concurrent_clients}")
        print(f"  - 每请求Batch数: {batch_size}")
        print(f"  - Token长度(精确): {token_length}")
        print(f"  - 每客户端请求数: {requests_per_client}")
        print(f"  - 总请求数: {concurrent_clients * requests_per_client}")
        print(f"  - 模型: {self.model}")
        print(f"  - Tokenizer: {self.tokenizer.name_or_path}")
        print(f"{'='*60}\n")
        
        self.metrics.start_time = time.time()
        self.running = True
        
        # 创建并发任务
        tasks = []
        for i in range(concurrent_clients):
            for _ in range(requests_per_client):
                task = asyncio.create_task(
                    self.client_worker(
                        client_id=i,
                        batch_size=batch_size,
                        token_length=token_length,
                        continuous=False
                    )
                )
                tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        self.metrics.end_time = time.time()
        
        # 关闭 session
        await self.close()
    
    def print_results(self):
        """打印测试结果"""
        print(f"\n{'='*80}")
        print(f"压力测试结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        stats = self.metrics.get_statistics()
        
        print("📊 性能指标:")
        print(f"  ✓ {stats['总请求数']} 个请求")
        print(f"  ✓ {stats['成功请求数']} 成功")
        print(f"  ✗ {stats['失败请求数']} 失败")
        print(f"  📈 成功率: {stats['成功率']}")
        print(f"  ⏱  运行时长: {stats['运行时长(秒)']} 秒")
        
        if '平均延迟(ms)' in stats:
            print(f"\n⏱ 延迟统计:")
            print(f"  • 平均延迟: {stats['平均延迟(ms)']} ms")
            print(f"  • 中位数延迟: {stats['中位数延迟(ms)']} ms")
            print(f"  • 最小延迟: {stats['最小延迟(ms)']} ms")
            print(f"  • 最大延迟: {stats['最大延迟(ms)']} ms")
            print(f"  • 标准差: {stats['标准差(ms)']} ms")
            
            print(f"\n📊 百分位数:")
            print(f"  • P50: {stats['P50延迟(ms)']} ms")
            print(f"  • P90: {stats['P90延迟(ms)']} ms")
            print(f"  • P95: {stats['P95延迟(ms)']} ms")
            print(f"  • P99: {stats['P99延迟(ms)']} ms")
        
        if '当前QPS' in stats:
            print(f"\n🚀 QPS指标:")
            print(f"  • 当前QPS: {stats['当前QPS']}")
            print(f"  • 平均QPS(成功): {stats['平均QPS(成功)']}")
        
        # 打印错误详情
        if self.metrics.failed_requests > 0:
            print(f"\n❌ 失败详情 (共 {self.metrics.failed_requests} 个):")
            # 统计错误类型
            error_types = {}
            for error in self.metrics.errors:
                error_type = error.split(':')[0] if ':' in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {error_type}: {count} 次")
            
            # 显示前5个详细错误
            print(f"\n  详细错误信息 (前5个):")
            for i, error in enumerate(self.metrics.errors[:5], 1):
                print(f"    {i}. {error}")
        
        print(f"\n📁 日志文件: {self.log_file}")
        print(f"\n{'='*80}\n")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='OpenAI Compatible Embedding API 压力测试工具 (支持 vLLM, SGLang 等)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 持续压力测试（默认模式）
  python stress_test.py --concurrent-clients 10 --batch-size 32 --token-length 512 \\
      --model Qwen/Qwen3-Embedding-0.6B --base-url http://localhost:8000

  # 固定次数测试
  python stress_test.py --concurrent-clients 10 --batch-size 32 --token-length 512 \\
      --requests-per-client 100 --model Qwen/Qwen3-Embedding-0.6B --base-url http://localhost:8000
        """
    )
    parser.add_argument('--concurrent-clients', type=int, required=True,
                        help='并发客户端数量')
    parser.add_argument('--batch-size', type=int, required=True,
                        help='每个请求的batch大小')
    parser.add_argument('--token-length', type=int, required=True,
                        help='每个文本的精确token长度')
    parser.add_argument('--requests-per-client', type=int, default=None,
                        help='每个客户端的请求数 (不设置则持续运行)')
    parser.add_argument('--model', type=str, required=True,
                        help='Embedding模型名称 (例如: Qwen/Qwen2.5-1.5B)')
    parser.add_argument('--base-url', type=str, required=True,
                        help='API基础URL (例如: http://localhost:8000/v1)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API密钥 (默认: OPENAI_API_KEY环境变量)')
    parser.add_argument('--log-file', type=str, default='stress_test_metrics.jsonl',
                        help='日志文件路径 (默认: stress_test_metrics.jsonl)')
    parser.add_argument('--log-interval', type=float, default=2.0,
                        help='日志记录间隔(秒) (默认: 2.0)')
    parser.add_argument('--display-interval', type=float, default=1.0,
                        help='屏幕刷新间隔(秒) (默认: 1.0)')
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key or os.getenv('OPENAI_API_KEY', 'EMPTY')
    
    # tokenizer 使用 model 名称
    tokenizer_name = args.model
    
    # 创建测试器
    try:
        tester = EmbeddingStressTester(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            tokenizer_name=tokenizer_name,
            log_file=args.log_file
        )
        tester.log_interval = args.log_interval
        tester.display_interval = args.display_interval
    except Exception as e:
        print(f"❌ 初始化失败: {type(e).__name__} - {str(e)}")
        print(f"\n提示: 请确保 tokenizer '{tokenizer_name}' 可以正确加载")
        sys.exit(1)
    
    # 运行测试
    try:
        if args.requests_per_client is None:
            # 持续压力测试模式
            await tester.run_continuous_stress_test(
                concurrent_clients=args.concurrent_clients,
                batch_size=args.batch_size,
                token_length=args.token_length
            )
        else:
            # 固定次数测试模式
            await tester.run_stress_test(
                concurrent_clients=args.concurrent_clients,
                batch_size=args.batch_size,
                token_length=args.token_length,
                requests_per_client=args.requests_per_client
            )
            
            # 打印结果
            tester.print_results()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断", flush=True)
        tester.running = False
        await asyncio.sleep(1)  # 等待工作器停止
        await tester.close()
        
        # 清屏后打印最终结果
        print("\033[2J\033[H", end='', flush=True)
        tester.print_results()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 测试出现错误: {type(e).__name__} - {str(e)}")
        tester.running = False
        await tester.close()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
