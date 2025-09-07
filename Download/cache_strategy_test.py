#!/usr/bin/env python3
"""
缓存策略测试脚本
实现三种缓存策略的测试：
1. Full KV recompute（无缓存复用）
2. Prefix caching（仅复用前缀缓存）
3. Full KV reuse（复用所有缓存但忽略交叉注意力）
"""

import requests
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import string

def rand_ascii(n: int) -> str:
    """生成随机 ASCII 字符串"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))

class CacheStrategyTester:
    """缓存策略测试器"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        
    def measure_ttft(self, prompt: str, description: str, use_cache: bool = True) -> Tuple[float, Dict]:
        """测量 TTFT"""
        print(f"📤 {description}...")
        start = time.perf_counter()
        
        try:
            # 根据缓存策略设置参数
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 20,
                "stream": False
            }
            
            # 添加缓存控制参数
            if not use_cache:
                request_data["lmcache_model_request"] = {
                    "store_cache": False,
                    "retrieve_cache": False
                }
            
            response = requests.post(
                f"{self.base_url}/completions",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            end = time.perf_counter()
            ttft = end - start
            
            result = response.json()
            generated_text = result['choices'][0]['text']
            usage = result.get('usage', {})
            
            print(f"📥 响应: {generated_text[:30]}...")
            print(f"📊 Tokens: {usage.get('prompt_tokens', 0)} -> {usage.get('completion_tokens', 0)}")
            print(f"⏱️  TTFT: {ttft:.3f}s")
            
            return ttft, result
            
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return 0.0, {}
    
    def flush_cache(self):
        """刷新缓存"""
        print("🧹 刷新 KV 缓存...")
        filler_prompt = f"Cache flush: {rand_ascii(1000)}"
        try:
            requests.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.model,
                    "prompt": filler_prompt,
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "stream": False
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
        except:
            pass  # 忽略刷新错误
        time.sleep(1)

    def test_strategy_1_full_recompute(self, prompt: str) -> Dict[str, Any]:
        """策略1: Full KV recompute（无缓存复用）"""
        print("\n" + "="*60)
        print("🔄 策略1: Full KV recompute（无缓存复用）")
        print("="*60)
        
        results = {}
        
        # 刷新缓存确保干净状态
        self.flush_cache()
        
        # 第一次请求（无缓存）
        ttft1, result1 = self.measure_ttft(
            prompt, 
            "第一次请求（无缓存）", 
            use_cache=False
        )
        results["first_request"] = {"ttft": ttft1, "result": result1}
        
        # 第二次请求（仍然无缓存）
        ttft2, result2 = self.measure_ttft(
            prompt, 
            "第二次请求（仍然无缓存）", 
            use_cache=False
        )
        results["second_request"] = {"ttft": ttft2, "result": result2}
        
        # 分析结果
        speedup = ttft1 / ttft2 if ttft2 > 0 else 1.0
        results["speedup"] = speedup
        results["strategy"] = "Full KV recompute"
        
        print(f"📊 加速比: {speedup:.2f}x (应该接近 1.0x)")
        
        return results

    def test_strategy_2_prefix_caching(self, prompt: str, prefix_length: int = 50) -> Dict[str, Any]:
        """策略2: Prefix caching（仅复用前缀缓存）"""
        print("\n" + "="*60)
        print("🔄 策略2: Prefix caching（仅复用前缀缓存）")
        print("="*60)
        
        results = {}
        
        # 刷新缓存确保干净状态
        self.flush_cache()
        
        # 创建前缀和后缀
        prefix = prompt[:prefix_length]
        suffix = prompt[prefix_length:]
        
        print(f"📝 前缀长度: {len(prefix)} 字符")
        print(f"📝 后缀长度: {len(suffix)} 字符")
        
        # 第一次请求（完整 prompt，建立缓存）
        ttft1, result1 = self.measure_ttft(
            prompt, 
            "第一次请求（完整 prompt，建立缓存）", 
            use_cache=True
        )
        results["full_request"] = {"ttft": ttft1, "result": result1}
        
        # 第二次请求（仅前缀，应该复用缓存）
        ttft2, result2 = self.measure_ttft(
            prefix, 
            "第二次请求（仅前缀，复用缓存）", 
            use_cache=True
        )
        results["prefix_request"] = {"ttft": ttft2, "result": result2}
        
        # 第三次请求（前缀+新后缀，部分复用）
        new_suffix = suffix + " " + rand_ascii(20)
        new_prompt = prefix + new_suffix
        ttft3, result3 = self.measure_ttft(
            new_prompt, 
            "第三次请求（前缀+新后缀，部分复用）", 
            use_cache=True
        )
        results["prefix_new_suffix"] = {"ttft": ttft3, "result": result3}
        
        # 分析结果
        prefix_speedup = ttft1 / ttft2 if ttft2 > 0 else 1.0
        partial_speedup = ttft1 / ttft3 if ttft3 > 0 else 1.0
        
        results["prefix_speedup"] = prefix_speedup
        results["partial_speedup"] = partial_speedup
        results["strategy"] = "Prefix caching"
        
        print(f"📊 前缀复用加速比: {prefix_speedup:.2f}x")
        print(f"📊 部分复用加速比: {partial_speedup:.2f}x")
        
        return results

    def test_strategy_3_full_reuse(self, prompt: str) -> Dict[str, Any]:
        """策略3: Full KV reuse（复用所有缓存但忽略交叉注意力）"""
        print("\n" + "="*60)
        print("🔄 策略3: Full KV reuse（复用所有缓存但忽略交叉注意力）")
        print("="*60)
        
        results = {}
        
        # 刷新缓存确保干净状态
        self.flush_cache()
        
        # 第一次请求（建立完整缓存）
        ttft1, result1 = self.measure_ttft(
            prompt, 
            "第一次请求（建立完整缓存）", 
            use_cache=True
        )
        results["first_request"] = {"ttft": ttft1, "result": result1}
        
        # 第二次请求（完全相同的 prompt，应该完全复用缓存）
        ttft2, result2 = self.measure_ttft(
            prompt, 
            "第二次请求（完全复用缓存）", 
            use_cache=True
        )
        results["second_request"] = {"ttft": ttft2, "result": result2}
        
        # 第三次请求（相似 prompt，部分复用）
        similar_prompt = prompt + " " + rand_ascii(10)
        ttft3, result3 = self.measure_ttft(
            similar_prompt, 
            "第三次请求（相似 prompt，部分复用）", 
            use_cache=True
        )
        results["similar_request"] = {"ttft": ttft3, "result": result3}
        
        # 分析结果
        full_speedup = ttft1 / ttft2 if ttft2 > 0 else 1.0
        partial_speedup = ttft1 / ttft3 if ttft3 > 0 else 1.0
        
        results["full_speedup"] = full_speedup
        results["partial_speedup"] = partial_speedup
        results["strategy"] = "Full KV reuse"
        
        print(f"📊 完全复用加速比: {full_speedup:.2f}x")
        print(f"📊 部分复用加速比: {partial_speedup:.2f}x")
        
        return results

    def run_comprehensive_test(self, prompt: str, prefix_length: int = 50) -> Dict[str, Any]:
        """运行综合测试"""
        print("🚀 开始缓存策略综合测试")
        print("="*80)
        print(f"📝 测试 prompt: {prompt[:100]}...")
        print(f"📏 Prompt 长度: {len(prompt)} 字符")
        print("="*80)
        
        all_results = {}
        
        # 测试策略1
        all_results["strategy_1"] = self.test_strategy_1_full_recompute(prompt)
        
        # 测试策略2
        all_results["strategy_2"] = self.test_strategy_2_prefix_caching(prompt, prefix_length)
        
        # 测试策略3
        all_results["strategy_3"] = self.test_strategy_3_full_reuse(prompt)
        
        # 生成总结报告
        self.generate_summary_report(all_results)
        
        return all_results

    def generate_summary_report(self, results: Dict[str, Any]):
        """生成总结报告"""
        print("\n" + "="*80)
        print("📋 缓存策略测试总结报告")
        print("="*80)
        
        strategies = [
            ("策略1: Full KV recompute", results.get("strategy_1", {})),
            ("策略2: Prefix caching", results.get("strategy_2", {})),
            ("策略3: Full KV reuse", results.get("strategy_3", {}))
        ]
        
        for name, data in strategies:
            print(f"\n{name}:")
            print("-" * 40)
            
            if "speedup" in data:
                print(f"  加速比: {data['speedup']:.2f}x")
            if "prefix_speedup" in data:
                print(f"  前缀复用加速比: {data['prefix_speedup']:.2f}x")
            if "partial_speedup" in data:
                print(f"  部分复用加速比: {data['partial_speedup']:.2f}x")
            if "full_speedup" in data:
                print(f"  完全复用加速比: {data['full_speedup']:.2f}x")
        
        print("\n💡 策略说明:")
        print("  策略1: 完全不使用缓存，每次重新计算")
        print("  策略2: 仅复用前缀部分的缓存")
        print("  策略3: 复用所有可能的缓存（忽略交叉注意力）")

def main():
    parser = argparse.ArgumentParser(description="缓存策略测试脚本")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--model", default="Qwen2.5-0.5B-Instruct", help="模型名称")
    parser.add_argument("--prompt", help="测试 prompt（如果不提供将使用默认）")
    parser.add_argument("--prefix-length", type=int, default=50, help="前缀长度（策略2）")
    parser.add_argument("--output", help="输出结果到文件")
    
    args = parser.parse_args()
    
    # 默认测试 prompt
    if not args.prompt:
        args.prompt = """
        请分析以下技术文档并回答相关问题：
        
        在深度学习领域，Transformer 架构已经成为自然语言处理的主流模型。
        Transformer 的核心是自注意力机制，它允许模型在处理序列时关注到序列中的任何位置。
        
        自注意力机制的计算公式为：
        Attention(Q,K,V) = softmax(QK^T/√d_k)V
        
        其中 Q、K、V 分别代表查询、键和值矩阵，d_k 是键向量的维度。
        
        请详细解释自注意力机制的工作原理，并说明它在 Transformer 中的作用。
        """
    
    # 创建测试器
    tester = CacheStrategyTester(args.api_base, args.model)
    
    # 运行测试
    results = tester.run_comprehensive_test(args.prompt, args.prefix_length)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {output_path}")

if __name__ == "__main__":
    main()
