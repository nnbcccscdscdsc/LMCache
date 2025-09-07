#!/usr/bin/env python3
"""
严格的 KV Cache 测试
通过比较相同 prompt 和不同 prompt 的性能来验证 KV cache 是否真的在工作
支持命令行指定 vLLM 实例端口
"""

import requests
import json
import time
import random
import string
import argparse
from pathlib import Path

def rand_ascii(n: int) -> str:
    """生成随机 ASCII 字符串"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))

def measure_ttft(base_url: str, model: str, prompt: str, description: str) -> tuple[float, dict]:
    """测量 TTFT"""
    print(f"📤 {description}...")
    start = time.perf_counter()
    
    try:
        response = requests.post(
            f"{base_url}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 20,
                "stream": False
            },
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
        
        return ttft, result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        raise

def test_single_instance_cache(base_url: str, model: str, prompt: str):
    """测试单实例的缓存功能"""
    results = []
    
    print(f"\n🔄 === 单实例 LMCache 测试 ===")
    print(f"📄 测试 prompt: {len(prompt)} 字符")
    print(f"🌐 测试实例: {base_url}")
    
    # 1. 冷启动
    print(f"\n🥶 冷启动")
    ttft1, result1 = measure_ttft(base_url, model, prompt, "冷启动请求")
    results.append(("cold_start", ttft1, result1))
    print(f"🕐 TTFT = {ttft1:.3f}s")
    
    # 2. 相同 prompt (应该命中缓存)
    print(f"\n🔥 相同 prompt (应该命中缓存)")
    ttft2, result2 = measure_ttft(base_url, model, prompt, "缓存命中请求")
    results.append(("cached", ttft2, result2))
    print(f"🕐 TTFT = {ttft2:.3f}s")
    
    # 3. 不同 prompt (应该不命中缓存)
    different_prompt = "Document: " + rand_ascii(1000) + "\nSummary:"
    print(f"\n🆕 不同 prompt (应该不命中缓存)")
    ttft3, result3 = measure_ttft(base_url, model, different_prompt, "不同 prompt 请求")
    results.append(("different_prompt", ttft3, result3))
    print(f"🕐 TTFT = {ttft3:.3f}s")
    
    # 4. 再次相同 prompt (应该再次命中缓存)
    print(f"\n🔄 再次相同 prompt (应该再次命中缓存)")
    ttft4, result4 = measure_ttft(base_url, model, prompt, "再次相同 prompt 请求")
    results.append(("cached_again", ttft4, result4))
    print(f"🕐 TTFT = {ttft4:.3f}s")
    
    # 5. 刷新缓存后测试
    print(f"\n🧹 刷新缓存后测试")
    flush_kv_cache(base_url, model)
    time.sleep(2)  # 等待缓存刷新
    ttft5, result5 = measure_ttft(base_url, model, prompt, "刷新缓存后请求")
    results.append(("after_flush", ttft5, result5))
    print(f"🕐 TTFT = {ttft5:.3f}s")
    
    return results

def test_cross_instance_cache(base_urls, model, prompt):
    """测试跨实例的缓存共享"""
    results = []
    
    print(f"\n🔄 === 跨实例 LMCache 共享测试 ===")
    print(f"📄 测试 prompt: {len(prompt)} 字符")
    print(f"🌐 测试实例数量: {len(base_urls)}")
    
    # 1. 在第一个实例上冷启动
    print(f"\n🥶 实例 1 (端口 {base_urls[0].split(':')[-1]}): 冷启动")
    ttft1, result1 = measure_ttft(base_urls[0], model, prompt, "冷启动请求")
    results.append(("instance_1_cold", ttft1, result1))
    print(f"🕐 TTFT = {ttft1:.3f}s")
    
    # 2. 在第一个实例上再次请求（应该命中 LMCache）
    print(f"\n🔥 实例 1 (端口 {base_urls[0].split(':')[-1]}): 相同 prompt")
    ttft1_cached, result1_cached = measure_ttft(base_urls[0], model, prompt, "缓存命中请求")
    results.append(("instance_1_cached", ttft1_cached, result1_cached))
    print(f"🕐 TTFT = {ttft1_cached:.3f}s")
    
    # 3. 在其他实例上请求相同 prompt（测试跨实例 LMCache 共享）
    for i, base_url in enumerate(base_urls[1:], 2):
        port = base_url.split(':')[-1]
        print(f"\n🔄 实例 {i} (端口 {port}): 测试跨实例 LMCache 共享")
        ttft_cross, result_cross = measure_ttft(base_url, model, prompt, f"实例 {i} 跨实例请求")
        results.append((f"instance_{i}_cross", ttft_cross, result_cross))
        print(f"🕐 TTFT = {ttft_cross:.3f}s")
        
        # 分析跨实例缓存效果
        if ttft_cross < ttft1 * 0.8:  # 如果比冷启动快 20% 以上
            print(f"✅ 实例 {i} 成功复用了实例 1 的 LMCache！")
        elif ttft_cross < ttft1 * 1.1:  # 如果接近冷启动时间
            print(f"🟡 实例 {i} 可能有部分 LMCache 复用")
        else:
            print(f"🔴 实例 {i} 没有 LMCache 复用")
    
    # 4. 回到第一个实例，再次请求（验证缓存仍然有效）
    print(f"\n🔄 实例 1 (端口 {base_urls[0].split(':')[-1]}): 验证缓存持久性")
    ttft1_verify, result1_verify = measure_ttft(base_urls[0], model, prompt, "验证缓存持久性")
    results.append(("instance_1_verify", ttft1_verify, result1_verify))
    print(f"🕐 TTFT = {ttft1_verify:.3f}s")
    
    return results

def flush_kv_cache(base_url: str, model: str) -> None:
    """使用智能截断的激进策略刷新 KV 缓存"""
    print("🔄 智能截断策略刷新 KV 缓存...")
    
    # 使用与 openai_chat_completion_client.py 相同的参数，但会智能截断
    FILLER_LEN_CHARS = 100_000  # 原始长度
    NUM_FILLER_PROMPTS = 10      # 发送 10 个填充请求
    
    # 安全长度：确保在模型限制内
    # facebook/opt-125m 上下文窗口是 2048 tokens，留出安全边距
    SAFE_TOKEN_LIMIT = 600  # 保守估计，约 1500 tokens
    
    for i in range(NUM_FILLER_PROMPTS):
        # 生成超长 prompt
        long_prompt = f"Random filler text {i}: {rand_ascii(FILLER_LEN_CHARS)}"
        
        # 智能截断：如果 prompt 太长，截断到安全长度
        if len(long_prompt) > SAFE_TOKEN_LIMIT * 4:  # 假设 4 字符 ≈ 1 token
            safe_prompt = long_prompt[:SAFE_TOKEN_LIMIT * 4]
            print(f"  📏 截断 prompt {i+1} 到安全长度: {len(safe_prompt)} 字符")
        else:
            safe_prompt = long_prompt
        
        try:
            response = requests.post(
                f"{base_url}/completions",
                json={
                    "model": model,
                    "prompt": safe_prompt,
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "stream": False
                },
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            print(f"  ✅ 填充 {i+1}/{NUM_FILLER_PROMPTS} 完成")
        except Exception as e:
            print(f"  ⚠️ 填充 {i+1}/{NUM_FILLER_PROMPTS} 失败: {e}")
            # 如果仍然失败，尝试更短的 prompt
            try:
                short_prompt = f"Filler {i}: {rand_ascii(1000)}"
                response = requests.post(
                    f"{base_url}/completions",
                    json={
                        "model": model,
                        "prompt": short_prompt,
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "stream": False
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                response.raise_for_status()
                print(f"  ✅ 填充 {i+1}/{NUM_FILLER_PROMPTS} 完成 (使用短 prompt)")
            except Exception as e2:
                print(f"  ❌ 填充 {i+1}/{NUM_FILLER_PROMPTS} 完全失败: {e2}")
    
    print("🔄 智能截断策略 KV 缓存刷新完成")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LMCache 测试脚本")
    parser.add_argument(
        "--ports", 
        nargs="+", 
        type=int, 
        default=[8000],
        help="要测试的 vLLM 实例端口列表 (默认: 8000)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="facebook/opt-125m",
        help="要测试的模型名称 (默认: facebook/opt-125m)"
    )
    parser.add_argument(
        "--test-type",
        choices=["single", "cross"],
        default="single",
        help="测试类型: single(单实例) 或 cross(跨实例) (默认: single)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 根据命令行参数构建 base_urls
    base_urls = [f"http://localhost:{port}/v1" for port in args.ports]
    
    model = args.model
    test_type = args.test_type
    
    print("🧪 LMCache 测试")
    print(f"📍 测试类型: {test_type}")
    print(f"📍 测试实例: {len(base_urls)} 个")
    print(f"🤖 Model: {model}")
    print(f"🌐 端口: {args.ports}")
    print("=" * 80)
    
    # 检查实例是否可用
    print("🔍 检查 vLLM 实例状态...")
    available_instances = []
    for base_url in base_urls:
        try:
            response = requests.get(f"{base_url}/models", timeout=10)
            if response.status_code == 200:
                available_instances.append(base_url)
                port = base_url.split(':')[-1]
                print(f"✅ 实例 {port} 可用")
            else:
                print(f"❌ 实例 {base_url.split(':')[-1]} 响应异常: {response.status_code}")
        except Exception as e:
            print(f"❌ 实例 {base_url.split(':')[-1]} 不可用: {e}")
    
    if len(available_instances) == 0:
        print(f"\n❌ 没有可用的 vLLM 实例")
        print("💡 请确保至少启动一个 vLLM 实例")
        return 1
    
    print(f"\n✅ 可用实例: {len(available_instances)} 个")
    
    # 创建测试 prompt
    test_prompt = "Document: " + rand_ascii(1000) + "\nSummary:"
    print(f"📄 测试 prompt 长度: {len(test_prompt)} 字符")
    
    try:
        # 根据测试类型执行不同的测试
        if test_type == "single" or len(available_instances) == 1:
            print("\n🔄 执行单实例测试...")
            results = test_single_instance_cache(available_instances[0], model, test_prompt)
        else:
            print("\n🔄 执行跨实例测试...")
            results = test_cross_instance_cache(available_instances, model, test_prompt)
        
        # 分析结果
        print(f"\n\033[32m🎯 === LMCache 测试分析结果 ===")
        
        # 找到冷启动时间
        cold_start = None
        for test_type, ttft, _ in results:
            if "cold" in test_type:
                cold_start = ttft
                break
        
        if cold_start is None:
            print("❌ 未找到冷启动数据")
            return 1
        
        print(f"🥶 冷启动时间: {cold_start:.3f}s")
        
        # 分析每个测试的性能
        for test_type, ttft, _ in results:
            if "cold" not in test_type:
                speedup = cold_start / ttft
                if speedup > 1.2:
                    print(f"✅ {test_type}: {ttft:.3f}s (加速: {speedup:.2f}x)")
                elif speedup > 1.1:
                    print(f"🟡 {test_type}: {ttft:.3f}s (加速: {speedup:.2f}x)")
                else:
                    print(f"🔴 {test_type}: {ttft:.3f}s (加速: {speedup:.2f}x)")
        
        # 保存详细结果
        out_path = Path(f"lmcache_test_results_{test_type}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for i, (test_type, ttft, result) in enumerate(results, 1):
                json.dump({
                    "test_index": i,
                    "test_type": test_type,
                    "ttft_seconds": ttft,
                    "usage": result.get('usage', {}),
                    "instance_port": test_type.split('_')[1] if '_' in test_type else "unknown"
                }, f)
                f.write("\n")
        
        print(f"\n📄 详细结果保存到: {out_path}")
        print("\033[0m")
        
        return 0
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
