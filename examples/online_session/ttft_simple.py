#!/usr/bin/env python3
"""
简单的 TTFT 基准测试 - 专门为 facebook/opt-125m 设计
避免复杂的流式解析，使用非流式请求测量 TTFT
"""

import requests
import json
import time
import random
import string
from pathlib import Path

def rand_ascii(n: int) -> str:
    """生成随机 ASCII 字符串"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))

def measure_ttft(base_url: str, model: str, prompt: str) -> tuple[float, dict]:
    """测量 TTFT - 使用非流式请求"""
    print(f"📤 发送请求...")
    start = time.perf_counter()
    
    try:
        response = requests.post(
            f"{base_url}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 50,  # 生成更多 tokens
                "stream": False   # 使用非流式请求
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
        
        print(f"📥 响应收到: {generated_text[:50]}...")
        print(f"📊 Token 统计: {usage}")
        
        return ttft, result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")
        raise

def flush_kv_cache(base_url: str, model: str) -> None:
    """通过发送大量填充提示来刷新 KV 缓存"""
    print("🔄 开始刷新 KV 缓存...")
    
    for i in range(5):  # 发送 5 个填充提示
        filler_prompt = f"Random filler text: {rand_ascii(10000)}"
        try:
            response = requests.post(
                f"{base_url}/completions",
                json={
                    "model": model,
                    "prompt": filler_prompt,
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "stream": False
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            print(f"  ✅ 填充提示 {i+1}/5 发送成功")
        except Exception as e:
            print(f"  ⚠️ 填充提示 {i+1}/5 失败: {e}")
    
    print("🔄 KV 缓存刷新完成")

def main():
    base_url = "http://localhost:8000/v1"
    model = "facebook/opt-125m"
    
    print("🚀 TTFT 基准测试开始")
    print(f"📍 API base: {base_url}")
    print(f"🤖 Model: {model}")
    print("=" * 60)
    
    # 生成测试文档（使用较短的文档来避免超出 context 限制）
    doc_length = 2000  # 约 500 tokens (4 chars/token)
    doc = rand_ascii(doc_length)
    prompt = f"Document: {doc}\nSummary:"
    
    print(f"📄 文档长度: {doc_length} 字符 (约 {doc_length//4} tokens)")
    
    # 创建输出文件
    out_path = Path("ttft_results.jsonl")
    out_path.write_text("", encoding="utf-8")
    
    try:
        # ---------------- 第一次运行：冷启动 ----------------
        print("\n🥶 === Run 1: 冷启动 TTFT ===")
        ttft1, result1 = measure_ttft(base_url, model, prompt)
        print(f"\033[33m🕐 TTFT_1 = {ttft1:.3f}s\033[0m")
        
        # 记录结果
        with out_path.open("a", encoding="utf-8") as f:
            json.dump({
                "run_index": 1,
                "type": "cold_start",
                "context_chars": doc_length,
                "context_tokens_est": doc_length // 4,
                "ttft_seconds": ttft1,
                "usage": result1.get('usage', {})
            }, f)
            f.write("\n")
        
        # 等待一下
        print("⏳ 等待 3 秒...")
        time.sleep(3)
        
        # ---------------- 第二次运行：缓存命中 ----------------
        print("\n🔥 === Run 2: 缓存命中 TTFT ===")
        ttft2, result2 = measure_ttft(base_url, model, prompt)
        print(f"\033[33m🕐 TTFT_2 = {ttft2:.3f}s\033[0m")
        
        # 记录结果
        with out_path.open("a", encoding="utf-8") as f:
            json.dump({
                "run_index": 2,
                "type": "cache_hit",
                "context_chars": doc_length,
                "context_tokens_est": doc_length // 4,
                "ttft_seconds": ttft2,
                "usage": result2.get('usage', {})
            }, f)
            f.write("\n")
        
        # 计算加速比
        speedup = ttft1 / ttft2 if ttft2 > 0 else float('inf')
        improvement = ((ttft1 - ttft2) / ttft1) * 100 if ttft1 > 0 else 0
        
        print(f"\n\033[32m🎯 === 结果总结 ===")
        print(f"🥶 冷启动 TTFT: {ttft1:.3f}s")
        print(f"🔥 缓存命中 TTFT: {ttft2:.3f}s")
        print(f"🚀 加速比: {speedup:.2f}x")
        print(f"📈 性能提升: {improvement:.1f}%")
        print(f"📄 结果保存到: {out_path}")
        print("\033[0m")
        
        # 显示是否有明显的缓存效果
        if speedup > 1.5:
            print("✅ LMCache 缓存效果明显！")
        elif speedup > 1.1:
            print("🟡 LMCache 有一定缓存效果")
        else:
            print("🔴 LMCache 缓存效果不明显，可能需要检查配置")
            
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
