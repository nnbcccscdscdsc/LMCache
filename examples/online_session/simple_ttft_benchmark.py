#!/usr/bin/env python3
"""
简化的 TTFT 基准测试脚本
避免下载 tokenizer，直接使用字符估算
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

def build_chat(system_doc: str, user_prompt: str) -> list[dict]:
    """构建聊天消息"""
    return [
        {"role": "user", "content": f"I've got a document:\n```\n{system_doc}\n```"},
        {"role": "assistant", "content": "I've got your document."},
        {"role": "user", "content": user_prompt},
    ]

def ttft_stream_requests(base_url: str, model: str, prompt: str) -> tuple[float, str]:
    """测量 TTFT 的流式请求"""
    start = time.perf_counter()
    
    try:
        response = requests.post(
            f"{base_url}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.0,
                "stream": True,
                "max_tokens": 1024
            },
            stream=True,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        first_tok_t = None
        buf = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 去掉 'data: ' 前缀
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk['choices'][0]['delta']
                        if 'content' in delta and delta['content']:
                            if first_tok_t is None:
                                first_tok_t = time.perf_counter()
                            print(delta['content'], end="", flush=True)
                            buf += delta['content']
                    except json.JSONDecodeError:
                        continue
        
        print()  # newline after streaming
        if first_tok_t is None:
            raise RuntimeError("no tokens returned")
        return first_tok_t - start, buf
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        raise

def flush_kv_cache(base_url: str, model: str) -> None:
    """通过发送填充提示来刷新 KV 缓存"""
    filler_prompt = f"I've got a document:\n```\n{rand_ascii(100000)}\n```\nI've got your document.\nnoop"
    for i in range(10):  # 发送 10 个填充提示
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
            print(f"Cache flush prompt {i+1}/10 sent")
        except Exception as e:
            print(f"Warning: Cache flush prompt {i+1} failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="简化的 TTFT 基准测试")
    parser.add_argument("--api_base", default="http://localhost:8000/v1",
                       help="API base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--model", default="facebook/opt-125m",
                       help="Model name (default: facebook/opt-125m)")
    parser.add_argument("--prompt", default="Summarize this text",
                       help="User prompt (default: Summarize this text)")
    parser.add_argument("--flush_cache", action="store_true",
                       help="Flush KV cache between runs")
    parser.add_argument("--out", default="simple_benchmark.jsonl",
                       help="Output file (default: simple_benchmark.jsonl)")
    
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    print(f"API base: {args.api_base}")
    
    # 生成文档（使用字符估算，避免 tokenizer）
    doc_length = 50000  # 约 12500 tokens (4 chars/token)
    doc = rand_ascii(doc_length)
    
    out_path = Path(args.out)
    out_path.write_text("", encoding="utf-8")  # 清空文件
    
    # ---------------- 第一次运行：基准 TTFT ----------------
    print("\n=== Run 1: baseline TTFT ===")
    base_prompt = f"I've got a document:\n```\n{doc}\n```\nI've got your document.\n{args.prompt}"
    
    try:
        ttft1, gen1 = ttft_stream_requests(args.api_base, args.model, base_prompt)
        print(f"\033[33mTTFT_1 = {ttft1:.3f}s\033[0m")
        
        # 记录结果
        with out_path.open("a", encoding="utf-8") as f:
            json.dump({
                "run_index": 1,
                "context_tokens": doc_length // 4,  # 粗略估算
                "ttft_seconds": ttft1,
            }, f)
            f.write("\n")
        
        # ---------------- 第二次运行：缓存命中 ----------------
        print("\n=== Run 2: cache hit TTFT ===")
        
        if args.flush_cache:
            print("Flushing KV-cache...")
            flush_kv_cache(args.api_base, args.model)
        else:
            print("(no KV-cache flush requested)")
        
        ttft2, gen2 = ttft_stream_requests(args.api_base, args.model, base_prompt)
        print(f"\033[33mTTFT_2 = {ttft2:.3f}s\033[0m")
        
        # 记录结果
        with out_path.open("a", encoding="utf-8") as f:
            json.dump({
                "run_index": 2,
                "context_tokens": doc_length // 4,
                "ttft_seconds": ttft2,
            }, f)
            f.write("\n")
        
        # 计算加速比
        speedup = ttft1 / ttft2 if ttft2 > 0 else float('inf')
        print(f"\n\033[32m=== 结果总结 ===")
        print(f"冷启动 TTFT: {ttft1:.3f}s")
        print(f"缓存命中 TTFT: {ttft2:.3f}s")
        print(f"加速比: {speedup:.2f}x\033[0m")
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")

if __name__ == "__main__":
    main()
