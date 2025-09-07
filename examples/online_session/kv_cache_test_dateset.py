#!/usr/bin/env python3
"""
简化版 LMCache 数据集测试
- 仅支持本地 vLLM 实例 (http://localhost:8000/v1)
- 仅支持从本机缓存文件加载数据集 (.json 或 .txt)
"""

import requests
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def load_dataset_from_file(file_path: str) -> List[str]:
    """从本地文件加载数据集 (.json 或 .txt)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'prompts' in data:
                return data['prompts']
            else:
                raise ValueError("JSON 文件格式不正确")
        elif file_path.endswith('.txt'):
            return [line.strip() for line in f if line.strip()]
        else:
            raise ValueError("不支持的文件格式，请使用 .json 或 .txt 文件")


def measure_ttft(base_url: str, model: str, prompt: str, tag: str) -> float:
    """测量 TTFT"""
    print(f"📤 {tag}...")
    start = time.perf_counter()
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
    print(f"🕐 TTFT = {ttft:.3f}s")
    return ttft


def test_dataset(base_url: str, model: str, dataset: List[str]):
    results = []

    for i, prompt in enumerate(dataset, 1):
        print(f"\n📝 Prompt {i}/{len(dataset)} (长度 {len(prompt)} 字符)")

        # 冷启动
        ttft_cold = measure_ttft(base_url, model, prompt, "冷启动")
        # 第一次缓存复用
        ttft_cached1 = measure_ttft(base_url, model, prompt, "第一次缓存复用")
        # 第二次缓存复用
        ttft_cached2 = measure_ttft(base_url, model, prompt, "第二次缓存复用")

        print(f"🚀 加速比: {ttft_cold/ttft_cached1:.2f}x, {ttft_cold/ttft_cached2:.2f}x")

        results.append((ttft_cold, ttft_cached1, ttft_cached2))

        time.sleep(1)  # 避免请求过快

    # 统计平均
    avg_cold = sum(r[0] for r in results) / len(results)
    avg_cached1 = sum(r[1] for r in results) / len(results)
    avg_cached2 = sum(r[2] for r in results) / len(results)

    print("\n🎯 测试结果汇总")
    print(f"🥶 平均冷启动: {avg_cold:.3f}s")
    print(f"🔥 平均缓存1: {avg_cached1:.3f}s")
    print(f"🔄 平均缓存2: {avg_cached2:.3f}s")
    print(f"🚀 平均加速比: {avg_cold/avg_cached1:.2f}x / {avg_cold/avg_cached2:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser(description="简化版 LMCache 测试")
    parser.add_argument("--dataset-file", type=str, required=True,
                        help="本地数据集文件路径 (.json 或 .txt)")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="模型名称 (默认: facebook/opt-125m)")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM 端口 (默认: 8000)")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = f"http://localhost:{args.port}/v1"

    print(f"🧪 LMCache 测试 (模型={args.model}, 实例={base_url})")

    dataset = load_dataset_from
#!/usr/bin/env python3
"""
从本地 Hugging Face 缓存加载数据集的 LMCache 测试脚本
避免使用 datasets 库，直接从缓存文件读取数据
"""

import requests
import json
import time
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple


def find_hf_cache_dir() -> Path:
    """查找 Hugging Face 缓存目录"""
    hf_cache_home = os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
    cache_dir = Path(hf_cache_home) / "hub"
    return cache_dir


def find_dataset_cache(dataset_name: str) -> Path:
    """查找数据集缓存目录"""
    cache_dir = find_hf_cache_dir()
    print(f"📁 搜索缓存目录: {cache_dir}")
    
    # 转换数据集名称为缓存目录名
    cache_name = dataset_name.replace("/", "--")
    
    for item in cache_dir.iterdir():
        if item.is_dir() and cache_name.lower() in item.name.lower():
            print(f"✅ 找到缓存目录: {item}")
            return item
    
    print(f"❌ 未找到数据集缓存: {dataset_name}")
    print(f"💡 请先下载数据集或检查缓存目录")
    return None


def load_from_cache_file(cache_dir: Path, max_samples: int = 20) -> List[str]:
    """从缓存文件加载数据"""
    # 优先查找 snapshots 目录中的数据文件
    snapshots_dir = cache_dir / "snapshots"
    data_files = []
    
    if snapshots_dir.exists():
        for pattern in ["*.json", "*.jsonl", "*.parquet"]:
            data_files.extend(snapshots_dir.rglob(pattern))
    
    # 如果 snapshots 中没有找到，再查找整个目录
    if not data_files:
        for pattern in ["*.json", "*.jsonl", "*.parquet"]:
            data_files.extend(cache_dir.rglob(pattern))
    
    if not data_files:
        print(f"❌ 未找到数据文件")
        return []
    
    # 优先选择 parquet 文件，然后是 jsonl，最后是 json
    data_files.sort(key=lambda x: (x.suffix != '.parquet', x.suffix != '.jsonl', x.suffix != '.json'))
    data_file = data_files[0]
    print(f"📄 使用数据文件: {data_file}")
    
    prompts = []
    try:
        if data_file.suffix == '.json':
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # 尝试找到数据列表
                    for key in ['train', 'data', 'examples']:
                        if key in data and isinstance(data[key], list):
                            items = data[key]
                            break
                    else:
                        items = [data]
                else:
                    items = [data]
        elif data_file.suffix == '.jsonl':
            items = []
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        elif data_file.suffix == '.parquet':
            try:
                import pandas as pd
                df = pd.read_parquet(data_file)
                items = df.to_dict('records')
            except ImportError:
                print("❌ 需要安装 pandas 来读取 parquet 文件: pip install pandas")
                return []
        else:
            print(f"❌ 不支持的文件格式: {data_file.suffix}")
            return []
        
        # 提取 prompts
        for i, item in enumerate(items):
            if i >= max_samples:
                break
                
            if isinstance(item, str):
                prompt = item
            elif isinstance(item, dict):
                # 根据数据集类型提取 prompt
                if 'text' in item:
                    prompt = item['text']
                elif 'question' in item and 'context' in item:
                    prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
                elif 'input' in item:
                    prompt = item['input']
                elif 'prompt' in item:
                    prompt = item['prompt']
                elif 'dialogue' in item:
                    prompt = f"Dialogue: {item['dialogue']}\nSummary:"
                elif 'article' in item:
                    prompt = f"Article: {item['article']}\nSummary:"
                else:
                    # 尝试使用第一个字符串字段
                    for key, value in item.items():
                        if isinstance(value, str) and len(value) > 50:
                            prompt = value
                            break
                    else:
                        continue
            else:
                continue
            
            # 限制长度
            if len(prompt) > 2000:
                prompt = prompt[:2000] + "..."
            
            prompts.append(prompt)
        
        print(f"✅ 成功加载 {len(prompts)} 个样本")
        return prompts
        
    except Exception as e:
        print(f"❌ 加载数据文件失败: {e}")
        return []


def measure_ttft(base_url: str, model: str, prompt: str, tag: str) -> float:
    """测量 TTFT"""
    print(f"📤 {tag}...")
    start = time.perf_counter()
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
    print(f"🕐 TTFT = {ttft:.3f}s")
    return ttft


def test_dataset(base_url: str, model: str, dataset: List[str]):
    """测试数据集性能"""
    results = []

    for i, prompt in enumerate(dataset, 1):
        print(f"\n📝 Prompt {i}/{len(dataset)} (长度 {len(prompt)} 字符)")

        # 冷启动
        ttft_cold = measure_ttft(base_url, model, prompt, "冷启动")
        # 第一次缓存复用
        ttft_cached1 = measure_ttft(base_url, model, prompt, "第一次缓存复用")
        # 第二次缓存复用
        ttft_cached2 = measure_ttft(base_url, model, prompt, "第二次缓存复用")

        print(f"🚀 加速比: {ttft_cold/ttft_cached1:.2f}x, {ttft_cold/ttft_cached2:.2f}x")

        results.append((ttft_cold, ttft_cached1, ttft_cached2))

        time.sleep(1)  # 避免请求过快

    # 统计平均
    avg_cold = sum(r[0] for r in results) / len(results)
    avg_cached1 = sum(r[1] for r in results) / len(results)
    avg_cached2 = sum(r[2] for r in results) / len(results)

    print("\n🎯 测试结果汇总")
    print(f"🥶 平均冷启动: {avg_cold:.3f}s")
    print(f"🔥 平均缓存1: {avg_cached1:.3f}s")
    print(f"🔄 平均缓存2: {avg_cached2:.3f}s")
    print(f"🚀 平均加速比: {avg_cold/avg_cached1:.2f}x / {avg_cold/avg_cached2:.2f}x")


def list_available_datasets():
    """列出可用的数据集"""
    cache_dir = find_hf_cache_dir()
    print(f"📋 可用的 Hugging Face 数据集缓存:")
    print(f"📁 缓存目录: {cache_dir}")
    
    if not cache_dir.exists():
        print("❌ 缓存目录不存在")
        return
    
    datasets = []
    for item in cache_dir.iterdir():
        if item.is_dir() and item.name.startswith("datasets--"):
            dataset_name = item.name.replace("datasets--", "").replace("--", "/")
            datasets.append(dataset_name)
    
    if datasets:
        for dataset in sorted(datasets):
            print(f"   📊 {dataset}")
    else:
        print("❌ 未找到任何数据集缓存")


def parse_args():
    parser = argparse.ArgumentParser(description="从 Hugging Face 缓存加载数据集的 LMCache 测试")
    parser.add_argument("--dataset", type=str, help="Hugging Face 数据集名称 (例如: knkarthick/samsum)")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B-Instruct",
                        help="模型名称 (默认: Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--port", type=int, default=8002,
                        help="vLLM 端口 (默认: 8002)")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="最大样本数 (默认: 10)")
    parser.add_argument("--list", action="store_true",
                        help="列出可用的数据集")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.list:
        list_available_datasets()
        return
    
    if not args.dataset:
        print("❌ 请指定数据集名称，使用 --dataset 参数")
        print("💡 使用 --list 查看可用数据集")
        return
    
    base_url = f"http://localhost:{args.port}/v1"
    print(f"🧪 LMCache 测试 (模型={args.model}, 实例={base_url})")
    print(f"📊 数据集: {args.dataset}")
    
    # 查找数据集缓存
    cache_dir = find_dataset_cache(args.dataset)
    if not cache_dir:
        return
    
    # 加载数据
    dataset = load_from_cache_file(cache_dir, args.max_samples)
    if not dataset:
        return
    
    # 测试数据集
    test_dataset(base_url, args.model, dataset)


if __name__ == "__main__":
    main()
