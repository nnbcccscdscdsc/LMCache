#!/usr/bin/env python3
"""
数据集样本抽取脚本
从四个数据集中抽取指定数量的测试用例：
- 2WikiMQA: 200 个测试用例
- Musique: 150 个测试用例  
- SAMSum: 200 个测试用例
- MultiNews: 60 个抽样用例
python extract_dataset_samples.py --2wikimqa 200 --musique 150 --samsum 200 --multinews 60 --output-dir ./test_samples
"""

import os
import json
import random
from pathlib import Path
from datasets import load_dataset
import argparse

def set_random_seed(seed=42):
    """设置随机种子确保结果可复现"""
    random.seed(seed)
    print(f"🎲 设置随机种子: {seed}")

def extract_2wikimqa_samples(num_samples=200):
    """从 2WikiMQA 数据集中抽取样本"""
    print(f"📊 正在从 2WikiMQA 数据集中抽取 {num_samples} 个样本...")
    
    try:
        # 加载数据集
        dataset = load_dataset('presencesw/complexquestion_2WIKIMQA_1000')
        
        # 获取测试集
        if 'test' in dataset:
            test_data = dataset['test']
        elif 'validation' in dataset:
            test_data = dataset['validation']
        else:
            # 如果没有明确的测试集，使用第一个可用的分割
            test_data = dataset[list(dataset.keys())[0]]
        
        total_samples = len(test_data)
        print(f"   总样本数: {total_samples}")
        
        if total_samples < num_samples:
            print(f"   ⚠️  警告: 数据集只有 {total_samples} 个样本，少于请求的 {num_samples} 个")
            num_samples = total_samples
        
        # 随机抽取样本
        indices = random.sample(range(total_samples), num_samples)
        selected_samples = [test_data[i] for i in indices]
        
        print(f"   ✅ 成功抽取 {len(selected_samples)} 个样本")
        return selected_samples
        
    except Exception as e:
        print(f"   ❌ 加载 2WikiMQA 数据集失败: {e}")
        return []

def extract_musique_samples(num_samples=150):
    """从 Musique 数据集中抽取样本"""
    print(f"📊 正在从 Musique 数据集中抽取 {num_samples} 个样本...")
    
    try:
        # 加载数据集
        dataset = load_dataset('dgslibisey/MuSiQue')
        
        # 获取测试集
        if 'test' in dataset:
            test_data = dataset['test']
        elif 'validation' in dataset:
            test_data = dataset['validation']
        else:
            test_data = dataset[list(dataset.keys())[0]]
        
        total_samples = len(test_data)
        print(f"   总样本数: {total_samples}")
        
        if total_samples < num_samples:
            print(f"   ⚠️  警告: 数据集只有 {total_samples} 个样本，少于请求的 {num_samples} 个")
            num_samples = total_samples
        
        # 随机抽取样本
        indices = random.sample(range(total_samples), num_samples)
        selected_samples = [test_data[i] for i in indices]
        
        print(f"   ✅ 成功抽取 {len(selected_samples)} 个样本")
        return selected_samples
        
    except Exception as e:
        print(f"   ❌ 加载 Musique 数据集失败: {e}")
        return []

def extract_samsum_samples(num_samples=200):
    """从 SAMSum 数据集中抽取样本"""
    print(f"📊 正在从 SAMSum 数据集中抽取 {num_samples} 个样本...")
    
    try:
        # 加载数据集
        dataset = load_dataset('knkarthick/samsum')
        
        # 获取测试集
        if 'test' in dataset:
            test_data = dataset['test']
        elif 'validation' in dataset:
            test_data = dataset['validation']
        else:
            test_data = dataset[list(dataset.keys())[0]]
        
        total_samples = len(test_data)
        print(f"   总样本数: {total_samples}")
        
        if total_samples < num_samples:
            print(f"   ⚠️  警告: 数据集只有 {total_samples} 个样本，少于请求的 {num_samples} 个")
            num_samples = total_samples
        
        # 随机抽取样本
        indices = random.sample(range(total_samples), num_samples)
        selected_samples = [test_data[i] for i in indices]
        
        print(f"   ✅ 成功抽取 {len(selected_samples)} 个样本")
        return selected_samples
        
    except Exception as e:
        print(f"   ❌ 加载 SAMSum 数据集失败: {e}")
        return []

def extract_multinews_samples(num_samples=60):
    """从 MultiNews 数据集中抽取样本"""
    print(f"📊 正在从 MultiNews 数据集中抽取 {num_samples} 个样本...")
    
    try:
        # 加载数据集
        dataset = load_dataset('Awesome075/multi_news_parquet')
        
        # 获取测试集
        if 'test' in dataset:
            test_data = dataset['test']
        elif 'validation' in dataset:
            test_data = dataset['validation']
        else:
            test_data = dataset[list(dataset.keys())[0]]
        
        total_samples = len(test_data)
        print(f"   总样本数: {total_samples}")
        
        if total_samples < num_samples:
            print(f"   ⚠️  警告: 数据集只有 {total_samples} 个样本，少于请求的 {num_samples} 个")
            num_samples = total_samples
        
        # 随机抽取样本
        indices = random.sample(range(total_samples), num_samples)
        selected_samples = [test_data[i] for i in indices]
        
        print(f"   ✅ 成功抽取 {len(selected_samples)} 个样本")
        return selected_samples
        
    except Exception as e:
        print(f"   ❌ 加载 MultiNews 数据集失败: {e}")
        return []

def save_samples_to_file(samples, dataset_name, output_dir):
    """保存抽取的样本到文件"""
    output_path = output_dir / f"{dataset_name}_samples.json"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"   💾 样本已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"   ❌ 保存文件失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="从四个数据集中抽取指定数量的测试用例")
    parser.add_argument("--output-dir", default="./extracted_samples", help="输出目录 (默认: ./extracted_samples)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--2wikimqa", type=int, default=200, help="2WikiMQA 样本数量 (默认: 200)")
    parser.add_argument("--musique", type=int, default=150, help="Musique 样本数量 (默认: 150)")
    parser.add_argument("--samsum", type=int, default=200, help="SAMSum 样本数量 (默认: 200)")
    parser.add_argument("--multinews", type=int, default=60, help="MultiNews 样本数量 (默认: 60)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 开始抽取数据集样本")
    print("=" * 50)
    print(f"📁 输出目录: {output_dir}")
    print(f"🎲 随机种子: {args.seed}")
    print()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 抽取各数据集的样本
    results = {}
    
    # 1. 2WikiMQA
    print("1️⃣ 处理 2WikiMQA 数据集")
    print("-" * 30)
    wikimqa_samples = extract_2wikimqa_samples(getattr(args, '2wikimqa'))
    if wikimqa_samples:
        save_samples_to_file(wikimqa_samples, "2WikiMQA", output_dir)
        results["2WikiMQA"] = len(wikimqa_samples)
    print()
    
    # 2. Musique
    print("2️⃣ 处理 Musique 数据集")
    print("-" * 30)
    musique_samples = extract_musique_samples(getattr(args, 'musique'))
    if musique_samples:
        save_samples_to_file(musique_samples, "Musique", output_dir)
        results["Musique"] = len(musique_samples)
    print()
    
    # 3. SAMSum
    print("3️⃣ 处理 SAMSum 数据集")
    print("-" * 30)
    samsum_samples = extract_samsum_samples(getattr(args, 'samsum'))
    if samsum_samples:
        save_samples_to_file(samsum_samples, "SAMSum", output_dir)
        results["SAMSum"] = len(samsum_samples)
    print()
    
    # 4. MultiNews
    print("4️⃣ 处理 MultiNews 数据集")
    print("-" * 30)
    multinews_samples = extract_multinews_samples(getattr(args, 'multinews'))
    if multinews_samples:
        save_samples_to_file(multinews_samples, "MultiNews", output_dir)
        results["MultiNews"] = len(multinews_samples)
    print()
    
    # 总结
    print("📋 抽取结果总结")
    print("=" * 30)
    total_samples = 0
    for dataset_name, count in results.items():
        print(f"✅ {dataset_name}: {count} 个样本")
        total_samples += count
    
    print(f"📊 总计: {total_samples} 个样本")
    print(f"📁 所有文件保存在: {output_dir}")
    
    # 保存总结信息
    summary = {
        "extraction_date": str(Path().cwd()),
        "random_seed": args.seed,
        "output_directory": str(output_dir),
        "results": results,
        "total_samples": total_samples
    }
    
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📄 总结信息已保存到: {summary_path}")

if __name__ == "__main__":
    main()
