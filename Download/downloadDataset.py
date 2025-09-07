#!/usr/bin/env python3
"""
下载任意 Hugging Face 数据集
使用方法: python downloadDataset.py <dataset_name>
例如: python downloadDataset.py wikitext
"""

import argparse
import sys
from datasets import load_dataset

# 下载数据集
def download_dataset(dataset_name: str):
    """下载指定的 Hugging Face 数据集"""
    print(f"⬇️ 开始下载数据集: {dataset_name}")
    try:
        # 默认下载全部子集
        dataset = load_dataset(dataset_name)
        print("✅ 数据集下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="下载 Hugging Face 数据集")
    parser.add_argument("dataset_name", help="数据集名称 (例如: wikitext 或 glue)")

    args = parser.parse_args()
    
    # 验证格式（数据集名一般不用组织名，但某些需要 org/dataset）
    if "/" not in args.dataset_name and not args.dataset_name.isidentifier():
        print("⚠️ 数据集名称可能不符合 Hugging Face 格式")
    
    download_dataset(args.dataset_name)

if __name__ == "__main__":
    main()
