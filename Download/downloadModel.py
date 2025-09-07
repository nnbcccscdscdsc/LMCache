#!/usr/bin/env python3
"""
下载任意 Hugging Face 模型和 tokenizer
使用方法: python downloadModel.py <model_name>
例如: python downloadModel.py 01-ai/Yi-34B-200K
"""

import argparse
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# 下载模型
def download_model(model_name: str):
    """下载指定的 Hugging Face 模型和 tokenizer"""
    print(f"⬇️ 开始下载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    print("✅ 模型下载完成！")

def main():
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型")
    parser.add_argument("model_name", help="模型名称 (例如: 01-ai/Yi-34B-200K)")

    args = parser.parse_args()
    
    # 验证模型名称格式
    if "/" not in args.model_name:
        print("❌ 模型名称格式错误！请使用 'organization/model-name' 格式")
        sys.exit(1)
    
    download_model(args.model_name)

if __name__ == "__main__":
    main()
