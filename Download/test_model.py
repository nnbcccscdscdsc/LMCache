#!/usr/bin/env python3
"""
通用 HuggingFace 模型测试脚本
支持本地模型 / HuggingFace Hub 模型
用法:
    python test_model.py meta-llama/Llama-2-7b-chat-hf --cpu
    python test_model.py Qwen/Qwen2.5-1.5B-Instruct --prompt "你好"
    python test_model.py /path/to/local/model --local
    python test_model.py --list  # 列出本地可用模型
    python test_model.py ~/.cache/huggingface/hub/models--Yukang--LongAlpaca-70B-16k/snapshots/594d3c7ba4fa3ea0720b9918820ef73dfcc5ab9b --local --cpu --prompt "Hello" --max-tokens 10
    python test_model.py ~/.cache/huggingface/hub/models--01-ai--Yi-34B-200K/snapshots/e0ae1afac6b69f604556efd441ab4befafb2a835 --local --cpu --prompt "Hello" --max-tokens 20
    python test_model.py ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca --local --cpu --prompt "Hello" --max-tokens 20
"""

import argparse
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def test_model(model_path: str, prompt: str, max_tokens: int = 128, use_gpu: bool = True, local_only: bool = False):
    """
    通用模型测试函数
    Args:
        model_path: 模型路径（本地路径或 HuggingFace Hub 名称）
        prompt: 输入提示
        max_tokens: 最大生成token数
        use_gpu: 是否使用GPU
        local_only: 是否只使用本地文件
    """
    print(f"🔍 正在加载模型: {model_path}")
    
    # 改进的路径验证逻辑
    if local_only or os.path.exists(model_path):
        # 本地路径验证
        if not os.path.exists(model_path):
            print(f"❌ 本地模型路径不存在: {model_path}")
            return False
        
        # 检查是否是snapshots路径，如果是则直接使用（不转换）
        if "snapshots" in model_path:
            print(f"📁 检测到 snapshots 路径，直接使用: {model_path}")
            # 不转换路径，直接使用 snapshots 路径
    else:
        # HuggingFace Hub 模型名称验证
        if "/" not in model_path:
            print(f"⚠️  警告: 模型名称 '{model_path}' 可能不是有效的 HuggingFace Hub 格式")
            print("💡 建议使用格式: 用户名/模型名 (例如: Qwen/Qwen2.5-1.5B-Instruct)")

    try:
        # 加载 tokenizer
        print("📝 加载 tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=local_only
            )
        except Exception as tokenizer_error:
            print(f"⚠️  tokenizer 加载失败: {tokenizer_error}")
            print("🔄 尝试使用备用 tokenizer...")
            # 对于有问题的模型，尝试使用 Llama tokenizer
            try:
                from transformers import LlamaTokenizer
                # 确保 model_path 是字符串
                if not isinstance(model_path, str):
                    model_path = str(model_path)
                tokenizer = LlamaTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=local_only
                )
                print("✅ 使用 Llama tokenizer 成功")
            except Exception as llama_error:
                print(f"❌ Llama tokenizer 也失败: {llama_error}")
                # 尝试使用 GPT2 tokenizer 作为最后的备用方案
                try:
                    from transformers import GPT2Tokenizer
                    print("🔄 尝试使用 GPT2 tokenizer...")
                    tokenizer = GPT2Tokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=local_only
                    )
                    print("✅ 使用 GPT2 tokenizer 成功")
                except Exception as gpt2_error:
                    print(f"❌ GPT2 tokenizer 也失败: {gpt2_error}")
                    raise tokenizer_error

        # 设置 pad_token（兼容大多数模型）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("🔧 设置 pad_token 为 eos_token")

        # 检查 GPU/CPU
        if use_gpu and torch.cuda.is_available():
            print("🚀 使用 GPU 模式")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            print("💻 使用 CPU 模式")
            device_map = "cpu"
            torch_dtype = torch.float32

        # 加载模型
        print("🤖 加载模型...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                local_files_only=local_only
            )
        except Exception as model_error:
            print(f"⚠️  模型加载失败: {model_error}")
            print("🔄 尝试使用 Llama 模型架构...")
            try:
                from transformers import LlamaForCausalLM
                model = LlamaForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    local_files_only=local_only
                )
                print("✅ 使用 Llama 模型架构成功")
            except Exception as llama_model_error:
                print(f"❌ Llama 模型架构也失败: {llama_model_error}")
                raise model_error

        print("✅ 模型加载成功，开始生成...")

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成文本（优化的参数）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,  # 稍微降低top_p提高质量
                top_k=50,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                output_scores=False,
                return_dict_in_generate=False
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("🤖 模型输出:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def list_local_models():
    """列出本地可用的模型"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    local_models = []
    
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if item.startswith("models--"):
                model_path = os.path.join(cache_dir, item)
                if os.path.isdir(model_path):
                    # 提取模型名称
                    model_name = item.replace("models--", "").replace("--", "/")
                    local_models.append((model_name, model_path))
    
    if local_models:
        print("📋 本地可用的模型:")
        for i, (model_name, model_path) in enumerate(local_models, 1):
            print(f"  {i}. {model_name}")
            print(f"     路径: {model_path}")
    else:
        print("❌ 未找到本地模型")
        print("💡 提示: 使用 HuggingFace Hub 模型名称会自动下载到本地")
    
    return local_models


def main():
    parser = argparse.ArgumentParser(description="通用 HuggingFace 模型测试脚本")
    parser.add_argument("model_name", nargs="?", help="模型名称或本地路径")
    parser.add_argument("--prompt", default="你好，请介绍一下你自己", help="输入提示 (默认: 你好，请介绍一下你自己)")
    parser.add_argument("--max-tokens", type=int, default=128, help="最大生成 token 数 (默认: 128)")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 模式")
    parser.add_argument("--local", action="store_true", help="强制使用本地文件模式")
    parser.add_argument("--list", action="store_true", help="列出本地可用的模型")

    args = parser.parse_args()
    
    # 处理 --list 参数
    if args.list:
        list_local_models()
        return
    
    # 检查是否提供了模型名称
    if not args.model_name:
        print("❌ 请提供模型名称")
        print("💡 使用示例:")
        print("  python test_model.py Qwen/Qwen2.5-1.5B-Instruct --cpu")
        print("  python test_model.py /path/to/local/model --local")
        print("  python test_model.py --list  # 查看本地可用模型")
        return

    # 如果使用 --local 参数，将 HuggingFace Hub 名称转换为本地路径
    model_path = args.model_name
    if args.local and "/" in args.model_name and not os.path.exists(args.model_name):
        # 将 HuggingFace Hub 名称转换为本地缓存路径
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_name = f"models--{args.model_name.replace('/', '--')}"
        model_path = os.path.join(cache_dir, model_cache_name)
        print(f"🔄 转换 HuggingFace Hub 名称为本地路径: {model_path}")

    success = test_model(
        model_path,
        args.prompt,
        max_tokens=args.max_tokens,
        use_gpu=not args.cpu,
        local_only=args.local
    )

    if success:
        print("🎉 模型测试成功！")
    else:
        print("💥 模型测试失败！")
        print("💡 故障排除建议:")
        print("  1. 检查模型名称是否正确")
        print("  2. 确保网络连接正常（如果使用 HuggingFace Hub）")
        print("  3. 检查本地路径是否存在（如果使用 --local）")
        print("  4. 使用 --list 查看本地可用模型")
        sys.exit(1)


if __name__ == "__main__":
    main()
