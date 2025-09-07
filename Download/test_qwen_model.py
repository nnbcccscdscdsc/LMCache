#!/usr/bin/env python3
"""
专门用于测试Qwen模型的脚本
支持Qwen2.5系列模型的本地测试
用法:
    python test_qwen_model.py <model_name> --cpu
    python test_qwen_model.py 1.5B --cpu
"""

import argparse
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_qwen_model(model_path: str, prompt: str, max_tokens: int = 128, use_gpu: bool = True):
    """
    测试Qwen模型
    
    Args:
        model_path: 模型路径（可以是主目录或snapshots路径）
        prompt: 输入提示
        max_tokens: 最大生成token数
        use_gpu: 是否使用GPU
    """
    print(f"🔍 正在加载Qwen模型: {model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查是否是snapshots路径，如果是则使用主目录
    if "snapshots" in model_path:
        # 从snapshots路径提取主目录路径
        main_path = model_path.split("/snapshots/")[0]
        if os.path.exists(main_path):
            model_path = main_path
            print(f"🔄 使用主目录路径: {model_path}")
        else:
            print(f"❌ 主目录路径不存在: {main_path}")
            return False
    
    try:
        # 加载tokenizer
        print("📝 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 设置pad_token（Qwen模型特有）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("🔧 设置pad_token为eos_token")
        
        # 检查CUDA是否可用
        if use_gpu and torch.cuda.is_available():
            print("🚀 使用GPU模式")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            print("💻 使用CPU模式")
            device_map = "cpu"
            torch_dtype = torch.float32
        
        # 加载模型（Qwen模型特有配置）
        print("🤖 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        print("✅ 模型加载成功，开始生成...")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成文本（Qwen模型优化参数）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,           # Qwen模型推荐温度
                top_p=0.8,                # Qwen模型推荐top_p
                top_k=50,                 # Qwen模型推荐top_k
                repetition_penalty=1.05,   # 轻微重复惩罚
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,           # Qwen模型支持缓存
                output_scores=False,
                return_dict_in_generate=False
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("🤖 Qwen模型输出:")
        print("=" * 50)
        print(generated_text)
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def list_qwen_models():
    """列出本地可用的Qwen模型"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    qwen_models = []
    
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            if item.startswith("models--Qwen"):
                model_path = os.path.join(cache_dir, item)
                if os.path.isdir(model_path):
                    qwen_models.append(model_path)
    
    if qwen_models:
        print("📋 本地可用的Qwen模型:")
        for i, model in enumerate(qwen_models, 1):
            print(f"  {i}. {model}")
    else:
        print("❌ 未找到本地Qwen模型")
    
    return qwen_models

def main():
    parser = argparse.ArgumentParser(description="测试Qwen模型")
    parser.add_argument("model_name", nargs="?")
    parser.add_argument("--prompt", default="你好，请介绍一下你自己", help="输入提示 (默认: 你好，请介绍一下你自己)")
    parser.add_argument("--max-tokens", type=int, default=128, help="最大生成token数 (默认: 128)")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU模式")
    parser.add_argument("--list", action="store_true", help="列出本地可用的Qwen模型")
    
    args = parser.parse_args()
    
    if args.list:
        list_qwen_models()
        return
    
    if not args.model_name:
        print("❌ 请提供模型名称")
        #目前支持的模型列表
        for model in qwen_models:
            print(model)
        print("💡 使用 --list 查看完整的模型路径")
        return
    
    # 根据模型名称构建路径
    model_path = f"/home/limingjie/.cache/huggingface/hub/models--Qwen--Qwen2.5-{args.model_name}-Instruct"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    success = test_qwen_model(
        model_path, 
        args.prompt, 
        max_tokens=args.max_tokens,
        use_gpu=not args.cpu
    )
    
    if success:
        print("🎉 Qwen模型测试成功！")
    else:
        print("💥 Qwen模型测试失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()
