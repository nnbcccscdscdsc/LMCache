#!/usr/bin/env python3
"""
下载和预处理 complexquestion_2WIKIMQA_1000 数据集
用于 LMCache 性能测试
"""

import json
import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖库是否可用"""
    print("🔍 检查依赖库...")
    
    # 检查 NumPy 版本
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"✅ NumPy: {numpy_version}")
        
        # 检查 NumPy 版本兼容性
        if numpy_version.startswith('2.'):
            print("⚠️  警告: 检测到 NumPy 2.x 版本")
            print("💡 建议: 降级到 NumPy 1.x 以避免兼容性问题")
            print("   命令: pip install 'numpy<2.0'")
    except ImportError:
        print("❌ NumPy 未安装")
        return False
    
    # 检查 datasets 库
    try:
        from datasets import load_dataset
        print("✅ datasets 库可用")
        return True
    except ImportError as e:
        print(f"❌ datasets 库导入失败: {e}")
        print("💡 请安装: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ datasets 库存在但有问题: {e}")
        if "NumPy" in str(e) or "array_api" in str(e):
            print("🔧 这是 NumPy 兼容性问题")
            print("💡 解决方案: pip install 'numpy<2.0'")
        return False

def download_complex_qa_dataset():
    """下载复杂问答数据集"""
    print("📥 开始下载 complexquestion_2WIKIMQA_1000 数据集...")
    
    # 首先检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，无法继续")
        return None
    
    try:
        # 导入 datasets 库
        from datasets import load_dataset
        
        # 设置 Hugging Face 配置
        print("⚙️ 配置 Hugging Face 连接...")
        
        # 尝试设置镜像（如果可用）
        try:
            os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')
            print(f"🌐 使用端点: {os.environ.get('HF_ENDPOINT')}")
        except:
            pass
        
        # 下载数据集
        print("🔄 正在从 Hugging Face 下载数据集...")
        dataset = load_dataset('presencesw/complexquestion_2WIKIMQA_1000')
        
        print(f"✅ 数据集下载成功!")
        print(f"📊 数据集信息:")
        print(f"   - 训练集大小: {len(dataset['train'])} 行")
        print(f"   - 列名: {dataset['train'].column_names}")
        
        # 显示示例数据
        print(f"\n📝 示例数据:")
        example = dataset['train'][0]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集下载失败: {e}")
        
        # 根据错误类型提供具体的诊断
        error_str = str(e).lower()
        
        if "numpy" in error_str or "array_api" in error_str:
            print(f"\n🔧 这是 NumPy 兼容性问题!")
            print(f"💡 解决方案:")
            print(f"   1. 降级 NumPy: pip install 'numpy<2.0'")
            print(f"   2. 或者升级相关模块: pip install --upgrade pandas numexpr bottleneck")
            print(f"   3. 或者使用 conda 环境: conda install numpy pandas")
        
        elif "connection" in error_str or "timeout" in error_str:
            print(f"\n🔧 这是网络连接问题!")
            print(f"💡 解决方案:")
            print(f"   1. 检查代理设置: env | grep -i proxy")
            print(f"   2. 设置正确的代理: export http_proxy='http://代理地址:端口'")
            print(f"   3. 或者禁用代理: export http_proxy='' && export https_proxy=''")
        
        elif "not found" in error_str or "doesn't exist" in error_str:
            print(f"\n🔧 这是数据集路径问题!")
            print(f"💡 解决方案:")
            print(f"   1. 检查数据集名称是否正确")
            print(f"   2. 确认数据集是否公开可用")
        
        else:
            print(f"\n🔧 未知错误类型")
            print(f"💡 建议操作:")
            print(f"   1. 检查网络连接")
            print(f"   2. 检查代理设置")
            print(f"   3. 尝试使用 VPN")
        
        return None

def extract_prompts_for_testing(dataset, max_prompts=20):
    """提取用于测试的 prompts"""
    print(f"\n�� 提取测试 prompts (最多 {max_prompts} 个)...")
    
    prompts = []
    train_data = dataset['train']
    
    for i in range(min(max_prompts, len(train_data))):
        item = train_data[i]
        
        # 构建测试 prompt
        # 优先使用 user_prompt，因为它包含了完整的上下文和问题
        if 'user_prompt' in item and item['user_prompt']:
            prompt = item['user_prompt']
        elif 'complex_question' in item and item['complex_question']:
            # 如果没有 user_prompt，使用 complex_question 构建简单 prompt
            prompt = f"Question: {item['complex_question']}\nAnswer:"
        else:
            continue
        
        # 限制 prompt 长度，避免超出模型上下文窗口
        if len(prompt) > 1500:  # facebook/opt-125m 上下文窗口是 2048 tokens，留出安全边距
            prompt = prompt[:1500] + "..."
        
        prompts.append({
            "index": i + 1,
            "original_length": len(item.get('user_prompt', '')),
            "truncated_length": len(prompt),
            "prompt": prompt,
            "complex_question": item.get('complex_question', ''),
            "answer": item.get('gpt_answer', ''),
            "entities": item.get('entities', []),
            "triplets": item.get('triplets', [])
        })
    
    print(f"✅ 成功提取 {len(prompts)} 个测试 prompts")
    return prompts

def save_prompts_to_files(prompts):
    """保存 prompts 到不同格式的文件"""
    print(f"\n💾 保存 prompts 到文件...")
    
    # 创建输出目录
    output_dir = Path("dataset_prompts")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 保存为 JSON 格式
    json_file = output_dir / "complex_qa_prompts.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset_name": "complexquestion_2WIKIMQA_1000",
            "source": "https://huggingface.co/datasets/presencesw/complexquestion_2WIKIMQA_1000",
            "description": "1000个复杂问答对，包含多步推理问题",
            "total_prompts": len(prompts),
            "prompts": prompts
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 格式保存到: {json_file}")
    
    # 2. 保存为 TXT 格式（每行一个 prompt）
    txt_file = output_dir / "complex_qa_prompts.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        for prompt_data in prompts:
            f.write(prompt_data['prompt'] + '\n')
    print(f"✅ TXT 格式保存到: {txt_file}")
    
    # 3. 保存为 JSONL 格式（每行一个 JSON 对象）
    jsonl_file = output_dir / "complex_qa_prompts.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for prompt_data in prompts:
            json.dump(prompt_data, f, ensure_ascii=False)
            f.write('\n')
    print(f"✅ JSONL 格式保存到: {jsonl_file}")
    
    return output_dir

def create_test_config():
    """创建测试配置文件"""
    print(f"\n⚙️ 创建测试配置文件...")
    
    config = {
        "dataset_info": {
            "name": "complexquestion_2WIKIMQA_1000",
            "source": "https://huggingface.co/datasets/presencesw/complexquestion_2WIKIMQA_1000",
            "description": "1000个复杂问答对，包含多步推理问题",
            "note": "从 Hugging Face 直接下载的真实数据集"
        },
        "test_config": {
            "max_prompts": 20,
            "max_prompt_length": 1500,
            "model": "facebook/opt-125m",
            "test_types": ["cold_start", "cached_first", "cached_second"]
        },
        "usage_instructions": {
            "json_format": "使用 --dataset custom --dataset-file dataset_prompts/complex_qa_prompts.json",
            "txt_format": "使用 --dataset custom --dataset-file dataset_prompts/complex_qa_prompts.txt",
            "jsonl_format": "使用 --dataset custom --dataset-file dataset_prompts/complex_qa_prompts.jsonl"
        },
        "dataset_features": [
            "complex_question - 复杂问题",
            "user_prompt - 用户提示（包含上下文）",
            "gpt_answer - GPT 答案",
            "entities - 实体序列",
            "triplets - 三元组序列"
        ]
    }
    
    config_file = Path("dataset_prompts/test_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 测试配置文件保存到: {config_file}")
    return config_file

def main():
    """主函数"""
    print("🧪 Complex QA 数据集下载工具")
    print("=" * 60)
    print("📥 目标: presencesw/complexquestion_2WIKIMQA_1000")
    print("🌐 来源: https://huggingface.co/datasets/presencesw/complexquestion_2WIKIMQA_1000")
    print("=" * 60)
    
    # 1. 下载数据集
    dataset = download_complex_qa_dataset()
    if dataset is None:
        print("\n❌ 无法下载数据集")
        print("💡 请根据上面的错误诊断信息解决问题")
        return 1
    
    # 2. 提取测试 prompts
    prompts = extract_prompts_for_testing(dataset, max_prompts=20)
    if not prompts:
        print("❌ 没有提取到有效的 prompts")
        return 1
    
    # 3. 保存到文件
    output_dir = save_prompts_to_files(prompts)
    
    # 4. 创建测试配置
    config_file = create_test_config()
    
    # 5. 显示使用说明
    print(f"\n\033[32m🎯 === 使用说明 ===")
    print(f"数据集下载和预处理完成!")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 可用 prompts: {len(prompts)} 个")
    print(f"📄 原始数据集大小: {len(dataset['train'])} 行")
    print(f"\n🚀 运行测试:")
    print(f"cd /home/limingjie/LMJWork/CacheBlend/LMCache/examples/online_session")
    print(f"python kv_cache_test_dateset.py --ports 8001 --dataset custom --dataset-file dataset_prompts/complex_qa_prompts.json")
    print(f"\n或者使用 TXT 格式:")
    print(f"python kv_cache_test_dateset.py --ports 8001 --dataset custom --dataset-file dataset_prompts/complex_qa_prompts.txt")
    print(f"\n📝 数据集特点:")
    print(f"   - 真实的复杂推理问题")
    print(f"   - 包含完整的上下文信息")
    print(f"   - 涵盖多个领域的问答对")
    print(f"   - 经过长度优化，适合测试模型")
    print("\033[0m")
    
    return 0

if __name__ == "__main__":
    exit(main())
