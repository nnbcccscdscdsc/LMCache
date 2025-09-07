#!/bin/bash
# 动态数据集测试脚本
# 使用修改后的 kv_cache_test_dateset.py 测试不同的 Hugging Face 数据集

echo "🧪 LMCache 动态数据集测试脚本"
echo "=================================="

# 设置默认参数
PORTS="8001 8002"  # 测试有/无 LMCache 的实例
MODEL="Qwen2.5-0.5B-Instruct"
MAX_SAMPLES=10

echo "📋 可用的数据集:"
python kv_cache_test_dateset.py --list-datasets

echo ""
echo "🚀 开始测试不同数据集..."

# 测试 1: Multi News 数据集
echo ""
echo "=== 测试 1: Multi News 数据集 ==="
python kv_cache_test_dateset.py \
  --ports $PORTS \
  --model $MODEL \
  --dataset huggingface \
  --hf-dataset multi_news \
  --max-samples $MAX_SAMPLES \
  --test-name "Multi News 数据集测试"

# 测试 2: MuSiQue 数据集
echo ""
echo "=== 测试 2: MuSiQue 数据集 ==="
python kv_cache_test_dateset.py \
  --ports $PORTS \
  --model $MODEL \
  --dataset huggingface \
  --hf-dataset musique \
  --max-samples $MAX_SAMPLES \
  --test-name "MuSiQue 数据集测试"

# 测试 3: SAMSum 数据集
echo ""
echo "=== 测试 3: SAMSum 数据集 ==="
python kv_cache_test_dateset.py \
  --ports $PORTS \
  --model $MODEL \
  --dataset huggingface \
  --hf-dataset samsum \
  --max-samples $MAX_SAMPLES \
  --test-name "SAMSum 数据集测试"

# 测试 4: Complex QA 数据集
echo ""
echo "=== 测试 4: Complex QA 数据集 ==="
python kv_cache_test_dateset.py \
  --ports $PORTS \
  --model $MODEL \
  --dataset huggingface \
  --hf-dataset complex_qa \
  --max-samples $MAX_SAMPLES \
  --test-name "Complex QA 数据集测试"

echo ""
echo "✅ 所有测试完成！"
echo "📊 查看生成的 dataset_test_results_*.jsonl 文件获取详细结果"
