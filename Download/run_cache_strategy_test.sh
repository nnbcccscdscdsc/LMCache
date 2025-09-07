#!/bin/bash
# 缓存策略测试运行脚本

echo "🚀 开始缓存策略测试"
echo "===================="

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dm_ev

# 检查 vLLM 服务是否运行
echo "🔍 检查 vLLM 服务状态..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM 服务未运行，请先启动 vLLM 服务"
    echo "💡 启动命令示例:"
    echo "   python -m vllm.entrypoints.openai.api_server --model Qwen2.5-0.5B-Instruct --port 8000"
    exit 1
fi

echo "✅ vLLM 服务运行正常"

# 运行缓存策略测试
echo ""
echo "🧪 运行缓存策略测试..."
python cache_strategy_test.py \
  --api-base http://localhost:8000 \
  --model Qwen2.5-0.5B-Instruct \
  --prefix-length 50 \
  --output cache_strategy_results.json

echo ""
echo "✅ 测试完成！"
echo "📊 查看 cache_strategy_results.json 获取详细结果"
