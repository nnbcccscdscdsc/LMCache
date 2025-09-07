#!/bin/bash
# 数据集样本抽取运行脚本

echo "🚀 开始抽取数据集样本"
echo "========================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dm_ev

# 运行抽取脚本
python extract_dataset_samples.py \
  --output-dir ./extracted_samples \
  --seed 42 \
  --2wikimqa 200 \
  --musique 150 \
  --samsum 200 \
  --multinews 60

echo ""
echo "✅ 抽取完成！"
echo "📁 查看 extracted_samples/ 目录获取结果"
