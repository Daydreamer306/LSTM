#!/bin/bash
# PAI-DSW环境初始化和部署脚本

echo "=== PAI-DSW LSTM+Transformer 工业时序预测模型部署 ==="

# 检查PAI-DSW环境
echo "检查PAI-DSW环境..."
if [ ! -d "/mnt/workspace" ]; then
    echo "错误: 未检测到PAI-DSW工作空间环境"
    exit 1
fi

# 设置工作目录
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

# 创建必要的目录
echo "创建目录结构..."
mkdir -p /mnt/workspace/data
mkdir -p /mnt/workspace/models  
mkdir -p /mnt/workspace/results
mkdir -p /mnt/workspace/checkpoints
mkdir -p /mnt/workspace/logs

# 检查GPU可用性
echo "检查GPU环境..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 验证关键依赖
echo "验证关键依赖..."
python -c "import torch, transformers, sklearn, pandas, numpy; print('所有依赖安装成功!')"

# 检查数据文件
echo "检查数据文件..."
if [ ! -f "/mnt/workspace/data/train.csv" ]; then
    echo "警告: train.csv 文件不存在，请上传数据文件到 /mnt/workspace/data/ 目录"
fi

if [ ! -f "/mnt/workspace/data/time_ranges.json" ]; then
    echo "警告: time_ranges.json 文件不存在，请上传数据文件到 /mnt/workspace/data/ 目录"  
fi

# 设置权限
chmod +x /mnt/workspace/src/training/train.py
chmod +x /mnt/workspace/src/inference/predict.py

echo "=== PAI-DSW环境配置完成! ==="
echo ""
echo "数据准备:"
echo "  请将 train.csv 和 time_ranges.json 上传到 /mnt/workspace/data/ 目录"
echo ""
echo "开始训练:"
echo "  python src/training/train.py --config configs/lstm_transformer.yaml"
echo ""
echo "模型推理:"
echo "  python src/inference/predict.py --model_path /mnt/workspace/models/best_model.pt"
echo ""
