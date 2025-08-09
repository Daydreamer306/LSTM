#!/bin/bash
# PAI-DSW 一键部署脚本
# 在PAI-DSW环境中运行此脚本进行完整的项目初始化和部署

set -e  # 遇到错误立即退出

echo "🚀 开始 LSTM+Transformer 工业时序预测模型 PAI-DSW 部署"
echo "============================================================"

# 检查PAI-DSW环境
echo "📋 1. 检查PAI-DSW环境..."
if [ ! -d "/mnt/workspace" ]; then
    echo "❌ 错误: 未检测到PAI-DSW工作空间环境 (/mnt/workspace 不存在)"
    echo "请确保在PAI-DSW实例中运行此脚本"
    exit 1
fi

# 进入工作目录
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

echo "✅ PAI-DSW环境检查通过"
echo "工作目录: $(pwd)"

# 检查Python环境
echo ""
echo "🐍 2. 检查Python环境..."
python_version=$(python --version 2>&1)
echo "Python版本: $python_version"

# 检查GPU
echo ""
echo "🖥️  3. 检查GPU环境..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  未检测到GPU，将使用CPU进行训练"
fi

# 创建目录结构
echo ""
echo "📁 4. 创建项目目录结构..."
directories=(
    "data/raw"
    "data/processed" 
    "data/external"
    "models"
    "results/training"
    "results/predictions"
    "results/visualizations"
    "logs"
    "checkpoints"
    "analysis"
    "exports"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    echo "✅ 创建目录: $dir"
done

# 检查项目文件
echo ""
echo "🗂️  5. 检查项目文件..."
required_files=(
    "setup_pai_dsw.py"
    "requirements.txt"
    "configs/lstm_transformer.yaml"
    "src/__init__.py"
    "src/data/__init__.py"
    "src/models/__init__.py"
    "src/training/__init__.py"
    "src/inference/__init__.py"
    "src/utils/__init__.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - 缺失"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "❌ 错误: 以下关键文件缺失:"
    printf '%s\n' "${missing_files[@]}"
    echo ""
    echo "请确保已将完整的项目文件上传到 /mnt/workspace 目录"
    exit 1
fi

echo "✅ 项目文件检查通过"

# 安装依赖
echo ""
echo "📦 6. 安装Python依赖包..."

# 使用国内镜像源
mirrors=(
    "https://pypi.tuna.tsinghua.edu.cn/simple/"
    "https://mirrors.aliyun.com/pypi/simple/"
    "https://pypi.douban.com/simple/"
)

for mirror in "${mirrors[@]}"; do
    echo "尝试镜像源: $mirror"
    if timeout 300 pip install -r requirements.txt -i "$mirror" --timeout 60; then
        echo "✅ 依赖安装成功!"
        break
    else
        echo "❌ 镜像源 $mirror 安装失败"
        if [ "$mirror" == "${mirrors[-1]}" ]; then
            echo "❌ 所有镜像源都失败，请检查网络连接"
            exit 1
        fi
    fi
done

# 验证关键依赖
echo ""
echo "🔍 7. 验证关键依赖..."
critical_packages=("torch" "numpy" "pandas" "matplotlib" "seaborn" "sklearn" "yaml")

for package in "${critical_packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package"
    else
        echo "❌ $package - 导入失败"
    fi
done

# 运行Python初始化脚本
echo ""
echo "⚙️  8. 运行环境初始化..."
if python setup_pai_dsw.py; then
    echo "✅ Python环境初始化成功"
else
    echo "⚠️  Python环境初始化有警告，请检查输出信息"
fi

# 检查数据文件
echo ""
echo "💾 9. 检查数据文件..."
data_found=false

# 检查常见的数据文件位置
data_locations=(
    "data/raw/*.csv"
    "data/*.csv"
    "/mnt/workspace/data/*.csv"
)

for location in "${data_locations[@]}"; do
    if ls $location 1> /dev/null 2>&1; then
        echo "✅ 发现数据文件: $location"
        data_found=true
        break
    fi
done

if [ "$data_found" = false ]; then
    echo "⚠️  未发现数据文件，请手动上传训练数据到以下位置之一:"
    echo "   - data/raw/"
    echo "   - data/"
    echo ""
    echo "支持的数据文件格式: CSV"
fi

# 创建快速启动脚本
echo ""
echo "📝 10. 创建快速启动脚本..."

# 数据预处理脚本
cat > run_preprocessing.sh << 'EOF'
#!/bin/bash
echo "开始数据预处理..."
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

python src/data/preprocess.py
echo "数据预处理完成!"
EOF

# 训练脚本
cat > run_training.sh << 'EOF'
#!/bin/bash
echo "开始模型训练..."
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

python src/training/train.py --config configs/lstm_transformer.yaml
echo "模型训练完成!"
EOF

# 预测脚本
cat > run_prediction.sh << 'EOF'
#!/bin/bash
echo "开始模型预测..."
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

python src/inference/predict.py \
    --config configs/lstm_transformer.yaml \
    --model_path /mnt/workspace/models/best_model.pt \
    --datasets test \
    --generate_viz

echo "模型预测完成!"
EOF

# 完整流水线脚本
cat > run_full_pipeline.sh << 'EOF'
#!/bin/bash
echo "开始完整训练流水线..."
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

set -e

# 1. 数据预处理
echo "步骤1: 数据预处理"
python src/data/preprocess.py

# 2. 模型训练
echo "步骤2: 模型训练"
python src/training/train.py --config configs/lstm_transformer.yaml

# 3. 模型预测
echo "步骤3: 模型预测"
python src/inference/predict.py \
    --config configs/lstm_transformer.yaml \
    --model_path /mnt/workspace/models/best_model.pt \
    --datasets test val \
    --generate_viz

# 4. 结果分析
echo "步骤4: 结果分析"
python src/inference/analyze_results.py \
    --results_dir /mnt/workspace/results \
    --output_dir /mnt/workspace/analysis

# 5. ONNX导出
echo "步骤5: ONNX导出"
python src/inference/export_onnx.py \
    --model_path /mnt/workspace/models/best_model.pt \
    --output_path /mnt/workspace/models/model.onnx

echo "完整流水线执行完成!"
EOF

# 设置脚本权限
chmod +x run_*.sh

echo "✅ 快速启动脚本已创建:"
echo "   - run_preprocessing.sh  : 数据预处理"
echo "   - run_training.sh       : 模型训练"  
echo "   - run_prediction.sh     : 模型预测"
echo "   - run_full_pipeline.sh  : 完整流水线"

# 创建使用说明
cat > QUICK_START.md << 'EOF'
# PAI-DSW 快速开始指南

## 🎯 项目已成功部署到PAI-DSW!

### 环境信息
- 工作目录: /mnt/workspace
- Python路径: $PYTHONPATH 已设置
- 项目文件: 已完整上传
- 依赖包: 已安装

### 📋 下一步操作

#### 方案1: 逐步执行
```bash
# 1. 数据预处理
./run_preprocessing.sh

# 2. 模型训练  
./run_training.sh

# 3. 模型预测
./run_prediction.sh
```

#### 方案2: 一键执行完整流水线
```bash
# 完整训练+预测+分析流水线
./run_full_pipeline.sh
```

#### 方案3: 手动执行
```bash
# 设置环境
cd /mnt/workspace
export PYTHONPATH=/mnt/workspace:$PYTHONPATH

# 数据预处理
python src/data/preprocess.py

# 模型训练
python src/training/train.py --config configs/lstm_transformer.yaml

# 模型预测
python src/inference/predict.py \
    --config configs/lstm_transformer.yaml \
    --model_path /mnt/workspace/models/best_model.pt
```

### 📊 监控训练进度

#### 在Jupyter Notebook中查看训练曲线:
```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('/mnt/workspace/results/training_history.json', 'r') as f:
    history = json.load(f)

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(history['train_mae'], label='Train MAE')
plt.plot(history['val_mae'], label='Val MAE')
plt.legend() 
plt.title('MAE Curves')
plt.show()
```

### 📁 重要目录
- `/mnt/workspace/data/` - 数据文件
- `/mnt/workspace/models/` - 训练好的模型
- `/mnt/workspace/results/` - 训练结果和日志
- `/mnt/workspace/analysis/` - 分析报告和图表

### ❓ 遇到问题?
1. 检查 `/mnt/workspace/logs/` 目录下的日志文件
2. 运行 `python setup_pai_dsw.py` 重新检查环境
3. 查看 PAI_DSW_DEPLOYMENT.md 获取详细部署指南
EOF

# 最终总结
echo ""
echo "🎉 PAI-DSW部署完成!"
echo "============================================================"
echo "✅ 环境检查: 通过"
echo "✅ 依赖安装: 完成"
echo "✅ 目录创建: 完成"
echo "✅ 脚本生成: 完成"

if [ "$data_found" = true ]; then
    echo "✅ 数据文件: 已发现"
    echo ""
    echo "🚀 您现在可以开始训练模型:"
    echo "   ./run_full_pipeline.sh"
else
    echo "⚠️  数据文件: 请手动上传"
    echo ""
    echo "📋 下一步操作:"
    echo "1. 上传训练数据到 data/raw/ 目录"
    echo "2. 运行: ./run_full_pipeline.sh"
fi

echo ""
echo "📖 详细说明请查看: QUICK_START.md"
echo "📚 部署文档请查看: PAI_DSW_DEPLOYMENT.md"
echo ""
echo "🎯 项目成功部署在: $(pwd)"
echo "============================================================"
