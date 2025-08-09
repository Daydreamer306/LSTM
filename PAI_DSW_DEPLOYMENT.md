# PAI-DSW 部署指南

## LSTM+Transformer 工业时序预测模型 - PAI-DSW部署

本指南将帮助您在阿里云PAI-DSW (Data Science Workshop) 环境中部署和运行LSTM+Transformer时序预测模型。

## 🎯 部署概述

PAI-DSW是阿里云提供的云上数据科学工作环境，提供了完整的机器学习开发和部署能力。本项目已经过优化，完全支持PAI-DSW环境。

### 环境特性
- **计算资源**: 支持CPU/GPU实例，推荐使用GPU加速训练
- **存储空间**: `/mnt/workspace` 作为持久化工作目录
- **Python环境**: 预装Python 3.8+，支持pip和conda
- **深度学习框架**: 预装PyTorch、TensorFlow等主流框架

## 📋 部署前准备

### 1. PAI-DSW实例要求
- **推荐配置**: GPU实例（V100/T4/A10等）
- **内存**: ≥16GB
- **存储**: ≥50GB
- **网络**: 公网访问（用于依赖下载）

### 2. 本地准备
- 将整个项目文件夹打包为zip文件
- 确保包含所有源码、配置文件和文档

## 🚀 快速部署步骤

### 步骤1: 创建PAI-DSW实例
1. 登录阿里云控制台
2. 进入PAI-DSW产品页面
3. 创建新的DSW实例
4. 选择合适的镜像和计算规格
5. 启动实例并进入Jupyter环境

### 步骤2: 上传项目文件
```bash
# 在PAI-DSW Jupyter Terminal中执行
cd /mnt/workspace

# 方法1: 直接上传zip文件并解压
unzip your-project.zip

# 方法2: 从Git仓库克隆
git clone [your-repo-url] lstm_transformer_project
cd lstm_transformer_project
```

### 步骤3: 环境初始化
```python
# 在Jupyter Notebook中运行
exec(open('setup_pai_dsw.py').read())
```

或者在Terminal中：
```bash
python setup_pai_dsw.py
```

### 步骤4: 数据准备
```bash
# 上传您的数据文件到 /mnt/workspace/data/ 目录
# 或使用PAI-DSW的数据集功能
```

### 步骤5: 运行训练
```bash
# 运行完整的训练流程
python src/training/train.py --config configs/lstm_transformer.yaml
```

## 📊 详细部署流程

### 1. 环境检查和配置

**创建初始化脚本** (已提供: `setup_pai_dsw.py`):
```python
# 检查环境状态
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.device_count()}")

# 检查工作空间
import os
print(f"工作目录: {os.getcwd()}")
print(f"存储空间: {os.statvfs('/mnt/workspace')}")
```

### 2. 依赖安装

项目已配置国内镜像源，确保快速安装：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**主要依赖**:
- torch>=1.13.0
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.5.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- PyYAML>=6.0

### 3. 数据管理

**数据上传方式**:
1. **直接上传**: 通过Jupyter界面上传CSV文件
2. **OSS连接**: 配置OSS数据源（推荐用于大数据集）
3. **PAI数据集**: 使用PAI平台的数据管理功能

**数据目录结构**:
```
/mnt/workspace/data/
├── raw/                 # 原始数据
│   ├── train_data.csv
│   ├── test_data.csv
│   └── ...
├── processed/           # 预处理后数据
│   ├── train_processed.npz
│   ├── val_processed.npz
│   └── test_processed.npz
└── external/           # 外部数据源
```

### 4. 模型训练

**基础训练命令**:
```bash
# 基础训练
python src/training/train.py --config configs/lstm_transformer.yaml

# 指定GPU设备
CUDA_VISIBLE_DEVICES=0 python src/training/train.py --config configs/lstm_transformer.yaml

# 后台训练（长时间任务）
nohup python src/training/train.py --config configs/lstm_transformer.yaml > training.log 2>&1 &
```

**监控训练进度**:
```python
# 在Jupyter Notebook中实时监控
import matplotlib.pyplot as plt
import json

# 加载训练历史
with open('/mnt/workspace/results/training_history.json', 'r') as f:
    history = json.load(f)

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_mae'], label='Train MAE')
plt.plot(history['val_mae'], label='Val MAE')
plt.legend()
plt.show()
```

### 5. 模型评估和预测

**运行预测**:
```bash
# 完整预测评估
python src/inference/predict.py \
    --config configs/lstm_transformer.yaml \
    --model_path /mnt/workspace/models/best_model.pt \
    --datasets test \
    --generate_viz
```

**批量预测**:
```bash
# 配置批量预测
python src/inference/batch_predict.py \
    --config configs/batch_prediction_config.json
```

### 6. 结果分析和可视化

**生成分析报告**:
```bash
python src/inference/analyze_results.py \
    --results_dir /mnt/workspace/results \
    --output_dir /mnt/workspace/analysis
```

## 🔧 高级配置

### 1. 多GPU训练配置

```yaml
# configs/lstm_transformer.yaml
training:
  use_multi_gpu: true
  gpu_ids: [0, 1]  # 使用多个GPU
  batch_size: 64   # 增大批量大小
```

### 2. 内存优化配置

```yaml
training:
  gradient_checkpointing: true  # 节省显存
  mixed_precision: true         # 使用混合精度
  accumulate_grad_batches: 4    # 梯度累积
```

### 3. 自动化流水线

创建完整的自动化训练脚本：
```bash
#!/bin/bash
# 完整训练流水线
set -e

echo "开始自动化训练流水线..."

# 1. 数据预处理
python src/data/preprocess.py

# 2. 模型训练
python src/training/train.py --config configs/lstm_transformer.yaml

# 3. 模型评估
python src/inference/predict.py \
    --config configs/lstm_transformer.yaml \
    --model_path /mnt/workspace/models/best_model.pt \
    --datasets test val

# 4. 结果分析
python src/inference/analyze_results.py \
    --results_dir /mnt/workspace/results \
    --output_dir /mnt/workspace/analysis

# 5. ONNX导出
python src/inference/export_onnx.py \
    --model_path /mnt/workspace/models/best_model.pt \
    --output_path /mnt/workspace/models/model.onnx

echo "训练流水线完成!"
```

## 📈 性能优化建议

### 1. 数据加载优化
- 使用`num_workers`参数并行加载数据
- 预处理数据并保存为NPZ格式
- 使用内存映射减少内存占用

### 2. 模型训练优化
- 使用混合精度训练(FP16)节省显存
- 启用梯度检查点减少内存占用
- 合理设置批量大小平衡速度和内存

### 3. 监控和调试
- 使用TensorBoard或自定义可视化监控训练
- 定期保存检查点避免训练中断
- 设置早停策略避免过拟合

## 🚨 常见问题和解决方案

### 1. 内存不足错误
```bash
# 减小批量大小
# 在configs/lstm_transformer.yaml中修改
training:
  batch_size: 16  # 从32减少到16

# 或者启用梯度累积
training:
  accumulate_grad_batches: 4
```

### 2. CUDA错误
```python
# 检查GPU状态
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 重启CUDA环境
torch.cuda.empty_cache()
```

### 3. 依赖安装失败
```bash
# 使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或者使用conda
conda install pytorch torchvision -c pytorch
```

### 4. 数据加载缓慢
```python
# 增加数据加载器的工作进程
# 在配置文件中设置
data:
  num_workers: 4
  pin_memory: true
```

## 📋 部署检查清单

### 部署前检查
- [ ] PAI-DSW实例已创建并启动
- [ ] 工作空间可用存储 > 50GB
- [ ] 网络连接正常
- [ ] 项目文件已上传到 `/mnt/workspace`

### 环境检查
- [ ] Python环境正常 (版本 ≥ 3.8)
- [ ] PyTorch已安装且CUDA可用
- [ ] 所有项目依赖已安装
- [ ] 目录结构完整

### 数据检查
- [ ] 训练数据已上传
- [ ] 数据格式正确
- [ ] 数据预处理脚本运行正常
- [ ] 数据路径配置正确

### 模型检查
- [ ] 模型配置文件正确
- [ ] 训练脚本可以正常启动
- [ ] 可以保存和加载模型检查点
- [ ] GPU内存使用合理

### 结果检查
- [ ] 训练日志正常输出
- [ ] 模型性能指标合理
- [ ] 可视化图表生成正常
- [ ] 预测结果可以导出

## 📞 技术支持

如果遇到部署问题，可以：

1. **查看日志**: 检查 `/mnt/workspace/results/` 目录下的日志文件
2. **内存分析**: 运行 `python memory_analysis.py` 检查内存使用
3. **环境诊断**: 运行 `python setup_pai_dsw.py` 检查环境状态
4. **问题排查**: 参考上述常见问题解决方案

---

## 🎉 部署成功标志

当您看到以下输出时，说明部署成功：

```
=== 模型训练完成 ===
最佳验证MAE: 0.0234
最佳验证R²: 0.9567
模型已保存: /mnt/workspace/models/best_model.pt

=== 预测评估完成 ===
TEST 数据集:
  MAE: 0.0256
  RMSE: 0.0423
  R²: 0.9532

可视化图表已保存到: /mnt/workspace/results/
分析报告已保存到: /mnt/workspace/analysis/
```

部署完成后，您就可以开始在PAI-DSW环境中进行模型训练和预测了！
