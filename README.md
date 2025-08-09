# LSTM + Transformer 工业时序预测模型

## 项目简介

本项目开发了一套LSTM融合Transformer自注意力机制的工业时序预测模型，用于中控·SUPCON 2025年首届中控杯工业AI创新挑战赛。支持阿里云PAI-DSW环境完整部署。

## 🚀 PAI-DSW快速部署

### 一键部署 (推荐)

```bash
# 在PAI-DSW Terminal中运行
bash deploy_pai_dsw_full.sh
```

### Jupyter Notebook部署

1. 在PAI-DSW环境中打开 `PAI_DSW_QuickStart.ipynb`
2. 按顺序运行所有代码单元
3. 自动完成环境初始化、模型训练和预测

### 详细部署指南

查看 [`PAI_DSW_DEPLOYMENT.md`](PAI_DSW_DEPLOYMENT.md) 获取完整的部署文档。

## 技术方案

### 模型架构
- **LSTM层**: 捕捉时序数据的局部长短期依赖关系
- **Transformer编码器**: 利用多头自注意力机制捕捉全局依赖关系
- **融合策略**: LSTM输出作为Transformer的输入，实现层次化特征提取
- **预测头**: 全连接层输出未来prediction_length步长的两个目标变量

### 目录结构
```
.
├── src/                     # 源代码目录
│   ├── data/               # 数据处理模块
│   ├── models/             # 模型定义模块
│   ├── training/           # 训练相关模块
│   ├── inference/          # 推理和分析模块
│   └── utils/              # 工具函数
├── data/                   # 数据文件夹
├── models/                 # 保存的模型文件
├── results/                # 结果输出（图表、指标等）
├── configs/                # 配置文件
├── docker/                 # Docker部署相关
├── PAI_DSW_DEPLOYMENT.md   # PAI-DSW部署指南
├── PAI_DSW_QuickStart.ipynb # Jupyter快速启动
├── deploy_pai_dsw_full.sh  # 一键部署脚本
├── requirements.txt        # Python依赖
└── README.md              # 项目说明
```

## 主要依赖

- **PyTorch**: 深度学习框架
- **scikit-learn**: 数据预处理和评估指标
- **matplotlib/seaborn**: 数据可视化
- **onnx/onnxruntime**: 模型导出和推理
- **pandas/numpy**: 数据处理
- **PyYAML**: 配置文件解析

## 环境要求

### PAI-DSW环境 (推荐)
- 阿里云PAI-DSW实例
- Python 3.8+
- GPU支持 (推荐)

### 本地环境
- Python 3.8+
- CUDA 11.0+ (GPU训练)
- 16GB+ RAM

## 快速开始

### PAI-DSW环境 (推荐)

1. **创建PAI-DSW实例**
   ```bash
   # 推荐配置：GPU实例，16GB内存，50GB存储
   ```

2. **一键部署**
   ```bash
   # 上传项目文件到 /mnt/workspace
   # 运行部署脚本
   bash deploy_pai_dsw_full.sh
   ```

3. **开始训练**
   ```bash
   # 自动化流水线
   ./run_full_pipeline.sh
   
   # 或者分步执行
   ./run_preprocessing.sh
   ./run_training.sh 
   ./run_prediction.sh
   ```

### 本地环境
```bash
pip install -r requirements.txt
```

### 2. 数据准备
```bash
python src/data/preprocess.py --data_path data/train.csv --time_ranges data/time_ranges.json
```

### 3. 模型训练
```bash
python src/training/train.py --config configs/lstm_transformer.yaml
```

### 4. 模型推理
```bash
python src/inference/predict.py --model_path models/best_model.pt --data_path data/test_data.npy
```

### 5. ONNX导出
```bash
python src/inference/export_onnx.py --model_path models/best_model.pt --output_path models/model.onnx
```

## PAI-DSW部署 (推荐)

### 方法1: Jupyter Notebook环境
1. **创建PAI-DSW实例**
   - 登录阿里云PAI控制台
   - 选择GPU实例 (推荐: ecs.gn6i-c4g1.xlarge)
   - 创建Jupyter Notebook环境

2. **上传项目代码**
   ```bash
   # 将项目文件上传到 /mnt/workspace/
   # 或使用Git克隆
   cd /mnt/workspace
   git clone <your_repo_url> .
   ```

3. **环境初始化**
   ```python
   # 在Jupyter Notebook中运行
   exec(open('setup_pai_dsw.py').read())
   ```

4. **上传数据文件**
   - 将 `train.csv` 和 `time_ranges.json` 上传到 `/mnt/workspace/data/`

5. **开始训练**
   ```python
   !python src/training/train.py --config configs/lstm_transformer.yaml
   ```

### 方法2: Terminal环境  
```bash
# SSH连接到PAI-DSW实例或使用Terminal
cd /mnt/workspace
chmod +x deploy_pai_dsw.sh
./deploy_pai_dsw.sh

# 上传数据后开始训练
python src/training/train.py --config configs/lstm_transformer.yaml
```

### 方法3: Docker部署 (PAI-DSW兼容)
```bash
cd docker
docker build -t lstm-transformer-pai .
docker run --gpus all -v /mnt/workspace:/mnt/workspace lstm-transformer-pai
```

## 模型输入输出规范

### ONNX模型规范
- **输入**: (batch_size, seq_length, num_features) - float64
- **输出**: (batch_size, prediction_length, num_targets) - float32
- **目标变量**: 信号123、信号124

### 评估指标
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)  
- R² (Coefficient of Determination)

## 可视化功能

- 训练过程Loss曲线
- 验证集MAE/MSE曲线
- 预测值vs真实值对比图
- 特征重要性分析

## 配置说明

所有超参数通过YAML配置文件管理，支持：
- 模型结构参数 (层数、隐藏维度、dropout等)
- 训练参数 (batch_size、学习率、epoch等)
- 数据处理参数 (序列长度、预测长度等)

## 许可证

MIT License
