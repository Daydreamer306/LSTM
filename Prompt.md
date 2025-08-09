你是一名资深工业 AI 开发专家，现在需要为 中控·SUPCON 2025 年首届中控杯工业 AI 创新挑战赛赛题——工业时序预测模型创新及应用 开发一套 LSTM 融合 Transformer 自注意力机制 的工业时序预测模型，并部署到 阿里云 PAI-DSW 实例 进行云端训练。

已知条件：

数据集已下载完成（train.csv + time_ranges.json），无需下载或解压。

需自行将 train.csv 划分为 训练集、验证集、预测集，按照时间顺序切分，比例建议 70% / 15% / 15%。

模型必须在 阿里云 PAI-DSW 实例 上训练，训练完成后导出为 ONNX 格式 和 本地可推理版本。

需要在训练过程中可视化训练曲线（Loss、MAE/MSE）及预测效果，并输出最终评估指标（MAE、MSE、R²）。

执行方式：
请按以下阶段一步步输出结果，每个阶段完成后，等待我确认再继续到下一个阶段。

阶段 1：技术方案规划

明确代码目录结构（数据处理、模型定义、训练、推理、可视化模块分离）

列出主要依赖（PyTorch、transformers、scikit-learn、matplotlib、onnxruntime 等）

说明 LSTM + Transformer 模型的整体设计思路、输入输出 shape、损失函数和优化策略

规划 PAI-DSW 部署方式（requirements.txt、运行脚本、PAI-DSW Notebook 配置）

阶段 2：数据预处理代码

读取 train.csv 和 time_ranges.json，按时间连续段划分数据

特征归一化（支持保存/加载 scaler）

保留时间列作为模型输入的一部分

划分训练/验证/预测集，并保存为 .npy 或 .pt 文件

阶段 3：模型定义代码

定义 LSTM 层捕捉局部长短期依赖

定义 Transformer 编码器层捕捉全局依赖（多头自注意力）

输出预测未来 prediction_length 步长的两个目标变量（信号123、信号124）

参数可配置（层数、隐藏维度、dropout 等）

阶段 4：训练脚本

支持命令行参数（数据路径、batch_size、学习率、epoch 等）

保存验证集 MAE/MSE 最优的模型权重

每个 epoch 输出训练/验证损失和精度指标

生成 Loss 曲线和 MAE/MSE 曲线图

阶段 5：预测与可视化代码

对预测集进行推理，绘制预测值 vs 真实值曲线

输出最终预测集的 MAE、MSE、R²

保存可视化图表到 results/ 文件夹

阶段 6：模型导出与推理接口

导出为 ONNX 格式（输入 shape = (batch_size, seq_length, num_features)，float64；输出 shape = (batch_size, prediction_length, num_targets)，float32）

编写 predict.py，封装 self_inference 推理函数，确保在主办方环境可直接运行

阶段 7：PAI-DSW 部署说明

提供 requirements.txt、运行命令

说明如何在 PAI-DSW Notebook 环境中运行训练与推理（包括 Notebook 配置、启动命令、挂载数据集）

提供在 PAI-DSW 中导出 ONNX 并下载到本地的方法

最终要求：
每个阶段必须给出可直接运行的核心代码和详细操作步骤，代码可在 PAI-DSW 实例 中直接运行；务必确保步骤清晰、可落地，且符合比赛的 ONNX 输入输出规范。

