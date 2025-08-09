<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# LSTM + Transformer 工业时序预测项目

本项目专注于开发LSTM融合Transformer自注意力机制的工业时序预测模型，用于中控·SUPCON 2025年首届中控杯工业AI创新挑战赛。

## 项目特点
- 使用PyTorch框架
- LSTM捕捉局部长短期依赖
- Transformer编码器捕捉全局依赖关系
- 支持阿里云GPU实例部署
- 模型可导出为ONNX格式
- 包含完整的数据预处理、训练、验证和推理流程

## 代码规范
- 使用类型提示(type hints)
- 模块化设计，职责分离
- 支持配置文件管理超参数
- 包含详细的文档字符串
- 遵循PEP 8编码规范
