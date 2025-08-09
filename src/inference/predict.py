"""
预测推理脚本
对测试集进行推理，生成预测结果并进行评估和可视化
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from pathlib import Path
import pickle

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import TimeSeriesDataModule
from src.models.model_utils import ModelFactory, ModelManager
from src.training.metrics import MetricsCalculator
from src.utils.pai_dsw_utils import pai_dsw_utils
from src.utils.visualization import (
    TrainingVisualizer, 
    PredictionVisualizer, 
    MetricsVisualizer,
    create_comprehensive_report
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        """
        初始化预测器
        
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.model = None
        self.data_module = None
        self.metrics_calculator = MetricsCalculator()
        self.model_factory = ModelFactory()
        self.model_manager = ModelManager()
        
        # 结果存储
        self.predictions = {}
        self.metrics = {}
        
        logger.info(f"预测器初始化完成，使用设备: {self.device}")
    
    def load_model(self, model_path: str) -> None:
        """
        加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        logger.info(f"加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            # 更新配置中的输入维度（如果需要）
            self.config.update(model_config)
        
        # 创建模型
        self.model = self.model_factory.create_model(self.config, model_type="basic")
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 兼容直接保存权重的情况
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型加载成功，参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data(self) -> None:
        """设置数据模块"""
        logger.info("设置数据模块...")
        
        self.data_module = TimeSeriesDataModule(self.config)
        self.data_module.setup()
        
        logger.info("数据模块设置完成")
    
    def predict_dataset(self, dataset_name: str = 'test') -> Tuple[np.ndarray, np.ndarray]:
        """
        对指定数据集进行预测
        
        Args:
            dataset_name: 数据集名称 ('train', 'val', 'test')
            
        Returns:
            predictions: 预测结果
            targets: 真实值
        """
        logger.info(f"开始对 {dataset_name} 数据集进行预测...")
        
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        if self.data_module is None:
            raise ValueError("数据模块未设置，请先调用 setup_data()")
        
        # 获取数据加载器
        if dataset_name == 'train':
            dataloader = self.data_module.train_dataloader()
        elif dataset_name == 'val':
            dataloader = self.data_module.val_dataloader()
        elif dataset_name == 'test':
            dataloader = self.data_module.test_dataloader()
        else:
            raise ValueError(f"不支持的数据集名称: {dataset_name}")
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                # 数据移动到设备
                data = data.to(self.device)
                target = target.to(self.device)
                
                # 模型预测
                output = self.model(data)
                
                # 维度调整
                if output.dim() == 3 and target.dim() == 2:
                    # 如果输出是 (batch, seq_len, features)，取最后一个时间步
                    output = output[:, -1, :]
                elif output.dim() == 3 and target.dim() == 3:
                    # 如果都是3维，确保维度匹配
                    if output.shape != target.shape:
                        output = output[:, -target.shape[1]:, :]
                
                # 收集结果
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"已处理 {batch_idx + 1}/{len(dataloader)} 个批次")
        
        # 合并所有预测结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        logger.info(f"{dataset_name} 数据集预测完成，形状: {predictions.shape}")
        
        return predictions, targets
    
    def calculate_metrics(self, 
                         predictions: np.ndarray, 
                         targets: np.ndarray,
                         dataset_name: str = 'test') -> Dict[str, float]:
        """
        计算预测指标
        
        Args:
            predictions: 预测结果
            targets: 真实值
            dataset_name: 数据集名称
            
        Returns:
            指标字典
        """
        logger.info(f"计算 {dataset_name} 数据集的评估指标...")
        
        metrics = self.metrics_calculator.calculate_metrics(
            targets, predictions, metrics=['mae', 'mse', 'rmse', 'r2', 'mape']
        )
        
        # 按特征计算指标
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            target_names = self.config.get('data', {}).get('target_columns', ['信号123', '信号124'])
            per_feature_metrics = self.metrics_calculator.calculate_per_feature_metrics(
                targets, predictions, target_names
            )
            metrics['per_feature'] = per_feature_metrics
        
        # 打印指标摘要
        self.metrics_calculator.print_metrics_summary(metrics, f"{dataset_name.upper()} Dataset Performance")
        
        return metrics
    
    def run_full_evaluation(self, model_path: str, 
                          datasets: List[str] = ['test'],
                          save_results: bool = True) -> Dict[str, Any]:
        """
        运行完整的模型评估
        
        Args:
            model_path: 模型文件路径
            datasets: 要评估的数据集列表
            save_results: 是否保存结果
            
        Returns:
            评估结果字典
        """
        logger.info("=== 开始完整模型评估 ===")
        
        # 加载模型和数据
        self.load_model(model_path)
        self.setup_data()
        
        results = {
            'predictions': {},
            'targets': {},
            'metrics': {}
        }
        
        # 对每个数据集进行评估
        for dataset_name in datasets:
            logger.info(f"\n--- 评估 {dataset_name} 数据集 ---")
            
            # 预测
            predictions, targets = self.predict_dataset(dataset_name)
            
            # 计算指标
            metrics = self.calculate_metrics(predictions, targets, dataset_name)
            
            # 存储结果
            results['predictions'][dataset_name] = predictions
            results['targets'][dataset_name] = targets
            results['metrics'][dataset_name] = metrics
        
        # 保存结果
        if save_results:
            self._save_results(results)
        
        logger.info("=== 模型评估完成 ===")
        
        return results
    
    def generate_visualizations(self, results: Dict[str, Any], 
                               training_history_path: str = None) -> None:
        """
        生成可视化图表
        
        Args:
            results: 评估结果
            training_history_path: 训练历史文件路径
        """
        logger.info("生成可视化图表...")
        
        # 设置保存目录
        save_dir = pai_dsw_utils.get_results_path("")
        
        # 初始化可视化器
        prediction_viz = PredictionVisualizer(save_dir)
        metrics_viz = MetricsVisualizer(save_dir)
        
        # 获取目标列名称
        target_names = self.config.get('data', {}).get('target_columns', ['信号123', '信号124'])
        
        # 1. 预测结果可视化
        for dataset_name in results['predictions']:
            y_true = results['targets'][dataset_name]
            y_pred = results['predictions'][dataset_name]
            
            # 单样本预测对比
            prediction_viz.plot_prediction_vs_true(
                y_true, y_pred, target_names,
                sample_idx=0,
                save_name=f"{dataset_name}_prediction_vs_true.png"
            )
            
            # 多样本预测对比
            prediction_viz.plot_multiple_predictions(
                y_true, y_pred, target_names,
                num_samples=min(5, y_true.shape[0]),
                save_name=f"{dataset_name}_multiple_predictions.png"
            )
        
        # 2. 指标对比可视化
        metrics_viz.plot_metrics_comparison(
            results['metrics'],
            save_name="metrics_comparison.png"
        )
        
        # 3. 误差分布可视化
        test_predictions = results['predictions'].get('test')
        test_targets = results['targets'].get('test')
        
        if test_predictions is not None and test_targets is not None:
            errors = test_predictions - test_targets
            metrics_viz.plot_error_distribution(
                errors, target_names,
                save_name="error_distribution.png"
            )
        
        # 4. 训练曲线（如果有训练历史）
        if training_history_path and os.path.exists(training_history_path):
            try:
                with open(training_history_path, 'r') as f:
                    history = json.load(f)
                
                training_viz = TrainingVisualizer(save_dir)
                training_viz.plot_training_curves(history)
                
            except Exception as e:
                logger.warning(f"加载训练历史失败: {e}")
        
        logger.info(f"可视化图表已保存到: {save_dir}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """保存评估结果"""
        save_dir = pai_dsw_utils.get_results_path("")
        
        # 保存指标
        metrics_path = save_dir / "evaluation_metrics.json"
        metrics_only = {k: v for k, v in results['metrics'].items() if k != 'per_feature'}
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_only, f, ensure_ascii=False, indent=2)
        
        # 保存预测结果
        predictions_path = save_dir / "predictions.npz"
        np.savez(predictions_path, **results['predictions'])
        
        # 保存真实值
        targets_path = save_dir / "targets.npz"
        np.savez(targets_path, **results['targets'])
        
        logger.info(f"评估结果已保存到: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LSTM + Transformer 模型预测评估')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                       help='预训练模型路径')
    parser.add_argument('--datasets', nargs='+', default=['test'],
                       choices=['train', 'val', 'test'],
                       help='要评估的数据集')
    parser.add_argument('--training_history', type=str, default=None,
                       help='训练历史文件路径')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='是否保存结果')
    parser.add_argument('--generate_viz', action='store_true', default=True,
                       help='是否生成可视化')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 加载配置
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置PAI-DSW环境
    pai_dsw_utils.setup_directories()
    
    try:
        # 创建预测器
        predictor = ModelPredictor(config)
        
        # 运行完整评估
        results = predictor.run_full_evaluation(
            model_path=args.model_path,
            datasets=args.datasets,
            save_results=args.save_results
        )
        
        # 生成可视化
        if args.generate_viz:
            predictor.generate_visualizations(results, args.training_history)
        
        # 输出最终指标摘要
        logger.info("\n" + "="*60)
        logger.info("最终评估结果摘要")
        logger.info("="*60)
        
        for dataset_name, metrics in results['metrics'].items():
            logger.info(f"\n{dataset_name.upper()} 数据集:")
            for metric_name, value in metrics.items():
                if metric_name != 'per_feature' and not np.isnan(value):
                    if metric_name in ['mape', 'smape']:
                        logger.info(f"  {metric_name.upper()}: {value:.2f}%")
                    elif metric_name == 'r2':
                        logger.info(f"  {metric_name.upper()}: {value:.4f}")
                    else:
                        logger.info(f"  {metric_name.upper()}: {value:.6f}")
        
        logger.info("\n预测评估完成!")
        
    except Exception as e:
        logger.error(f"预测评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
