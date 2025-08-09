"""
评估指标计算器
支持多种时序预测评估指标
"""

import numpy as np
from typing import Dict, Union, Optional
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        """初始化指标计算器"""
        self.available_metrics = [
            'mae', 'mse', 'rmse', 'mape', 'smape', 'r2', 
            'explained_variance', 'mean_absolute_percentage_error',
            'symmetric_mean_absolute_percentage_error'
        ]
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         metrics: Optional[list] = None) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            y_true: 真实值 (batch_size, ...) 或 (batch_size, seq_len, features)
            y_pred: 预测值 (batch_size, ...) 或 (batch_size, seq_len, features)
            metrics: 要计算的指标列表，None表示计算所有可用指标
            
        Returns:
            指标字典
        """
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'mape', 'r2']
        
        # 确保输入是numpy数组
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # 展平多维数组
        y_true_flat = y_true.reshape(-1, y_true.shape[-1]) if y_true.ndim > 2 else y_true
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1]) if y_pred.ndim > 2 else y_pred
        
        results = {}
        
        for metric in metrics:
            if metric in self.available_metrics:
                try:
                    value = self._calculate_single_metric(y_true_flat, y_pred_flat, metric)
                    results[metric] = value
                except Exception as e:
                    print(f"计算指标 {metric} 时出错: {e}")
                    results[metric] = float('nan')
        
        return results
    
    def _calculate_single_metric(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                metric: str) -> float:
        """计算单个指标"""
        
        if metric == 'mae':
            return self.mean_absolute_error(y_true, y_pred)
        elif metric == 'mse':
            return self.mean_squared_error(y_true, y_pred)
        elif metric == 'rmse':
            return self.root_mean_squared_error(y_true, y_pred)
        elif metric == 'mape':
            return self.mean_absolute_percentage_error(y_true, y_pred)
        elif metric == 'smape':
            return self.symmetric_mean_absolute_percentage_error(y_true, y_pred)
        elif metric == 'r2':
            return self.r2_score(y_true, y_pred)
        elif metric == 'explained_variance':
            return self.explained_variance_score(y_true, y_pred)
        else:
            raise ValueError(f"不支持的指标: {metric}")
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均绝对误差 (MAE)"""
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方误差 (MSE)"""
        return np.mean((y_true - y_pred) ** 2)
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方根误差 (RMSE)"""
        return np.sqrt(self.mean_squared_error(y_true, y_pred))
    
    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     epsilon: float = 1e-8) -> float:
        """平均绝对百分比误差 (MAPE)"""
        # 避免除零错误
        y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    def symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, 
                                               y_pred: np.ndarray,
                                               epsilon: float = 1e-8) -> float:
        """对称平均绝对百分比误差 (SMAPE)"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        denominator = np.where(denominator < epsilon, epsilon, denominator)
        return np.mean(np.abs(y_true - y_pred) / denominator) * 100
    
    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R²决定系数"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def explained_variance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """解释方差得分"""
        y_true_var = np.var(y_true, ddof=1)
        if y_true_var == 0:
            return 1.0 if np.allclose(y_true, y_pred) else 0.0
        
        residual_var = np.var(y_true - y_pred, ddof=1)
        return 1 - (residual_var / y_true_var)
    
    def directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """方向准确率 - 预测变化方向的准确性"""
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            # 多元时序，计算每个特征的方向准确率然后平均
            accuracies = []
            for i in range(y_true.shape[1]):
                if len(y_true) < 2:
                    continue
                true_direction = np.sign(np.diff(y_true[:, i]))
                pred_direction = np.sign(np.diff(y_pred[:, i]))
                accuracy = np.mean(true_direction == pred_direction)
                accuracies.append(accuracy)
            return np.mean(accuracies) if accuracies else 0.0
        else:
            # 单元时序
            y_true_1d = y_true.flatten() if len(y_true.shape) > 1 else y_true
            y_pred_1d = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
            
            if len(y_true_1d) < 2:
                return 1.0
            
            true_direction = np.sign(np.diff(y_true_1d))
            pred_direction = np.sign(np.diff(y_pred_1d))
            return np.mean(true_direction == pred_direction)
    
    def normalized_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """归一化RMSE"""
        rmse = self.root_mean_squared_error(y_true, y_pred)
        y_range = np.max(y_true) - np.min(y_true)
        if y_range == 0:
            return 0.0
        return rmse / y_range
    
    def coefficient_of_variation_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """RMSE变异系数"""
        rmse = self.root_mean_squared_error(y_true, y_pred)
        y_mean = np.mean(y_true)
        if y_mean == 0:
            return float('inf') if rmse > 0 else 0.0
        return rmse / np.abs(y_mean)
    
    def quantile_loss(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     quantile: float = 0.5) -> float:
        """分位数损失"""
        error = y_true - y_pred
        return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
    
    def calculate_per_feature_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    feature_names: Optional[list] = None) -> Dict[str, Dict[str, float]]:
        """计算每个特征的指标"""
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            # 单特征情况
            overall_metrics = self.calculate_metrics(y_true, y_pred)
            return {'overall': overall_metrics}
        
        results = {}
        num_features = y_true.shape[-1]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(num_features)]
        
        # 计算每个特征的指标
        for i, feature_name in enumerate(feature_names):
            y_true_feature = y_true[..., i]
            y_pred_feature = y_pred[..., i]
            
            feature_metrics = self.calculate_metrics(
                y_true_feature.reshape(-1, 1), 
                y_pred_feature.reshape(-1, 1)
            )
            results[feature_name] = feature_metrics
        
        # 计算整体平均指标
        overall_metrics = {}
        for metric in ['mae', 'mse', 'rmse', 'mape', 'r2']:
            metric_values = [results[fname][metric] for fname in feature_names 
                           if not np.isnan(results[fname].get(metric, np.nan))]
            if metric_values:
                overall_metrics[metric] = np.mean(metric_values)
        
        results['overall'] = overall_metrics
        
        return results
    
    def print_metrics_summary(self, metrics: Dict[str, float], 
                            title: str = "Model Performance") -> None:
        """打印指标摘要"""
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")
        
        for metric, value in metrics.items():
            if not np.isnan(value):
                if metric in ['mape', 'smape']:
                    print(f"{metric.upper():>15}: {value:8.2f}%")
                elif metric in ['r2', 'explained_variance']:
                    print(f"{metric.upper():>15}: {value:8.4f}")
                else:
                    print(f"{metric.upper():>15}: {value:8.6f}")
        print(f"{'='*50}\n")
