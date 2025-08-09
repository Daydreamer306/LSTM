"""
工业时序预测可视化模块
支持训练曲线、预测结果、注意力权重等多种可视化功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import torch
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_training_curves(self, 
                           history: Dict[str, List[float]], 
                           title: str = "训练曲线",
                           save_name: str = "training_curves.png",
                           show: bool = False) -> None:
        """
        绘制训练曲线（Loss、MAE、MSE）
        
        Args:
            history: 训练历史字典，包含各项指标
            title: 图表标题
            save_name: 保存文件名
            show: 是否显示图表
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        epochs = np.arange(1, len(history['train_loss']) + 1)
        
        # Loss曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['train_loss'], 'b-', label='训练Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='验证Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失函数曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['train_mae'], 'b-', label='训练MAE', linewidth=2)
        ax2.plot(epochs, history['val_mae'], 'r-', label='验证MAE', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('平均绝对误差曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MSE曲线
        ax3 = axes[1, 0]
        ax3.plot(epochs, history['train_mse'], 'b-', label='训练MSE', linewidth=2)
        ax3.plot(epochs, history['val_mse'], 'r-', label='验证MSE', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MSE')
        ax3.set_title('均方误差曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax4 = axes[1, 1]
        if 'learning_rate' in history:
            ax4.plot(epochs, history['learning_rate'], 'g-', label='学习率', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('学习率变化曲线')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '学习率数据不可用', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('学习率变化曲线')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")
        
        if show:
            plt.show()
        plt.close()

class PredictionVisualizer:
    """预测结果可视化器"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_prediction_vs_true(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               target_names: List[str] = ["信号123", "信号124"],
                               sample_idx: int = 0,
                               save_name: str = "prediction_vs_true.png",
                               show: bool = False) -> None:
        """
        绘制预测值vs真实值对比曲线
        
        Args:
            y_true: 真实值 (num_samples, prediction_length, num_targets)
            y_pred: 预测值 (num_samples, prediction_length, num_targets)
            target_names: 目标变量名称列表
            sample_idx: 要绘制的样本索引
            save_name: 保存文件名
            show: 是否显示图表
        """
        num_targets = y_true.shape[-1]
        prediction_length = y_true.shape[1]
        
        fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5))
        if num_targets == 1:
            axes = [axes]
        
        fig.suptitle(f'预测结果对比 (样本 {sample_idx})', fontsize=14, fontweight='bold')
        
        time_steps = np.arange(1, prediction_length + 1)
        
        for i, target_name in enumerate(target_names[:num_targets]):
            ax = axes[i]
            
            true_values = y_true[sample_idx, :, i]
            pred_values = y_pred[sample_idx, :, i]
            
            ax.plot(time_steps, true_values, 'b-', label='真实值', 
                   linewidth=2, marker='o', markersize=4)
            ax.plot(time_steps, pred_values, 'r--', label='预测值', 
                   linewidth=2, marker='s', markersize=4)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel('数值')
            ax.set_title(f'{target_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加误差区域
            error = np.abs(true_values - pred_values)
            ax.fill_between(time_steps, true_values - error, true_values + error, 
                          alpha=0.2, color='gray', label='误差区域')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测对比图已保存至: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_multiple_predictions(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 target_names: List[str] = ["信号123", "信号124"],
                                 num_samples: int = 5,
                                 save_name: str = "multiple_predictions.png",
                                 show: bool = False) -> None:
        """
        绘制多个样本的预测结果
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            target_names: 目标变量名称
            num_samples: 绘制的样本数量
            save_name: 保存文件名
            show: 是否显示图表
        """
        num_targets = y_true.shape[-1]
        prediction_length = y_true.shape[1]
        
        fig, axes = plt.subplots(num_targets, num_samples, 
                               figsize=(4 * num_samples, 3 * num_targets))
        if num_targets == 1:
            axes = axes.reshape(1, -1)
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('多样本预测结果对比', fontsize=16, fontweight='bold')
        
        time_steps = np.arange(1, prediction_length + 1)
        
        for target_idx, target_name in enumerate(target_names[:num_targets]):
            for sample_idx in range(min(num_samples, y_true.shape[0])):
                ax = axes[target_idx, sample_idx]
                
                true_values = y_true[sample_idx, :, target_idx]
                pred_values = y_pred[sample_idx, :, target_idx]
                
                ax.plot(time_steps, true_values, 'b-', label='真实值', linewidth=1.5)
                ax.plot(time_steps, pred_values, 'r--', label='预测值', linewidth=1.5)
                
                if sample_idx == 0:
                    ax.set_ylabel(f'{target_name}')
                if target_idx == num_targets - 1:
                    ax.set_xlabel('时间步')
                
                ax.set_title(f'样本 {sample_idx}')
                ax.grid(True, alpha=0.3)
                
                if target_idx == 0 and sample_idx == 0:
                    ax.legend(fontsize=8)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多样本预测图已保存至: {save_path}")
        
        if show:
            plt.show()
        plt.close()

class MetricsVisualizer:
    """评估指标可视化器"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_comparison(self,
                              metrics_dict: Dict[str, Dict[str, float]],
                              save_name: str = "metrics_comparison.png",
                              show: bool = False) -> None:
        """
        绘制不同数据集的指标对比
        
        Args:
            metrics_dict: 指标字典 {'train': {'MAE': 0.1, 'MSE': 0.01, 'R2': 0.9}, ...}
            save_name: 保存文件名
            show: 是否显示图表
        """
        datasets = list(metrics_dict.keys())
        metric_names = list(next(iter(metrics_dict.values())).keys())
        
        fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
        if len(metric_names) == 1:
            axes = [axes]
        
        fig.suptitle('模型评估指标对比', fontsize=16, fontweight='bold')
        
        for i, metric_name in enumerate(metric_names):
            ax = axes[i]
            values = [metrics_dict[dataset][metric_name] for dataset in datasets]
            
            bars = ax.bar(datasets, values, alpha=0.7, 
                         color=['skyblue', 'lightcoral', 'lightgreen'][:len(datasets)])
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} 对比')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存至: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def plot_error_distribution(self,
                               errors: np.ndarray,
                               target_names: List[str] = ["信号123", "信号124"],
                               save_name: str = "error_distribution.png",
                               show: bool = False) -> None:
        """
        绘制预测误差分布直方图
        
        Args:
            errors: 预测误差 (num_samples, prediction_length, num_targets)
            target_names: 目标变量名称
            save_name: 保存文件名
            show: 是否显示图表
        """
        num_targets = errors.shape[-1]
        
        fig, axes = plt.subplots(1, num_targets, figsize=(6 * num_targets, 5))
        if num_targets == 1:
            axes = [axes]
        
        fig.suptitle('预测误差分布', fontsize=16, fontweight='bold')
        
        for i, target_name in enumerate(target_names[:num_targets]):
            ax = axes[i]
            
            # 展平误差数据
            target_errors = errors[:, :, i].flatten()
            
            # 绘制直方图
            n, bins, patches = ax.hist(target_errors, bins=50, alpha=0.7, 
                                      color='skyblue', edgecolor='black', linewidth=0.5)
            
            # 添加统计信息
            mean_error = np.mean(target_errors)
            std_error = np.std(target_errors)
            
            ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                      label=f'均值: {mean_error:.4f}')
            ax.axvline(mean_error + std_error, color='orange', linestyle='--', alpha=0.7,
                      label=f'±1σ: {std_error:.4f}')
            ax.axvline(mean_error - std_error, color='orange', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('预测误差')
            ax.set_ylabel('频次')
            ax.set_title(f'{target_name} 误差分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"误差分布图已保存至: {save_path}")
        
        if show:
            plt.show()
        plt.close()

def create_comprehensive_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              history: Dict[str, List[float]],
                              metrics: Dict[str, Dict[str, float]],
                              target_names: List[str] = ["信号123", "信号124"],
                              save_dir: str = "results") -> None:
    """
    生成综合评估报告
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        history: 训练历史
        metrics: 评估指标
        target_names: 目标变量名称
        save_dir: 保存目录
    """
    print("正在生成综合评估报告...")
    
    # 初始化可视化器
    training_viz = TrainingVisualizer(save_dir)
    prediction_viz = PredictionVisualizer(save_dir)
    metrics_viz = MetricsVisualizer(save_dir)
    
    # 1. 绘制训练曲线
    training_viz.plot_training_curves(history)
    
    # 2. 绘制预测对比
    prediction_viz.plot_prediction_vs_true(y_true, y_pred, target_names)
    
    # 3. 绘制多样本预测
    prediction_viz.plot_multiple_predictions(y_true, y_pred, target_names, num_samples=3)
    
    # 4. 绘制指标对比
    metrics_viz.plot_metrics_comparison(metrics)
    
    # 5. 绘制误差分布
    errors = y_pred - y_true
    metrics_viz.plot_error_distribution(errors, target_names)
    
    print(f"综合评估报告已生成完成，保存在: {save_dir}/")

# 导出主要类和函数
__all__ = [
    'TrainingVisualizer',
    'PredictionVisualizer', 
    'MetricsVisualizer',
    'create_comprehensive_report'
]