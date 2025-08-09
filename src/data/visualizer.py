"""
数据可视化工具
用于生成数据探索性分析图表
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pai_dsw_utils import pai_dsw_utils

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        train_file = pai_dsw_utils.get_data_path("train.csv")
        if not train_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {train_file}")
        
        df = pd.read_csv(train_file)
        return df
    
    def plot_time_series_overview(self, df: pd.DataFrame, target_columns: List[str], 
                                 max_points: int = 5000):
        """绘制时间序列概览"""
        fig, axes = plt.subplots(len(target_columns), 1, figsize=self.figsize, 
                                sharex=True)
        if len(target_columns) == 1:
            axes = [axes]
        
        # 如果数据点太多，进行采样
        if len(df) > max_points:
            step = len(df) // max_points
            df_plot = df.iloc[::step].copy()
        else:
            df_plot = df.copy()
        
        for i, col in enumerate(target_columns):
            if col in df_plot.columns:
                axes[i].plot(df_plot.index, df_plot[col], alpha=0.7, linewidth=0.8)
                axes[i].set_title(f'{col} 时序图')
                axes[i].set_ylabel('数值')
                axes[i].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('时间索引')
        plt.suptitle('目标变量时间序列概览', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        save_path = pai_dsw_utils.get_results_path("time_series_overview.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时序概览图已保存: {save_path}")
        plt.close()
    
    def plot_feature_distributions(self, df: pd.DataFrame, max_features: int = 20):
        """绘制特征分布图"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) > max_features:
            numeric_columns = numeric_columns[:max_features]
        
        n_cols = 4
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(numeric_columns):
            axes[i].hist(df[col].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{col} 分布')
            axes[i].set_xlabel('数值')
            axes[i].set_ylabel('频次')
            axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('特征分布图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = pai_dsw_utils.get_results_path("feature_distributions.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征分布图已保存: {save_path}")
        plt.close()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, target_columns: List[str]):
        """绘制相关性矩阵"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 计算相关性矩阵
        corr_matrix = df[numeric_columns].corr()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 生成相关性热图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        # 高亮目标列
        target_indices = [numeric_columns.index(col) for col in target_columns if col in numeric_columns]
        for idx in target_indices:
            # 添加边框高亮
            ax.add_patch(plt.Rectangle((idx, idx), 1, 1, fill=False, edgecolor='red', lw=3))
        
        ax.set_title('特征相关性矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = pai_dsw_utils.get_results_path("correlation_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"相关性矩阵已保存: {save_path}")
        plt.close()
    
    def plot_data_quality_summary(self, df: pd.DataFrame):
        """绘制数据质量摘要"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. 缺失值统计
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        
        if len(missing_counts) > 0:
            axes[0, 0].bar(range(len(missing_counts)), missing_counts.values)
            axes[0, 0].set_xticks(range(len(missing_counts)))
            axes[0, 0].set_xticklabels(missing_counts.index, rotation=45, ha='right')
            axes[0, 0].set_title('缺失值统计')
            axes[0, 0].set_ylabel('缺失值数量')
        else:
            axes[0, 0].text(0.5, 0.5, '无缺失值', ha='center', va='center', 
                           transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title('缺失值统计')
        
        # 2. 数据类型分布
        dtype_counts = df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('数据类型分布')
        
        # 3. 数值分布统计
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            stats_df = numeric_df.describe().T
            axes[1, 0].scatter(stats_df['mean'], stats_df['std'], alpha=0.6)
            axes[1, 0].set_xlabel('均值')
            axes[1, 0].set_ylabel('标准差')
            axes[1, 0].set_title('均值 vs 标准差')
            
            # 添加标签
            for i, col in enumerate(stats_df.index):
                axes[1, 0].annotate(col, (stats_df.iloc[i]['mean'], stats_df.iloc[i]['std']),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. 数据长度和完整性
        info_data = {
            '总行数': len(df),
            '总列数': len(df.columns),
            '数值列数': len(df.select_dtypes(include=[np.number]).columns),
            '完整行数': df.dropna().shape[0]
        }
        
        axes[1, 1].bar(info_data.keys(), info_data.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 1].set_title('数据基本信息')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('数据质量摘要', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = pai_dsw_utils.get_results_path("data_quality_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据质量摘要已保存: {save_path}")
        plt.close()
    
    def plot_target_analysis(self, df: pd.DataFrame, target_columns: List[str]):
        """目标变量深度分析"""
        if not all(col in df.columns for col in target_columns):
            print("部分目标列不存在，跳过目标分析")
            return
        
        fig, axes = plt.subplots(2, len(target_columns), figsize=(6*len(target_columns), 10))
        if len(target_columns) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(target_columns):
            data = df[col].dropna()
            
            # 时序图
            axes[0, i].plot(data.index, data.values, alpha=0.7, linewidth=0.8)
            axes[0, i].set_title(f'{col} 时序图')
            axes[0, i].set_ylabel('数值')
            axes[0, i].grid(True, alpha=0.3)
            
            # 分布图
            axes[1, i].hist(data, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {data.mean():.2f}')
            axes[1, i].axvline(data.median(), color='orange', linestyle='--', linewidth=2, label=f'中位数: {data.median():.2f}')
            axes[1, i].set_title(f'{col} 分布')
            axes[1, i].set_xlabel('数值')
            axes[1, i].set_ylabel('频次')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle('目标变量深度分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = pai_dsw_utils.get_results_path("target_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"目标分析图已保存: {save_path}")
        plt.close()
    
    def generate_comprehensive_report(self, target_columns: List[str] = ["信号123", "信号124"]):
        """生成综合数据分析报告"""
        print("=== 生成数据可视化报告 ===")
        
        try:
            df = self.load_data()
            print(f"加载数据成功，形状: {df.shape}")
            
            # 生成各种图表
            self.plot_time_series_overview(df, target_columns)
            self.plot_feature_distributions(df)
            self.plot_correlation_matrix(df, target_columns)
            self.plot_data_quality_summary(df)
            self.plot_target_analysis(df, target_columns)
            
            print("=== 所有图表生成完成 ===")
            
        except Exception as e:
            print(f"生成报告失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据可视化工具')
    parser.add_argument('--target_columns', nargs='+', 
                       default=["信号123", "信号124"],
                       help='目标列名称')
    
    args = parser.parse_args()
    
    # 初始化环境
    pai_dsw_utils.setup_directories()
    
    # 创建可视化器并生成报告
    visualizer = DataVisualizer()
    visualizer.generate_comprehensive_report(args.target_columns)


if __name__ == "__main__":
    main()
