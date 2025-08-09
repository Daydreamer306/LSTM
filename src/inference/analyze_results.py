"""
结果分析工具
提供详细的预测结果分析、比较和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from ..training.metrics import calculate_metrics


@dataclass
class AnalysisConfig:
    """分析配置"""
    results_dir: str
    output_dir: str
    models: List[str] = None
    datasets: List[str] = None
    metrics: List[str] = None
    create_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认指标
        self.default_metrics = ['mae', 'mse', 'rmse', 'r2', 'mape']
        self.metrics = config.metrics or self.default_metrics
        
        # 设置绘图风格
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_results(self) -> Dict[str, Any]:
        """加载所有结果文件"""
        results = {}
        
        # 查找所有预测结果文件
        prediction_files = list(self.results_dir.glob("*_predictions.csv"))
        
        for pred_file in prediction_files:
            # 提取模型和数据集名称
            file_name = pred_file.stem.replace("_predictions", "")
            
            if file_name.startswith("ensemble_"):
                model_name = "ensemble"
                dataset_name = file_name.replace("ensemble_", "")
            else:
                parts = file_name.split("_", 1)
                model_name = parts[0] if len(parts) > 0 else "unknown"
                dataset_name = parts[1] if len(parts) > 1 else "unknown"
            
            # 过滤模型和数据集
            if self.config.models and model_name not in self.config.models:
                continue
            if self.config.datasets and dataset_name not in self.config.datasets:
                continue
                
            # 加载预测数据
            pred_df = pd.read_csv(pred_file)
            
            # 加载指标数据
            metrics_file = pred_file.parent / f"{file_name}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
            else:
                # 计算指标
                if 'predictions' in pred_df.columns and 'targets' in pred_df.columns:
                    predictions = pred_df['predictions'].values
                    targets = pred_df['targets'].values
                elif 'ensemble_predictions' in pred_df.columns and 'targets' in pred_df.columns:
                    predictions = pred_df['ensemble_predictions'].values
                    targets = pred_df['targets'].values
                else:
                    continue
                    
                metrics_data = calculate_metrics(targets, predictions)
            
            results[file_name] = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'predictions': pred_df,
                'metrics': metrics_data
            }
            
        return results
        
    def create_metrics_comparison(self, results: Dict[str, Any]) -> pd.DataFrame:
        """创建指标比较表"""
        comparison_data = []
        
        for result_name, result in results.items():
            row = {
                'Result': result_name,
                'Model': result['model_name'],
                'Dataset': result['dataset_name']
            }
            
            # 添加指标
            for metric in self.metrics:
                if metric in result['metrics']:
                    row[metric.upper()] = result['metrics'][metric]
                    
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较表
        comparison_path = self.output_dir / "metrics_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        return comparison_df
        
    def create_ranking_table(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """创建模型排名表"""
        ranking_data = []
        
        # 按数据集分组
        for dataset in comparison_df['Dataset'].unique():
            dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
            
            for metric in self.metrics:
                metric_col = metric.upper()
                if metric_col not in dataset_df.columns:
                    continue
                    
                # 排序（MAE, MSE, RMSE, MAPE越小越好，R2越大越好）
                ascending = metric.lower() != 'r2'
                sorted_df = dataset_df.sort_values(metric_col, ascending=ascending)
                
                for rank, (_, row) in enumerate(sorted_df.iterrows(), 1):
                    ranking_data.append({
                        'Dataset': dataset,
                        'Metric': metric_col,
                        'Rank': rank,
                        'Model': row['Model'],
                        'Value': row[metric_col]
                    })
                    
        ranking_df = pd.DataFrame(ranking_data)
        
        # 保存排名表
        ranking_path = self.output_dir / "model_ranking.csv"
        ranking_df.to_csv(ranking_path, index=False)
        
        return ranking_df
        
    def plot_metrics_comparison(self, comparison_df: pd.DataFrame):
        """绘制指标比较图"""
        if not self.config.create_plots:
            return
            
        # 为每个指标创建一个图
        for metric in self.metrics:
            metric_col = metric.upper()
            if metric_col not in comparison_df.columns:
                continue
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 创建分组条形图
            datasets = comparison_df['Dataset'].unique()
            models = comparison_df['Model'].unique()
            
            x = np.arange(len(datasets))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                model_data = []
                for dataset in datasets:
                    value = comparison_df[
                        (comparison_df['Dataset'] == dataset) & 
                        (comparison_df['Model'] == model)
                    ][metric_col].values
                    model_data.append(value[0] if len(value) > 0 else 0)
                    
                ax.bar(x + i * width, model_data, width, label=model, alpha=0.8)
                
            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric_col)
            ax.set_title(f'{metric_col} Comparison Across Models and Datasets')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(datasets, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{metric}_comparison.{self.config.plot_format}"
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
    def plot_prediction_scatter(self, results: Dict[str, Any]):
        """绘制预测散点图"""
        if not self.config.create_plots:
            return
            
        for result_name, result in results.items():
            pred_df = result['predictions']
            
            # 确定列名
            if 'predictions' in pred_df.columns and 'targets' in pred_df.columns:
                pred_col, target_col = 'predictions', 'targets'
            elif 'ensemble_predictions' in pred_df.columns and 'targets' in pred_df.columns:
                pred_col, target_col = 'ensemble_predictions', 'targets'
            else:
                continue
                
            predictions = pred_df[pred_col].values
            targets = pred_df[target_col].values
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 散点图
            ax.scatter(targets, predictions, alpha=0.6, s=20)
            
            # 完美预测线
            min_val = min(targets.min(), predictions.min())
            max_val = max(targets.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # 计算R²
            r2 = result['metrics'].get('r2', 0)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Prediction Scatter Plot - {result_name}\nR² = {r2:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{result_name}_scatter.{self.config.plot_format}"
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
    def plot_residuals(self, results: Dict[str, Any]):
        """绘制残差图"""
        if not self.config.create_plots:
            return
            
        for result_name, result in results.items():
            pred_df = result['predictions']
            
            # 确定列名
            if 'predictions' in pred_df.columns and 'targets' in pred_df.columns:
                pred_col, target_col = 'predictions', 'targets'
            elif 'ensemble_predictions' in pred_df.columns and 'targets' in pred_df.columns:
                pred_col, target_col = 'ensemble_predictions', 'targets'
            else:
                continue
                
            predictions = pred_df[pred_col].values
            targets = pred_df[target_col].values
            residuals = targets - predictions
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 残差散点图
            ax1.scatter(predictions, residuals, alpha=0.6, s=20)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title(f'Residual Plot - {result_name}')
            ax1.grid(True, alpha=0.3)
            
            # 残差分布直方图
            ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--')
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Residual Distribution - {result_name}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{result_name}_residuals.{self.config.plot_format}"
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
    def plot_time_series(self, results: Dict[str, Any], max_points: int = 1000):
        """绘制时间序列预测图"""
        if not self.config.create_plots:
            return
            
        for result_name, result in results.items():
            pred_df = result['predictions']
            
            # 确定列名
            if 'predictions' in pred_df.columns and 'targets' in pred_df.columns:
                pred_col, target_col = 'predictions', 'targets'
            elif 'ensemble_predictions' in pred_df.columns and 'targets' in pred_df.columns:
                pred_col, target_col = 'ensemble_predictions', 'targets'
            else:
                continue
                
            predictions = pred_df[pred_col].values
            targets = pred_df[target_col].values
            
            # 限制点数以提高绘图效率
            if len(predictions) > max_points:
                indices = np.linspace(0, len(predictions)-1, max_points).astype(int)
                predictions = predictions[indices]
                targets = targets[indices]
                
            time_axis = range(len(predictions))
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            ax.plot(time_axis, targets, label='Actual', alpha=0.7, linewidth=1)
            ax.plot(time_axis, predictions, label='Predicted', alpha=0.7, linewidth=1)
            
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Values')
            ax.set_title(f'Time Series Prediction - {result_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / f"{result_name}_timeseries.{self.config.plot_format}"
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
    def create_summary_report(
        self, 
        results: Dict[str, Any],
        comparison_df: pd.DataFrame,
        ranking_df: pd.DataFrame
    ):
        """创建汇总报告"""
        report = {
            'analysis_summary': {
                'total_results': len(results),
                'models': list(set(r['model_name'] for r in results.values())),
                'datasets': list(set(r['dataset_name'] for r in results.values())),
                'metrics_analyzed': self.metrics
            },
            'best_models': {},
            'detailed_metrics': {},
            'analysis_insights': []
        }
        
        # 找出每个数据集和指标的最佳模型
        for dataset in comparison_df['Dataset'].unique():
            dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
            report['best_models'][dataset] = {}
            
            for metric in self.metrics:
                metric_col = metric.upper()
                if metric_col not in dataset_df.columns:
                    continue
                    
                # 找最佳值
                ascending = metric.lower() != 'r2'
                best_row = dataset_df.loc[dataset_df[metric_col].idxmin() if ascending else dataset_df[metric_col].idxmax()]
                
                report['best_models'][dataset][metric] = {
                    'model': best_row['Model'],
                    'value': best_row[metric_col]
                }
                
        # 详细指标
        for result_name, result in results.items():
            report['detailed_metrics'][result_name] = result['metrics']
            
        # 分析洞察
        insights = []
        
        # 1. 整体最佳模型
        model_scores = {}
        for dataset in ranking_df['Dataset'].unique():
            dataset_ranks = ranking_df[ranking_df['Dataset'] == dataset]
            for model in dataset_ranks['Model'].unique():
                model_ranks = dataset_ranks[dataset_ranks['Model'] == model]['Rank'].mean()
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(model_ranks)
                
        avg_scores = {model: np.mean(ranks) for model, ranks in model_scores.items()}
        best_overall_model = min(avg_scores.keys(), key=lambda x: avg_scores[x])
        insights.append(f"整体表现最佳的模型是: {best_overall_model} (平均排名: {avg_scores[best_overall_model]:.2f})")
        
        # 2. 指标分析
        for metric in self.metrics:
            metric_col = metric.upper()
            if metric_col in comparison_df.columns:
                best_value = comparison_df[metric_col].min() if metric.lower() != 'r2' else comparison_df[metric_col].max()
                worst_value = comparison_df[metric_col].max() if metric.lower() != 'r2' else comparison_df[metric_col].min()
                improvement = ((worst_value - best_value) / worst_value * 100) if worst_value != 0 else 0
                insights.append(f"{metric_col}: 最佳值 {best_value:.4f}, 最差值 {worst_value:.4f}, 改进空间 {improvement:.2f}%")
                
        report['analysis_insights'] = insights
        
        # 保存报告
        report_path = self.output_dir / 'analysis_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 创建可读性更好的报告
        readable_report_path = self.output_dir / 'analysis_report.md'
        self._create_readable_report(report, readable_report_path)
        
    def _create_readable_report(self, report: Dict[str, Any], output_path: Path):
        """创建可读性更好的Markdown报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 模型性能分析报告\n\n")
            
            # 概述
            f.write("## 分析概述\n\n")
            summary = report['analysis_summary']
            f.write(f"- 分析结果数量: {summary['total_results']}\n")
            f.write(f"- 模型数量: {len(summary['models'])}\n")
            f.write(f"- 数据集数量: {len(summary['datasets'])}\n")
            f.write(f"- 分析指标: {', '.join(summary['metrics_analyzed'])}\n\n")
            
            # 最佳模型
            f.write("## 各数据集最佳模型\n\n")
            for dataset, metrics in report['best_models'].items():
                f.write(f"### {dataset}\n\n")
                for metric, info in metrics.items():
                    f.write(f"- **{metric}**: {info['model']} ({info['value']:.4f})\n")
                f.write("\n")
                
            # 分析洞察
            f.write("## 分析洞察\n\n")
            for insight in report['analysis_insights']:
                f.write(f"- {insight}\n")
            f.write("\n")
            
    def analyze(self) -> Dict[str, Any]:
        """执行完整分析"""
        print("开始结果分析...")
        
        # 加载结果
        results = self.load_results()
        if not results:
            raise ValueError("没有找到任何结果文件")
            
        print(f"加载了 {len(results)} 个结果")
        
        # 创建比较表
        comparison_df = self.create_metrics_comparison(results)
        print("指标比较表已创建")
        
        # 创建排名表
        ranking_df = self.create_ranking_table(comparison_df)
        print("模型排名表已创建")
        
        # 创建可视化
        if self.config.create_plots:
            print("正在生成可视化图表...")
            self.plot_metrics_comparison(comparison_df)
            self.plot_prediction_scatter(results)
            self.plot_residuals(results)
            self.plot_time_series(results)
            print("可视化图表已生成")
            
        # 创建汇总报告
        self.create_summary_report(results, comparison_df, ranking_df)
        print("分析报告已生成")
        
        return {
            'results': results,
            'comparison_df': comparison_df,
            'ranking_df': ranking_df,
            'output_dir': str(self.output_dir)
        }


def main():
    """结果分析主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="结果分析工具")
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='结果文件目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='分析输出目录'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='要分析的模型列表'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='要分析的数据集列表'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='不生成图表'
    )
    
    args = parser.parse_args()
    
    # 创建分析配置
    config = AnalysisConfig(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=args.models,
        datasets=args.datasets,
        create_plots=not args.no_plots
    )
    
    # 执行分析
    analyzer = ResultsAnalyzer(config)
    analyzer.analyze()
    
    print("结果分析完成!")


if __name__ == "__main__":
    main()
