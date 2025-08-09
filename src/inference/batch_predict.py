"""
批量预测工具
支持多个模型和多个数据集的批量预测
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from ..models.fusion_model import LSTMTransformerFusion
from ..models.model_utils import create_model
from ..data.dataset import IndustrialTimeSeriesDataset
from ..training.metrics import calculate_metrics
from .predict import ModelPredictor


@dataclass
class BatchPredictionConfig:
    """批量预测配置"""
    model_paths: List[str]
    data_paths: List[str] 
    output_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    sequence_length: int = 96
    max_workers: int = 4
    save_individual_results: bool = True
    ensemble_method: str = "mean"  # mean, median, weighted


class BatchPredictor:
    """批量预测器"""
    
    def __init__(self, config: BatchPredictionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'batch_prediction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self) -> Dict[str, torch.nn.Module]:
        """加载所有模型"""
        models = {}
        
        for model_path in self.config.model_paths:
            try:
                model_name = Path(model_path).stem
                
                # 加载模型检查点
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 创建模型
                if 'model_config' in checkpoint:
                    model = create_model(checkpoint['model_config'])
                else:
                    # 默认配置
                    model_config = {
                        'input_size': 1,
                        'hidden_size': 64,
                        'num_layers': 2,
                        'dropout': 0.1,
                        'num_heads': 8,
                        'sequence_length': self.config.sequence_length
                    }
                    model = create_model(model_config)
                
                # 加载权重
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(self.device)
                model.eval()
                models[model_name] = model
                
                self.logger.info(f"成功加载模型: {model_name}")
                
            except Exception as e:
                self.logger.error(f"加载模型失败 {model_path}: {e}")
                
        return models
        
    def load_datasets(self) -> Dict[str, torch.utils.data.DataLoader]:
        """加载所有数据集"""
        dataloaders = {}
        
        for data_path in self.config.data_paths:
            try:
                data_name = Path(data_path).stem
                
                # 加载数据
                df = pd.read_csv(data_path)
                
                # 创建数据集
                dataset = IndustrialTimeSeriesDataset(
                    data=df,
                    sequence_length=self.config.sequence_length,
                    target_columns=['value'],
                    is_train=False
                )
                
                # 创建数据加载器
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=0
                )
                
                dataloaders[data_name] = dataloader
                self.logger.info(f"成功加载数据集: {data_name}")
                
            except Exception as e:
                self.logger.error(f"加载数据集失败 {data_path}: {e}")
                
        return dataloaders
        
    def predict_single(
        self, 
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        model_name: str,
        data_name: str
    ) -> Dict[str, Any]:
        """单个模型单个数据集的预测"""
        predictions = []
        targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                
                # 预测
                pred = model(X)
                
                predictions.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        # 合并结果
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # 计算指标
        metrics = calculate_metrics(targets, predictions)
        
        prediction_time = time.time() - start_time
        
        result = {
            'model_name': model_name,
            'data_name': data_name,
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics,
            'prediction_time': prediction_time,
            'num_samples': len(predictions)
        }
        
        self.logger.info(
            f"完成预测 - 模型: {model_name}, 数据: {data_name}, "
            f"样本数: {len(predictions)}, 耗时: {prediction_time:.2f}s"
        )
        
        return result
        
    def predict_batch(self) -> Dict[str, Any]:
        """批量预测"""
        self.logger.info("开始批量预测...")
        
        # 加载模型和数据
        models = self.load_models()
        dataloaders = self.load_datasets()
        
        if not models:
            raise ValueError("没有成功加载任何模型")
        if not dataloaders:
            raise ValueError("没有成功加载任何数据集")
            
        # 创建预测任务
        tasks = []
        for model_name, model in models.items():
            for data_name, dataloader in dataloaders.items():
                tasks.append((model, dataloader, model_name, data_name))
                
        # 并行预测
        results = {}
        
        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.predict_single, *task): task 
                    for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = future.result()
                        key = f"{result['model_name']}_{result['data_name']}"
                        results[key] = result
                    except Exception as e:
                        self.logger.error(f"预测任务失败 {task}: {e}")
        else:
            for task in tasks:
                try:
                    result = self.predict_single(*task)
                    key = f"{result['model_name']}_{result['data_name']}"
                    results[key] = result
                except Exception as e:
                    self.logger.error(f"预测任务失败 {task}: {e}")
                    
        self.logger.info(f"批量预测完成，共完成 {len(results)} 个预测任务")
        
        return results
        
    def create_ensemble_predictions(
        self, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建集成预测"""
        # 按数据集分组
        data_groups = {}
        for key, result in results.items():
            data_name = result['data_name']
            if data_name not in data_groups:
                data_groups[data_name] = []
            data_groups[data_name].append(result)
            
        ensemble_results = {}
        
        for data_name, group_results in data_groups.items():
            if len(group_results) < 2:
                continue
                
            # 获取所有预测和目标
            all_predictions = [r['predictions'] for r in group_results]
            targets = group_results[0]['targets']  # 目标都相同
            
            # 集成预测
            if self.config.ensemble_method == "mean":
                ensemble_pred = np.mean(all_predictions, axis=0)
            elif self.config.ensemble_method == "median":
                ensemble_pred = np.median(all_predictions, axis=0)
            elif self.config.ensemble_method == "weighted":
                # 基于性能加权
                weights = []
                for result in group_results:
                    mae = result['metrics']['mae']
                    weight = 1.0 / (mae + 1e-8)  # 避免除零
                    weights.append(weight)
                    
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
                
            # 计算集成指标
            ensemble_metrics = calculate_metrics(targets, ensemble_pred)
            
            ensemble_results[data_name] = {
                'data_name': data_name,
                'ensemble_method': self.config.ensemble_method,
                'num_models': len(group_results),
                'predictions': ensemble_pred,
                'targets': targets,
                'metrics': ensemble_metrics,
                'individual_results': group_results
            }
            
        return ensemble_results
        
    def save_results(
        self, 
        results: Dict[str, Any],
        ensemble_results: Dict[str, Any] = None
    ):
        """保存结果"""
        # 保存个体结果
        if self.config.save_individual_results:
            for key, result in results.items():
                # 保存预测结果
                pred_df = pd.DataFrame({
                    'predictions': result['predictions'].flatten(),
                    'targets': result['targets'].flatten()
                })
                pred_path = self.output_dir / f"{key}_predictions.csv"
                pred_df.to_csv(pred_path, index=False)
                
                # 保存指标
                metrics_path = self.output_dir / f"{key}_metrics.json"
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(result['metrics'], f, indent=2, ensure_ascii=False)
                    
        # 保存集成结果
        if ensemble_results:
            for data_name, result in ensemble_results.items():
                # 保存集成预测
                ensemble_df = pd.DataFrame({
                    'ensemble_predictions': result['predictions'].flatten(),
                    'targets': result['targets'].flatten()
                })
                ensemble_path = self.output_dir / f"ensemble_{data_name}_predictions.csv"
                ensemble_df.to_csv(ensemble_path, index=False)
                
                # 保存集成指标
                ensemble_metrics_path = self.output_dir / f"ensemble_{data_name}_metrics.json"
                with open(ensemble_metrics_path, 'w', encoding='utf-8') as f:
                    ensemble_info = {
                        'ensemble_method': result['ensemble_method'],
                        'num_models': result['num_models'],
                        'metrics': result['metrics']
                    }
                    json.dump(ensemble_info, f, indent=2, ensure_ascii=False)
                    
        # 创建汇总报告
        self.create_summary_report(results, ensemble_results)
        
    def create_summary_report(
        self, 
        results: Dict[str, Any],
        ensemble_results: Dict[str, Any] = None
    ):
        """创建汇总报告"""
        report = {
            'batch_prediction_summary': {
                'total_tasks': len(results),
                'successful_tasks': len(results),
                'ensemble_results': len(ensemble_results) if ensemble_results else 0
            },
            'individual_results': {},
            'ensemble_results': {}
        }
        
        # 个体结果汇总
        for key, result in results.items():
            report['individual_results'][key] = {
                'model_name': result['model_name'],
                'data_name': result['data_name'],
                'num_samples': result['num_samples'],
                'prediction_time': result['prediction_time'],
                'metrics': result['metrics']
            }
            
        # 集成结果汇总
        if ensemble_results:
            for data_name, result in ensemble_results.items():
                report['ensemble_results'][data_name] = {
                    'ensemble_method': result['ensemble_method'],
                    'num_models': result['num_models'],
                    'metrics': result['metrics']
                }
                
        # 保存报告
        report_path = self.output_dir / 'batch_prediction_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"批量预测报告已保存到: {report_path}")
        
    def run(self) -> Dict[str, Any]:
        """运行批量预测"""
        try:
            # 执行批量预测
            results = self.predict_batch()
            
            # 创建集成预测
            ensemble_results = None
            if len(set(r['data_name'] for r in results.values())) > 0:
                ensemble_results = self.create_ensemble_predictions(results)
                
            # 保存结果
            self.save_results(results, ensemble_results)
            
            self.logger.info("批量预测任务完成!")
            
            return {
                'individual_results': results,
                'ensemble_results': ensemble_results or {}
            }
            
        except Exception as e:
            self.logger.error(f"批量预测失败: {e}")
            raise


def main():
    """批量预测主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量预测工具")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='批量预测配置文件路径'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
        
    config = BatchPredictionConfig(**config_dict)
    
    # 执行批量预测
    predictor = BatchPredictor(config)
    results = predictor.run()
    
    print("批量预测完成!")
    print(f"个体预测任务: {len(results['individual_results'])}")
    print(f"集成预测任务: {len(results['ensemble_results'])}")


if __name__ == "__main__":
    main()
