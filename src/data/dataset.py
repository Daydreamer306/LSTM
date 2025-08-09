"""
工业时序预测数据集类
支持滑动窗口、特征归一化、时间序列处理
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class IndustrialTimeSeriesDataset(Dataset):
    """工业时序预测数据集"""
    
    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int,
        prediction_length: int,
        target_columns: List[int],
        feature_columns: Optional[List[int]] = None,
        normalize: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data: 时序数据 (n_samples, n_features)
            sequence_length: 输入序列长度
            prediction_length: 预测序列长度
            target_columns: 目标列索引
            feature_columns: 特征列索引，None表示使用所有列
            normalize: 是否归一化
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.target_columns = target_columns
        self.feature_columns = feature_columns if feature_columns is not None else list(range(data.shape[1]))
        self.normalize = normalize
        
        # 计算有效样本数量
        self.n_samples = len(data) - sequence_length - prediction_length + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"数据长度不足: {len(data)}, 需要至少 {sequence_length + prediction_length} 个样本")
        
        logger.info(f"数据集初始化完成: {self.n_samples} 个样本, 序列长度: {sequence_length}, 预测长度: {prediction_length}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            x: 输入序列 (sequence_length, n_features)
            y: 目标序列 (prediction_length, n_targets)
        """
        # 输入序列
        start_idx = idx
        end_idx = idx + self.sequence_length
        x = self.data[start_idx:end_idx, self.feature_columns]
        
        # 目标序列
        target_start = end_idx
        target_end = target_start + self.prediction_length
        y = self.data[target_start:target_end, self.target_columns]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def get_full_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取完整序列用于可视化"""
        return torch.FloatTensor(self.data[:, self.feature_columns]), torch.FloatTensor(self.data[:, self.target_columns])


class TimeSeriesDataModule:
    """时序数据模块，处理数据加载、预处理、划分等"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据模块
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        self.scaler = None
        self.feature_columns = None
        self.target_columns = None
        
        # 路径配置
        from ..utils.pai_dsw_utils import pai_dsw_utils
        self.train_file = pai_dsw_utils.get_data_path("train.csv")
        self.time_ranges_file = pai_dsw_utils.get_data_path("time_ranges.json")
        
        logger.info(f"数据模块初始化: train_file={self.train_file}, time_ranges_file={self.time_ranges_file}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        logger.info(f"加载训练数据: {self.train_file}")
        
        if not self.train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {self.train_file}")
        
        # 读取CSV数据
        df = pd.read_csv(self.train_file)
        logger.info(f"原始数据形状: {df.shape}")
        logger.info(f"数据列: {list(df.columns)}")
        
        # 检查数据质量
        self._check_data_quality(df)
        
        return df
    
    def load_time_ranges(self) -> Dict[str, Any]:
        """加载时间范围信息"""
        import json
        
        if not self.time_ranges_file.exists():
            logger.warning(f"时间范围文件不存在: {self.time_ranges_file}")
            return {}
        
        logger.info(f"加载时间范围: {self.time_ranges_file}")
        
        with open(self.time_ranges_file, 'r', encoding='utf-8') as f:
            time_ranges = json.load(f)
        
        logger.info(f"时间范围信息: {time_ranges}")
        return time_ranges
    
    def _check_data_quality(self, df: pd.DataFrame):
        """检查数据质量"""
        # 检查缺失值
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"发现缺失值:\n{missing_counts[missing_counts > 0]}")
        
        # 检查目标列是否存在
        target_cols = self.data_config['target_columns']
        for col in target_cols:
            if col not in df.columns:
                raise ValueError(f"目标列不存在: {col}")
        
        # 检查数据类型
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        logger.info(f"数值列数量: {len(numeric_columns)}")
        
        # 统计信息
        logger.info(f"数据统计信息:\n{df.describe()}")
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """准备特征列和目标列"""
        target_columns = self.data_config['target_columns']
        
        # 获取特征列
        if self.data_config['feature_columns'] is None:
            # 使用除目标列外的所有数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col not in target_columns]
        else:
            feature_columns = self.data_config['feature_columns']
        
        # 验证列存在性
        for col in feature_columns + target_columns:
            if col not in df.columns:
                raise ValueError(f"列不存在: {col}")
        
        logger.info(f"特征列 ({len(feature_columns)}): {feature_columns}")
        logger.info(f"目标列 ({len(target_columns)}): {target_columns}")
        
        return feature_columns, target_columns
    
    def normalize_data(self, train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """归一化数据"""
        if not self.data_config['normalize']:
            return train_data, val_data, test_data
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        # 选择归一化方法
        scaler_type = self.data_config['scaler_type']
        if scaler_type == "StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler()
        elif scaler_type == "RobustScaler":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的归一化方法: {scaler_type}")
        
        # 基于训练数据拟合归一化器
        train_data_normalized = self.scaler.fit_transform(train_data)
        val_data_normalized = self.scaler.transform(val_data)
        test_data_normalized = self.scaler.transform(test_data)
        
        logger.info(f"数据归一化完成，使用 {scaler_type}")
        
        # 保存归一化器
        if self.data_config.get('save_scaler', True):
            self.save_scaler()
        
        return train_data_normalized, val_data_normalized, test_data_normalized
    
    def split_data_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按时间顺序划分数据"""
        n_total = len(df)
        train_ratio = self.data_config['train_ratio']
        val_ratio = self.data_config['val_ratio']
        test_ratio = self.data_config['test_ratio']
        
        # 验证比例
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"数据划分比例之和不为1: {train_ratio + val_ratio + test_ratio}")
        
        # 计算划分点
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"数据划分: 训练集 {len(train_df)} 样本, 验证集 {len(val_df)} 样本, 测试集 {len(test_df)} 样本")
        
        return train_df, val_df, test_df
    
    def create_datasets(self) -> Tuple[IndustrialTimeSeriesDataset, IndustrialTimeSeriesDataset, IndustrialTimeSeriesDataset]:
        """创建训练、验证、测试数据集"""
        # 加载原始数据
        df = self.load_raw_data()
        time_ranges = self.load_time_ranges()
        
        # 准备特征和目标列
        feature_columns, target_columns = self.prepare_features_and_targets(df)
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        # 按时间划分数据
        train_df, val_df, test_df = self.split_data_by_time(df)
        
        # 提取数据数组
        all_columns = feature_columns + target_columns
        train_data = train_df[all_columns].values.astype(np.float32)
        val_data = val_df[all_columns].values.astype(np.float32) 
        test_data = test_df[all_columns].values.astype(np.float32)
        
        # 归一化
        train_data, val_data, test_data = self.normalize_data(train_data, val_data, test_data)
        
        # 获取列索引
        feature_indices = list(range(len(feature_columns)))
        target_indices = list(range(len(feature_columns), len(all_columns)))
        
        # 创建数据集
        sequence_length = self.data_config['sequence_length']
        prediction_length = self.data_config['prediction_length']
        
        train_dataset = IndustrialTimeSeriesDataset(
            train_data, sequence_length, prediction_length,
            target_indices, feature_indices
        )
        
        val_dataset = IndustrialTimeSeriesDataset(
            val_data, sequence_length, prediction_length,
            target_indices, feature_indices
        )
        
        test_dataset = IndustrialTimeSeriesDataset(
            test_data, sequence_length, prediction_length,
            target_indices, feature_indices
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据加载器"""
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"数据加载器创建完成: batch_size={batch_size}")
        
        return train_loader, val_loader, test_loader
    
    def save_scaler(self):
        """保存数据归一化器"""
        if self.scaler is None:
            return
        
        from ..utils.pai_dsw_utils import pai_dsw_utils
        import joblib
        
        scaler_path = pai_dsw_utils.get_model_path("scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"归一化器已保存: {scaler_path}")
    
    def load_scaler(self):
        """加载数据归一化器"""
        from ..utils.pai_dsw_utils import pai_dsw_utils
        import joblib
        
        scaler_path = pai_dsw_utils.get_model_path("scaler.pkl")
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"归一化器已加载: {scaler_path}")
            return True
        return False
    
    def save_processed_data(self, train_dataset: IndustrialTimeSeriesDataset, 
                           val_dataset: IndustrialTimeSeriesDataset,
                           test_dataset: IndustrialTimeSeriesDataset):
        """保存预处理后的数据"""
        from ..utils.pai_dsw_utils import pai_dsw_utils
        
        # 保存数据集
        torch.save({
            'train_data': train_dataset.data,
            'val_data': val_dataset.data,
            'test_data': test_dataset.data,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'config': self.data_config
        }, pai_dsw_utils.get_data_path("processed_data.pt"))
        
        logger.info("预处理数据已保存")
