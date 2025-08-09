"""
训练器模块
负责模型训练、验证和测试的核心逻辑
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Any, Optional, List
import logging


class Trainer:
    """模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[object] = None,
                 device: torch.device = torch.device('cpu'),
                 config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None,
                 metrics_calculator: Optional[object] = None,
                 early_stopping: Optional[object] = None):
        """
        训练器初始化
        
        Args:
            model: 要训练的模型
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 计算设备
            config: 配置字典
            logger: 日志记录器
            metrics_calculator: 指标计算器
            early_stopping: 早停机制
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.logger = logger
        self.metrics_calculator = metrics_calculator
        self.early_stopping = early_stopping
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        self.learning_rates = []
        
        # 最佳模型状态
        self.best_model_state = None
        self.best_val_loss = float('inf')
        
        # 训练配置
        self.log_interval = self.config.get('output', {}).get('log_interval', 10)
        self.gradient_clip_val = self.config.get('training', {}).get('gradient_clip_val', None)
        self.val_interval = self.config.get('validation', {}).get('val_interval', 1)
        
    def _log(self, message: str, level: str = 'info') -> None:
        """记录日志"""
        if self.logger:
            getattr(self.logger, level.lower())(message)
        else:
            print(message)
    
    def _clip_gradients(self) -> None:
        """梯度裁剪"""
        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # 初始化指标累计器
        if self.metrics_calculator:
            epoch_predictions = []
            epoch_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 数据移动到设备
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # 调整输出维度以匹配目标
            if output.dim() == 3 and target.dim() == 2:
                # 如果输出是 (batch, seq_len, features)，取最后一个时间步
                output = output[:, -1, :]
            elif output.dim() == 3 and target.dim() == 3:
                # 如果都是3维，确保维度匹配
                if output.shape != target.shape:
                    output = output[:, -target.shape[1]:, :]
            
            # 计算损失
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            self._clip_gradients()
            
            # 参数更新
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测和真值用于指标计算
            if self.metrics_calculator:
                epoch_predictions.append(output.detach().cpu().numpy())
                epoch_targets.append(target.detach().cpu().numpy())
            
            # 日志输出
            if (batch_idx + 1) % self.log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                progress = 100. * (batch_idx + 1) / num_batches
                self._log(f'Epoch {epoch:3d} [{batch_idx+1:5d}/{num_batches:5d} '
                         f'({progress:6.2f}%)] Loss: {current_loss:.6f}')
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算epoch指标
        epoch_metrics = {'loss': avg_loss}
        if self.metrics_calculator:
            all_predictions = np.concatenate(epoch_predictions, axis=0)
            all_targets = np.concatenate(epoch_targets, axis=0)
            metrics = self.metrics_calculator.calculate_metrics(all_targets, all_predictions)
            epoch_metrics.update(metrics)
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # 初始化指标累计器
        if self.metrics_calculator:
            epoch_predictions = []
            epoch_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # 数据移动到设备
                data = data.to(self.device)
                target = target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 调整输出维度以匹配目标
                if output.dim() == 3 and target.dim() == 2:
                    output = output[:, -1, :]
                elif output.dim() == 3 and target.dim() == 3:
                    if output.shape != target.shape:
                        output = output[:, -target.shape[1]:, :]
                
                # 计算损失
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # 收集预测和真值用于指标计算
                if self.metrics_calculator:
                    epoch_predictions.append(output.cpu().numpy())
                    epoch_targets.append(target.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 计算epoch指标
        epoch_metrics = {'loss': avg_loss}
        if self.metrics_calculator:
            all_predictions = np.concatenate(epoch_predictions, axis=0)
            all_targets = np.concatenate(epoch_targets, axis=0)
            metrics = self.metrics_calculator.calculate_metrics(all_targets, all_predictions)
            epoch_metrics.update(metrics)
        
        return epoch_metrics
    
    def fit(self, 
            train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None,
            num_epochs: int = 100) -> Dict[str, List]:
        """训练模型"""
        
        self._log(f"开始训练 {num_epochs} 个epochs...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics['loss'])
            
            # 记录训练指标
            for metric, value in train_metrics.items():
                if metric not in self.train_metrics:
                    self.train_metrics[metric] = []
                self.train_metrics[metric].append(value)
            
            # 验证
            val_metrics = {}
            if val_loader is not None and epoch % self.val_interval == 0:
                val_metrics = self.validate_epoch(val_loader, epoch)
                self.val_losses.append(val_metrics['loss'])
                
                # 记录验证指标
                for metric, value in val_metrics.items():
                    if metric not in self.val_metrics:
                        self.val_metrics[metric] = []
                    self.val_metrics[metric].append(value)
                
                # 保存最佳模型
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_model_state = self.model.state_dict().copy()
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        monitor_metric = self.config.get('validation', {}).get('monitor_metric', 'loss')
                        if monitor_metric.startswith('val_'):
                            monitor_metric = monitor_metric[4:]  # 去除 'val_' 前缀
                        
                        if monitor_metric in val_metrics:
                            self.scheduler.step(val_metrics[monitor_metric])
                        else:
                            self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 计算epoch用时
            epoch_time = time.time() - epoch_start_time
            
            # 输出epoch总结
            train_loss_str = f"Train Loss: {train_metrics['loss']:.6f}"
            val_loss_str = f"Val Loss: {val_metrics['loss']:.6f}" if val_metrics else ""
            lr_str = f"LR: {current_lr:.2e}"
            time_str = f"Time: {epoch_time:.2f}s"
            
            summary = f"Epoch {epoch:3d}/{num_epochs} - {train_loss_str}"
            if val_loss_str:
                summary += f" - {val_loss_str}"
            summary += f" - {lr_str} - {time_str}"
            
            self._log(summary)
            
            # 输出其他指标
            if len(train_metrics) > 1:  # 除了loss还有其他指标
                train_metrics_str = " - ".join([f"Train {k}: {v:.6f}" 
                                               for k, v in train_metrics.items() if k != 'loss'])
                self._log(f"         {train_metrics_str}")
            
            if val_metrics and len(val_metrics) > 1:
                val_metrics_str = " - ".join([f"Val {k}: {v:.6f}" 
                                             for k, v in val_metrics.items() if k != 'loss'])
                self._log(f"         {val_metrics_str}")
            
            # 早停检查
            if self.early_stopping is not None and val_metrics:
                should_stop = self.early_stopping.step(val_metrics['loss'])
                if should_stop:
                    self._log(f"早停触发，在第 {epoch} epoch 停止训练")
                    break
        
        # 训练完成
        total_time = time.time() - start_time
        self._log(f"训练完成! 总用时: {total_time:.2f}s")
        
        # 返回训练历史
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'learning_rates': self.learning_rates,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        return history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """在测试集上评估模型"""
        self._log("开始测试集评估...")
        
        # 如果有最佳模型状态，加载它
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self._log("已加载最佳模型权重进行评估")
        
        test_metrics = self.validate_epoch(test_loader, -1)
        
        return test_metrics
    
    def predict(self, data_loader: DataLoader) -> tuple:
        """预测"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                
                # 调整输出维度
                if output.dim() == 3 and target.dim() == 2:
                    output = output[:, -1, :]
                elif output.dim() == 3 and target.dim() == 3:
                    if output.shape != target.shape:
                        output = output[:, -target.shape[1]:, :]
                
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        return predictions, targets
