"""
早停机制模块
用于防止模型过拟合，在验证损失不再改善时停止训练
"""

import numpy as np
from typing import Optional, Union


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, 
                 patience: int = 7,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True,
                 mode: str = 'min',
                 baseline: Optional[float] = None,
                 verbose: bool = True):
        """
        早停机制初始化
        
        Args:
            patience: 容忍的epoch数量，超过这个数量没有改善就停止
            min_delta: 改善的最小阈值
            restore_best_weights: 是否在停止时恢复最佳权重
            mode: 'min' 表示监控指标越小越好，'max' 表示越大越好
            baseline: 基线值，只有超过这个值才认为是改善
            verbose: 是否打印详细信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.baseline = baseline
        self.verbose = verbose
        
        # 内部状态
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        
        # 根据mode设置比较函数
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"不支持的mode: {mode}, 应该是 'min' 或 'max'")
    
    def step(self, current: float, model_weights: Optional[dict] = None) -> bool:
        """
        执行一步早停检查
        
        Args:
            current: 当前监控指标值
            model_weights: 当前模型权重（用于保存最佳权重）
            
        Returns:
            是否应该停止训练
        """
        if self.best is None:
            # 第一次调用，初始化最佳值
            self.best = current
            if model_weights is not None:
                self.best_weights = model_weights.copy()
            return False
        
        # 检查是否改善
        if self._is_improvement(current):
            self.best = current
            self.wait = 0
            if model_weights is not None:
                self.best_weights = model_weights.copy()
            if self.verbose:
                print(f"EarlyStopping: 指标改善到 {current:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"EarlyStopping: 指标未改善 {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.verbose:
                    print(f"EarlyStopping: 触发早停，最佳值为 {self.best:.6f}")
                return True
        
        return False
    
    def _is_improvement(self, current: float) -> bool:
        """检查是否是改善"""
        if self.baseline is not None:
            if self.mode == 'min':
                if current >= self.baseline:
                    return False
            else:  # mode == 'max'
                if current <= self.baseline:
                    return False
        
        return self.monitor_op(current - self.min_delta, self.best)
    
    def reset(self):
        """重置早停状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
    
    def get_best_weights(self) -> Optional[dict]:
        """获取最佳权重"""
        return self.best_weights
    
    def get_best_value(self) -> Optional[float]:
        """获取最佳指标值"""
        return self.best


class ReduceLROnPlateau:
    """学习率自适应调整"""
    
    def __init__(self,
                 optimizer,
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 10,
                 verbose: bool = False,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: Union[float, list] = 0,
                 eps: float = 1e-8):
        """
        学习率自适应调整初始化
        
        Args:
            optimizer: 优化器
            mode: 'min' 表示监控指标越小越好，'max' 表示越大越好
            factor: 学习率缩放因子
            patience: 容忍的epoch数量
            verbose: 是否打印详细信息
            threshold: 改善阈值
            threshold_mode: 阈值模式 'rel' 或 'abs'
            cooldown: 冷却期
            min_lr: 最小学习率
            eps: 数值稳定性参数
        """
        if factor >= 1.0:
            raise ValueError('Factor应该 < 1.0.')
        self.factor = factor
        
        # 将min_lr转换为列表
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # 比当前最佳值更差的值
        self.eps = eps
        self.last_epoch = 0
        self.optimizer = optimizer
        
        self._init_is_better(mode=mode, threshold=threshold,
                           threshold_mode=threshold_mode)
        self._reset()
    
    def _reset(self):
        """重置状态"""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def step(self, metrics, epoch=None):
        """执行一步学习率调整"""
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            if epoch < self.last_epoch:
                raise ValueError("step epoch应该递增")
        self.last_epoch = epoch
        
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # 冷却期内忽略坏epochs
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self, epoch):
        """减少学习率"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Epoch {epoch:5d}: 学习率从 {old_lr:.4e} 减少到 {new_lr:.4e}')
    
    @property
    def in_cooldown(self):
        """是否在冷却期"""
        return self.cooldown_counter > 0
    
    def is_better(self, a, best):
        """判断a是否比best更好"""
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold
    
    def _init_is_better(self, mode, threshold, threshold_mode):
        """初始化is_better函数"""
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
        
        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')
        
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
