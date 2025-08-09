"""
模型工厂和工具函数
提供模型创建、参数统计、模型保存加载等功能
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .fusion_model import LSTMTransformerFusion, AdvancedLSTMTransformerFusion

logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(config: Dict[str, Any], model_type: str = "basic") -> nn.Module:
        """
        创建模型实例
        
        Args:
            config: 模型配置
            model_type: 模型类型 ("basic" 或 "advanced")
            
        Returns:
            model: 模型实例
        """
        # 自动设置输入维度
        if config['model']['lstm']['input_size'] is None:
            # 从数据配置中推断
            if 'feature_count' in config.get('data', {}):
                config['model']['lstm']['input_size'] = config['data']['feature_count']
            else:
                logger.warning("输入维度未指定，使用默认值8")
                config['model']['lstm']['input_size'] = 8
        
        if model_type.lower() == "basic":
            model = LSTMTransformerFusion(config)
        elif model_type.lower() == "advanced":
            model = AdvancedLSTMTransformerFusion(config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        logger.info(f"创建{model_type}模型成功")
        return model
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        统计模型参数数量
        
        Args:
            model: 模型实例
            
        Returns:
            total_params: 总参数数量
            trainable_params: 可训练参数数量
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
    @staticmethod
    def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = None):
        """
        打印模型摘要
        
        Args:
            model: 模型实例
            input_shape: 输入形状 (不包含batch维度)
        """
        total_params, trainable_params = ModelFactory.count_parameters(model)
        
        print("=" * 80)
        print("模型摘要")
        print("=" * 80)
        print(f"模型类型: {model.__class__.__name__}")
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        if input_shape:
            print(f"输入形状: {input_shape}")
            
            # 创建虚拟输入测试模型
            try:
                batch_size = 1
                test_input = torch.randn(batch_size, *input_shape)
                model.eval()
                with torch.no_grad():
                    output = model(test_input)
                print(f"输出形状: {output.shape[1:]}")  # 不显示batch维度
            except Exception as e:
                print(f"无法测试模型输出: {e}")
        
        print("=" * 80)
    
    @staticmethod
    def analyze_model_complexity(model: nn.Module) -> Dict[str, Any]:
        """
        分析模型复杂度
        
        Args:
            model: 模型实例
            
        Returns:
            complexity_info: 复杂度信息字典
        """
        # 统计不同类型的层
        layer_counts = {}
        parameter_counts = {}
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if module_type not in layer_counts:
                layer_counts[module_type] = 0
                parameter_counts[module_type] = 0
            
            layer_counts[module_type] += 1
            parameter_counts[module_type] += sum(p.numel() for p in module.parameters())
        
        total_params, trainable_params = ModelFactory.count_parameters(model)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_counts': layer_counts,
            'parameter_counts': parameter_counts,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }


class ModelManager:
    """模型管理器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
    def save_model(self, filepath: Path, save_config: bool = True, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   epoch: Optional[int] = None, 
                   metrics: Optional[Dict[str, float]] = None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
            save_config: 是否保存配置
            optimizer: 优化器状态
            epoch: 训练轮次
            metrics: 评估指标
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__
        }
        
        if save_config:
            save_dict['config'] = self.config
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            save_dict['epoch'] = epoch
            
        if metrics is not None:
            save_dict['metrics'] = metrics
        
        torch.save(save_dict, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: Path, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            optimizer: 优化器实例
            device: 设备
            
        Returns:
            load_info: 加载信息
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        load_info = {
            'model_class': checkpoint.get('model_class', 'Unknown'),
            'epoch': checkpoint.get('epoch', None),
            'metrics': checkpoint.get('metrics', None)
        }
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            load_info['optimizer_loaded'] = True
        
        logger.info(f"模型已从 {filepath} 加载")
        return load_info
    
    def export_to_torchscript(self, filepath: Path, example_input: torch.Tensor):
        """
        导出为TorchScript格式
        
        Args:
            filepath: 导出路径
            example_input: 示例输入
        """
        self.model.eval()
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(filepath)
        logger.info(f"TorchScript模型已保存到: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        factory = ModelFactory()
        complexity_info = factory.analyze_model_complexity(self.model)
        
        return {
            'model_class': self.model.__class__.__name__,
            'complexity': complexity_info,
            'config': self.config
        }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    验证模型配置的有效性
    
    Args:
        config: 模型配置
        
    Returns:
        is_valid: 配置是否有效
    """
    try:
        # 检查必需的配置项
        required_keys = ['model', 'data']
        for key in required_keys:
            if key not in config:
                logger.error(f"缺少必需的配置项: {key}")
                return False
        
        # 检查模型配置
        model_config = config['model']
        required_model_keys = ['lstm', 'transformer', 'prediction_head']
        for key in required_model_keys:
            if key not in model_config:
                logger.error(f"缺少模型配置项: {key}")
                return False
        
        # 检查LSTM配置
        lstm_config = model_config['lstm']
        if lstm_config.get('hidden_size', 0) <= 0:
            logger.error("LSTM隐藏维度必须大于0")
            return False
        
        # 检查Transformer配置
        transformer_config = model_config['transformer']
        d_model = transformer_config.get('d_model', 0)
        nhead = transformer_config.get('nhead', 0)
        
        if d_model <= 0:
            logger.error("Transformer模型维度必须大于0")
            return False
        
        if nhead <= 0 or d_model % nhead != 0:
            logger.error("注意力头数必须大于0且能整除模型维度")
            return False
        
        # 检查数据配置
        data_config = config['data']
        seq_len = data_config.get('sequence_length', 0)
        pred_len = data_config.get('prediction_length', 0)
        
        if seq_len <= 0 or pred_len <= 0:
            logger.error("序列长度和预测长度必须大于0")
            return False
        
        logger.info("模型配置验证通过")
        return True
        
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        return False


def test_model_factory():
    """测试模型工厂"""
    # 创建测试配置
    config = {
        'data': {
            'sequence_length': 60,
            'prediction_length': 10,
            'feature_count': 8
        },
        'model': {
            'lstm': {
                'input_size': None,  # 将由工厂自动设置
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'bidirectional': True
            },
            'transformer': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'activation': 'gelu'
            },
            'prediction_head': {
                'hidden_dims': [128, 64],
                'dropout': 0.1,
                'output_dim': 2
            }
        }
    }
    
    # 验证配置
    print("验证配置...")
    is_valid = validate_model_config(config)
    print(f"配置有效性: {is_valid}")
    
    if is_valid:
        # 创建模型
        print("\n创建基础模型...")
        factory = ModelFactory()
        model = factory.create_model(config, "basic")
        
        # 打印模型摘要
        input_shape = (config['data']['sequence_length'], config['data']['feature_count'])
        factory.print_model_summary(model, input_shape)
        
        # 分析复杂度
        print("\n分析模型复杂度...")
        complexity_info = factory.analyze_model_complexity(model)
        print(f"模型复杂度: {complexity_info}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_model_factory()
