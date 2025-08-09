"""
LSTM + Transformer 融合模型训练脚本
支持命令行参数、模型保存、验证指标、早停等功能
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import TimeSeriesDataModule
from src.models.model_utils import ModelFactory, ModelManager
from src.utils.pai_dsw_utils import setup_pai_dsw_environment, get_pai_dsw_paths
from src.training.trainer import Trainer
from src.training.metrics import MetricsCalculator
from src.training.early_stopping import EarlyStopping


def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger('LSTMTransformerTraining')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device() -> torch.device:
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """创建优化器"""
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Optional[object]:
    """创建学习率调度器"""
    scheduler_name = config['training'].get('scheduler', None)
    
    if scheduler_name is None:
        return None
    
    scheduler_params = config['training'].get('scheduler_params', {})
    
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = scheduler_params.get('T_max', config['training']['num_epochs'])
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
    
    return scheduler


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """创建损失函数"""
    loss_name = config['training']['loss_function']
    
    if loss_name == 'MSELoss':
        criterion = nn.MSELoss()
    elif loss_name == 'L1Loss':
        criterion = nn.L1Loss()
    elif loss_name == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    elif loss_name == 'HuberLoss':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")
    
    return criterion


def save_training_info(config: Dict[str, Any], model_info: Dict[str, Any], 
                      save_path: str) -> None:
    """保存训练信息"""
    training_info = {
        'config': config,
        'model_info': model_info,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'pytorch_version': torch.__version__,
        'device': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
    }
    
    info_path = os.path.join(save_path, 'training_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, ensure_ascii=False, indent=2)


def train_model(config: Dict[str, Any], logger: logging.Logger) -> None:
    """训练模型主函数"""
    
    # 设置设备
    device = setup_device()
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建数据模块
    logger.info("准备数据...")
    data_module = TimeSeriesDataModule(config)
    data_module.setup()
    
    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # 获取特征维度
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[-1]  # 获取特征维度
    
    # 更新配置中的输入维度
    config['model']['lstm']['input_size'] = input_size
    logger.info(f"输入特征维度: {input_size}")
    
    # 创建模型
    logger.info("创建模型...")
    model_factory = ModelFactory()
    model = model_factory.create_model('lstm_transformer', config)
    model.to(device)
    
    # 模型信息
    model_manager = ModelManager()
    model_info = model_manager.get_model_info(model)
    logger.info(f"模型参数数量: {model_info['total_params']:,}")
    logger.info(f"可训练参数数量: {model_info['trainable_params']:,}")
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    logger.info(f"优化器: {config['training']['optimizer']}")
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config)
    if scheduler:
        logger.info(f"学习率调度器: {config['training']['scheduler']}")
    
    # 创建损失函数
    criterion = create_loss_function(config)
    logger.info(f"损失函数: {config['training']['loss_function']}")
    
    # 创建早停机制
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        min_delta=1e-6,
        restore_best_weights=True
    )
    
    # 创建指标计算器
    metrics_calculator = MetricsCalculator()
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        metrics_calculator=metrics_calculator,
        early_stopping=early_stopping
    )
    
    # 设置保存路径
    model_save_path = config['output']['model_save_path']
    results_save_path = config['output']['results_save_path']
    checkpoint_path = config['output']['checkpoint_path']
    
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 保存训练信息
    save_training_info(config, model_info, results_save_path)
    
    # 开始训练
    logger.info("开始训练...")
    logger.info("=" * 80)
    
    training_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    
    # 训练完成
    logger.info("=" * 80)
    logger.info("训练完成!")
    
    # 保存最终模型
    final_model_path = os.path.join(model_save_path, 'final_model.pt')
    model_manager.save_model(model, final_model_path, config, model_info)
    logger.info(f"最终模型已保存到: {final_model_path}")
    
    # 如果有最佳模型，保存最佳模型
    if hasattr(trainer, 'best_model_state'):
        best_model_path = os.path.join(model_save_path, 'best_model.pt')
        torch.save({
            'model_state_dict': trainer.best_model_state,
            'config': config,
            'model_info': model_info,
            'training_history': training_history
        }, best_model_path)
        logger.info(f"最佳模型已保存到: {best_model_path}")
    
    # 保存训练历史
    history_path = os.path.join(results_save_path, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    # 测试集评估
    if test_loader:
        logger.info("在测试集上评估模型...")
        test_metrics = trainer.evaluate(test_loader)
        logger.info("测试集结果:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.6f}")
        
        # 保存测试结果
        test_results_path = os.path.join(results_save_path, 'test_results.json')
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"所有结果已保存到: {results_save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LSTM + Transformer 融合模型训练')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='强制使用的设备')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 不存在!")
        sys.exit(1)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置PAI-DSW环境
    try:
        setup_pai_dsw_environment()
        paths = get_pai_dsw_paths()
        print("PAI-DSW环境设置成功")
        print(f"工作空间路径: {paths['workspace_path']}")
    except Exception as e:
        print(f"PAI-DSW环境设置失败: {e}")
        print("继续使用本地环境...")
    
    # 设置日志
    log_dir = config['output']['results_save_path']
    logger = setup_logging(log_dir, args.log_level)
    
    # 强制设备
    if args.device:
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.device == 'cuda' else ''
    
    logger.info("开始LSTM + Transformer融合模型训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"日志级别: {args.log_level}")
    
    try:
        # 训练模型
        train_model(config, logger)
        logger.info("训练任务成功完成!")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
