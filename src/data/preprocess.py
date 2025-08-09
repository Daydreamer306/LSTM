"""
数据预处理主脚本
读取train.csv和time_ranges.json，进行特征工程和数据划分
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pai_dsw_utils import pai_dsw_utils
from src.data.dataset import TimeSeriesDataModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = pai_dsw_utils.get_workspace_path("configs/lstm_transformer.yaml")
    
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"配置加载完成: {config_path}")
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='工业时序预测数据预处理')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径 (train.csv)')
    parser.add_argument('--time_ranges', type=str, default=None,
                       help='时间范围文件路径 (time_ranges.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--save_processed', action='store_true',
                       help='保存预处理后的数据')
    
    args = parser.parse_args()
    
    # 初始化PAI-DSW环境
    pai_dsw_utils.setup_directories()
    pai_dsw_utils.log_environment_info()
    
    # 加载配置
    config = load_config(args.config)
    
    # 更新配置中的路径（如果通过命令行参数指定）
    if args.data_path:
        config['data']['train_file'] = args.data_path
    if args.time_ranges:
        config['data']['time_ranges_file'] = args.time_ranges
    if args.output_dir:
        config['output']['results_save_path'] = args.output_dir
    
    logger.info("=== 开始数据预处理 ===")
    
    try:
        # 检查数据文件
        file_status = pai_dsw_utils.check_data_files()
        missing_files = [f for f, exists in file_status.items() if not exists]
        
        if missing_files:
            logger.error(f"缺失数据文件: {missing_files}")
            logger.info("请确保以下文件存在:")
            logger.info(f"  - {pai_dsw_utils.get_data_path('train.csv')}")
            logger.info(f"  - {pai_dsw_utils.get_data_path('time_ranges.json')}")
            return
        
        # 创建数据模块
        logger.info("创建数据模块...")
        data_module = TimeSeriesDataModule(config)
        
        # 创建数据集
        logger.info("创建数据集...")
        train_dataset, val_dataset, test_dataset = data_module.create_datasets()
        
        # 打印数据集信息
        logger.info(f"训练集: {len(train_dataset)} 个样本")
        logger.info(f"验证集: {len(val_dataset)} 个样本") 
        logger.info(f"测试集: {len(test_dataset)} 个样本")
        
        # 获取样本形状
        sample_x, sample_y = train_dataset[0]
        logger.info(f"输入形状: {sample_x.shape}")
        logger.info(f"目标形状: {sample_y.shape}")
        
        # 保存预处理后的数据
        if args.save_processed:
            logger.info("保存预处理数据...")
            data_module.save_processed_data(train_dataset, val_dataset, test_dataset)
        
        # 创建数据加载器进行测试
        logger.info("测试数据加载器...")
        batch_size = config['training']['batch_size']
        train_loader, val_loader, test_loader = data_module.create_dataloaders(batch_size)
        
        # 测试一个批次
        for batch_x, batch_y in train_loader:
            logger.info(f"批次输入形状: {batch_x.shape}")
            logger.info(f"批次目标形状: {batch_y.shape}")
            break
        
        logger.info("=== 数据预处理完成 ===")
        
        # 数据统计和可视化
        generate_data_report(data_module, train_dataset, val_dataset, test_dataset, config)
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_data_report(data_module, train_dataset, val_dataset, test_dataset, config):
    """生成数据报告和可视化"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        logger.info("生成数据分析报告...")
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('工业时序数据预处理报告', fontsize=16, fontweight='bold')
        
        # 1. 数据集大小分布
        ax = axes[0, 0]
        sizes = [len(train_dataset), len(val_dataset), len(test_dataset)]
        labels = ['训练集', '验证集', '测试集']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('数据集划分比例')
        
        # 2. 特征分布（取第一个样本）
        ax = axes[0, 1]
        sample_x, _ = train_dataset[0]
        feature_means = sample_x.mean(dim=0).numpy()
        ax.bar(range(len(feature_means)), feature_means)
        ax.set_title('特征均值分布')
        ax.set_xlabel('特征索引')
        ax.set_ylabel('均值')
        
        # 3. 时间序列长度
        ax = axes[0, 2]
        sequence_info = {
            '输入序列长度': config['data']['sequence_length'],
            '预测序列长度': config['data']['prediction_length']
        }
        ax.bar(sequence_info.keys(), sequence_info.values(), color=['#ff7f0e', '#2ca02c'])
        ax.set_title('序列长度配置')
        ax.set_ylabel('时间步数')
        
        # 4. 训练数据样例
        ax = axes[1, 0]
        train_x, train_y = train_dataset.get_full_sequence()
        # 显示前1000个时间步
        max_steps = min(1000, train_x.shape[0])
        for i, target_name in enumerate(data_module.target_columns):
            ax.plot(train_y[:max_steps, i].numpy(), label=target_name, alpha=0.7)
        ax.set_title('目标变量时序图 (训练集前1000步)')
        ax.set_xlabel('时间步')
        ax.set_ylabel('数值')
        ax.legend()
        
        # 5. 数据质量检查
        ax = axes[1, 1]
        # 计算数据统计
        train_x_flat = train_x.view(-1).numpy()
        stats = {
            '均值': np.mean(train_x_flat),
            '标准差': np.std(train_x_flat),
            '最小值': np.min(train_x_flat),
            '最大值': np.max(train_x_flat)
        }
        ax.bar(stats.keys(), stats.values())
        ax.set_title('训练数据统计')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. 归一化效果
        ax = axes[1, 2]
        if data_module.scaler is not None:
            # 显示归一化前后的分布对比
            ax.hist(train_x_flat, bins=50, alpha=0.7, label='归一化后')
            ax.set_title('数据分布 (归一化后)')
            ax.set_xlabel('数值')
            ax.set_ylabel('频次')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '未使用归一化', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('归一化状态')
        
        plt.tight_layout()
        
        # 保存报告
        report_path = pai_dsw_utils.get_results_path("data_preprocessing_report.png")
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        logger.info(f"数据分析报告已保存: {report_path}")
        
        # 保存文本报告
        save_text_report(data_module, train_dataset, val_dataset, test_dataset, config)
        
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib或seaborn未安装，跳过可视化")
    except Exception as e:
        logger.error(f"生成数据报告失败: {e}")


def save_text_report(data_module, train_dataset, val_dataset, test_dataset, config):
    """保存文本格式的数据报告"""
    report_path = pai_dsw_utils.get_results_path("data_preprocessing_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 工业时序预测数据预处理报告 ===\n\n")
        
        # 基本信息
        f.write("## 数据基本信息\n")
        f.write(f"训练集样本数: {len(train_dataset)}\n")
        f.write(f"验证集样本数: {len(val_dataset)}\n")
        f.write(f"测试集样本数: {len(test_dataset)}\n")
        
        # 特征信息
        f.write(f"\n## 特征信息\n")
        f.write(f"特征列数: {len(data_module.feature_columns)}\n")
        f.write(f"目标列数: {len(data_module.target_columns)}\n")
        f.write(f"特征列: {', '.join(data_module.feature_columns)}\n")
        f.write(f"目标列: {', '.join(data_module.target_columns)}\n")
        
        # 序列配置
        f.write(f"\n## 序列配置\n")
        f.write(f"输入序列长度: {config['data']['sequence_length']}\n")
        f.write(f"预测序列长度: {config['data']['prediction_length']}\n")
        
        # 数据预处理
        f.write(f"\n## 数据预处理\n")
        f.write(f"是否归一化: {config['data']['normalize']}\n")
        if config['data']['normalize']:
            f.write(f"归一化方法: {config['data']['scaler_type']}\n")
        
        # 数据划分
        f.write(f"\n## 数据划分\n")
        f.write(f"训练集比例: {config['data']['train_ratio']}\n")
        f.write(f"验证集比例: {config['data']['val_ratio']}\n")
        f.write(f"测试集比例: {config['data']['test_ratio']}\n")
        
        # 样本形状
        sample_x, sample_y = train_dataset[0]
        f.write(f"\n## 样本形状\n")
        f.write(f"输入形状: {sample_x.shape}\n")
        f.write(f"目标形状: {sample_y.shape}\n")
    
    logger.info(f"文本报告已保存: {report_path}")


if __name__ == "__main__":
    main()
