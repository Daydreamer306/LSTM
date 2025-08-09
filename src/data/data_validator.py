"""
数据验证和测试工具
用于验证数据完整性、格式正确性以及预处理结果
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pai_dsw_utils import pai_dsw_utils

logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_csv_file(self, file_path: Path) -> bool:
        """验证CSV文件格式和内容"""
        logger.info(f"验证CSV文件: {file_path}")
        
        try:
            # 检查文件存在性
            if not file_path.exists():
                logger.error(f"文件不存在: {file_path}")
                return False
            
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 基本检查
            checks = {
                'file_exists': True,
                'readable': True,
                'non_empty': len(df) > 0,
                'has_numeric_columns': len(df.select_dtypes(include=[np.number]).columns) > 0,
                'no_all_nan_columns': not df.isnull().all().any(),
                'reasonable_size': 1000 <= len(df) <= 1000000  # 合理的数据量范围
            }
            
            self.validation_results['csv_checks'] = checks
            
            # 数据质量检查
            quality_checks = {
                'missing_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)),
                'duplicate_rows': df.duplicated().sum(),
                'constant_columns': (df.nunique() == 1).sum(),
                'numeric_columns_count': len(df.select_dtypes(include=[np.number]).columns),
                'total_columns': len(df.columns)
            }
            
            self.validation_results['quality_checks'] = quality_checks
            
            # 打印结果
            logger.info(f"数据形状: {df.shape}")
            logger.info(f"数值列数量: {quality_checks['numeric_columns_count']}")
            logger.info(f"缺失值比例: {quality_checks['missing_percentage']:.2%}")
            logger.info(f"重复行数: {quality_checks['duplicate_rows']}")
            
            # 基本检查通过判断
            all_basic_passed = all(checks.values())
            quality_acceptable = (
                quality_checks['missing_percentage'] < 0.5 and  # 缺失值少于50%
                quality_checks['numeric_columns_count'] >= 2     # 至少2个数值列
            )
            
            return all_basic_passed and quality_acceptable
            
        except Exception as e:
            logger.error(f"CSV文件验证失败: {e}")
            return False
    
    def validate_json_file(self, file_path: Path) -> bool:
        """验证JSON文件格式和内容"""
        logger.info(f"验证JSON文件: {file_path}")
        
        try:
            import json
            
            if not file_path.exists():
                logger.warning(f"JSON文件不存在: {file_path}")
                return True  # JSON文件是可选的
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"JSON文件内容: {data}")
            return True
            
        except Exception as e:
            logger.error(f"JSON文件验证失败: {e}")
            return False
    
    def validate_target_columns(self, df: pd.DataFrame, target_columns: list) -> bool:
        """验证目标列是否存在"""
        logger.info(f"验证目标列: {target_columns}")
        
        missing_columns = [col for col in target_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"目标列不存在: {missing_columns}")
            logger.info(f"可用列: {list(df.columns)}")
            return False
        
        # 检查目标列数据质量
        for col in target_columns:
            col_data = df[col]
            if col_data.isnull().all():
                logger.error(f"目标列全为空: {col}")
                return False
            
            if col_data.nunique() == 1:
                logger.warning(f"目标列只有单一值: {col}")
        
        return True
    
    def validate_time_series_properties(self, df: pd.DataFrame, sequence_length: int, prediction_length: int) -> bool:
        """验证时序数据属性"""
        logger.info("验证时序数据属性...")
        
        min_required_length = sequence_length + prediction_length
        
        if len(df) < min_required_length:
            logger.error(f"数据长度不足: {len(df)} < {min_required_length}")
            return False
        
        # 检查时间序列的连续性（如果有时间列）
        time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        
        for time_col in time_columns:
            try:
                # 尝试转换为时间类型
                df[time_col] = pd.to_datetime(df[time_col])
                
                # 检查时间是否连续
                time_diff = df[time_col].diff().dropna()
                if not time_diff.nunique() == 1:  # 如果时间间隔不一致
                    logger.warning(f"时间列 {time_col} 间隔不一致")
                
                logger.info(f"时间列 {time_col} 验证通过")
                
            except:
                logger.warning(f"无法解析时间列: {time_col}")
        
        return True
    
    def run_comprehensive_validation(self, config: dict) -> bool:
        """运行综合验证"""
        logger.info("=== 开始数据综合验证 ===")
        
        # 获取文件路径
        train_file = pai_dsw_utils.get_data_path("train.csv")
        time_ranges_file = pai_dsw_utils.get_data_path("time_ranges.json")
        
        # 验证CSV文件
        csv_valid = self.validate_csv_file(train_file)
        if not csv_valid:
            return False
        
        # 验证JSON文件
        json_valid = self.validate_json_file(time_ranges_file)
        
        # 加载数据进行进一步验证
        try:
            df = pd.read_csv(train_file)
            
            # 验证目标列
            target_columns = config['data']['target_columns']
            targets_valid = self.validate_target_columns(df, target_columns)
            
            # 验证时序属性
            sequence_valid = self.validate_time_series_properties(
                df,
                config['data']['sequence_length'],
                config['data']['prediction_length']
            )
            
            # 综合评估
            all_valid = csv_valid and json_valid and targets_valid and sequence_valid
            
            logger.info(f"=== 验证结果: {'通过' if all_valid else '失败'} ===")
            
            return all_valid
            
        except Exception as e:
            logger.error(f"综合验证失败: {e}")
            return False
    
    def generate_validation_report(self) -> str:
        """生成验证报告"""
        report = ["=== 数据验证报告 ===\n"]
        
        if 'csv_checks' in self.validation_results:
            report.append("## CSV文件检查")
            for check, result in self.validation_results['csv_checks'].items():
                status = "✓" if result else "✗"
                report.append(f"{status} {check}: {result}")
            report.append("")
        
        if 'quality_checks' in self.validation_results:
            report.append("## 数据质量检查")
            for check, value in self.validation_results['quality_checks'].items():
                if isinstance(value, float):
                    report.append(f"• {check}: {value:.4f}")
                else:
                    report.append(f"• {check}: {value}")
            report.append("")
        
        return "\n".join(report)


def create_sample_data():
    """创建示例数据文件用于测试"""
    logger.info("创建示例数据文件...")
    
    # 创建示例train.csv
    np.random.seed(42)
    n_samples = 10000
    n_features = 8
    
    # 生成时间序列数据
    time_steps = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # 生成特征数据
    data = {
        'timestamp': time_steps,
        'feature_1': np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.normal(0, 0.1, n_samples),
        'feature_2': np.cos(np.linspace(0, 8*np.pi, n_samples)) + np.random.normal(0, 0.15, n_samples),
        'feature_3': np.random.normal(100, 20, n_samples),
        'feature_4': np.random.exponential(2, n_samples),
        'feature_5': np.random.uniform(0, 100, n_samples),
        'feature_6': np.cumsum(np.random.normal(0, 1, n_samples)),
    }
    
    # 生成目标变量（信号123和信号124）
    data['信号123'] = (
        data['feature_1'] * 10 + 
        data['feature_2'] * 5 + 
        np.random.normal(0, 2, n_samples)
    )
    
    data['信号124'] = (
        data['feature_3'] * 0.5 + 
        data['feature_4'] * 2 + 
        np.random.normal(0, 3, n_samples)
    )
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    
    train_file = pai_dsw_utils.get_data_path("train.csv")
    df.to_csv(train_file, index=False)
    logger.info(f"示例训练文件已创建: {train_file}")
    
    # 创建示例time_ranges.json
    time_ranges = {
        "start_time": "2023-01-01 00:00:00",
        "end_time": "2024-05-17 07:00:00", 
        "total_samples": n_samples,
        "frequency": "hourly",
        "continuous_segments": [
            {
                "start": 0,
                "end": n_samples//2,
                "description": "正常运行期"
            },
            {
                "start": n_samples//2,
                "end": n_samples,
                "description": "调整期"
            }
        ]
    }
    
    import json
    time_ranges_file = pai_dsw_utils.get_data_path("time_ranges.json")
    with open(time_ranges_file, 'w', encoding='utf-8') as f:
        json.dump(time_ranges, f, indent=2, ensure_ascii=False)
    
    logger.info(f"示例时间范围文件已创建: {time_ranges_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据验证和测试工具')
    parser.add_argument('--create_sample', action='store_true',
                       help='创建示例数据文件')
    parser.add_argument('--config', type=str, 
                       default=str(pai_dsw_utils.get_workspace_path("configs/lstm_transformer.yaml")),
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 初始化环境
    pai_dsw_utils.setup_directories()
    
    if args.create_sample:
        create_sample_data()
    
    # 加载配置
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 运行验证
    validator = DataValidator()
    is_valid = validator.run_comprehensive_validation(config)
    
    # 生成报告
    report = validator.generate_validation_report()
    print(report)
    
    # 保存报告
    report_path = pai_dsw_utils.get_results_path("data_validation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"验证报告已保存: {report_path}")
    
    if not is_valid:
        logger.error("数据验证失败，请检查数据文件")
        sys.exit(1)
    else:
        logger.info("数据验证通过！")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
