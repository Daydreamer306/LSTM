"""
快速数据检查脚本
用于快速验证数据文件是否存在和格式正确
"""

import sys
from pathlib import Path
import pandas as pd
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pai_dsw_utils import pai_dsw_utils


def quick_data_check():
    """快速数据检查"""
    print("=== 快速数据检查 ===")
    
    # 检查数据文件
    train_file = pai_dsw_utils.get_data_path("train.csv")
    time_ranges_file = pai_dsw_utils.get_data_path("time_ranges.json")
    
    print(f"检查训练文件: {train_file}")
    if train_file.exists():
        print("✓ train.csv 存在")
        try:
            df = pd.read_csv(train_file)
            print(f"  - 数据形状: {df.shape}")
            print(f"  - 列名: {list(df.columns)}")
            print(f"  - 数据类型: {df.dtypes.value_counts().to_dict()}")
            
            # 检查目标列
            target_columns = ["信号123", "信号124"]
            for col in target_columns:
                if col in df.columns:
                    print(f"  ✓ 目标列 '{col}' 存在")
                else:
                    print(f"  ✗ 目标列 '{col}' 不存在")
            
            # 数据质量
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            print(f"  - 缺失值比例: {missing_pct:.2%}")
            
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
    else:
        print("✗ train.csv 不存在")
    
    print(f"\n检查时间范围文件: {time_ranges_file}")
    if time_ranges_file.exists():
        print("✓ time_ranges.json 存在")
        try:
            with open(time_ranges_file, 'r', encoding='utf-8') as f:
                time_data = json.load(f)
            print(f"  - 内容: {time_data}")
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
    else:
        print("✗ time_ranges.json 不存在")
    
    # 检查目录结构
    print(f"\n检查目录结构:")
    required_dirs = ["data", "models", "results", "checkpoints"]
    for dir_name in required_dirs:
        dir_path = pai_dsw_utils.get_workspace_path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/ 目录存在")
        else:
            print(f"✗ {dir_name}/ 目录不存在")
    
    print("\n=== 检查完成 ===")


if __name__ == "__main__":
    quick_data_check()
