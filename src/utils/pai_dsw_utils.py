"""PAI-DSW平台工具函数"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PAIDSWUtils:
    """PAI-DSW平台工具类"""
    
    def __init__(self):
        self.workspace_root = Path("/mnt/workspace")
        self.is_pai_dsw = self._check_pai_dsw_environment()
        
    def _check_pai_dsw_environment(self) -> bool:
        """检查是否在PAI-DSW环境中运行"""
        return self.workspace_root.exists()
    
    def get_workspace_path(self, relative_path: str = "") -> Path:
        """获取工作空间路径"""
        if self.is_pai_dsw:
            return self.workspace_root / relative_path
        else:
            # 本地开发环境
            return Path(".") / relative_path
    
    def setup_directories(self) -> None:
        """创建必要的目录结构"""
        directories = [
            "data",
            "models", 
            "results",
            "checkpoints",
            "logs"
        ]
        
        for dir_name in directories:
            dir_path = self.get_workspace_path(dir_name)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"确保目录存在: {dir_path}")
    
    def get_data_path(self, filename: str) -> Path:
        """获取数据文件路径"""
        return self.get_workspace_path("data") / filename
    
    def get_model_path(self, filename: str) -> Path:
        """获取模型文件路径"""
        return self.get_workspace_path("models") / filename
    
    def get_results_path(self, filename: str) -> Path:
        """获取结果文件路径"""
        return self.get_workspace_path("results") / filename
    
    def get_checkpoint_path(self, filename: str) -> Path:
        """获取检查点文件路径"""
        return self.get_workspace_path("checkpoints") / filename
    
    def check_data_files(self) -> Dict[str, bool]:
        """检查必要的数据文件是否存在"""
        required_files = [
            "train.csv",
            "time_ranges.json"
        ]
        
        file_status = {}
        for filename in required_files:
            file_path = self.get_data_path(filename)
            file_status[filename] = file_path.exists()
            
            if file_status[filename]:
                logger.info(f"✓ 数据文件存在: {file_path}")
            else:
                logger.warning(f"✗ 数据文件缺失: {file_path}")
        
        return file_status
    
    def save_experiment_config(self, config: Dict[str, Any], 
                             experiment_name: str) -> None:
        """保存实验配置"""
        config_path = self.get_results_path(f"{experiment_name}_config.json")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验配置已保存: {config_path}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        gpu_info = {
            "available": False,
            "count": 0,
            "devices": []
        }
        
        try:
            import torch
            gpu_info["available"] = torch.cuda.is_available()
            gpu_info["count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["count"]):
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_reserved": torch.cuda.memory_reserved(i),
                    "memory_allocated": torch.cuda.memory_allocated(i)
                }
                gpu_info["devices"].append(device_info)
                
        except ImportError:
            logger.warning("PyTorch未安装，无法获取GPU信息")
        
        return gpu_info
    
    def log_environment_info(self) -> None:
        """记录环境信息"""
        import sys
        import platform
        
        logger.info("=== PAI-DSW环境信息 ===")
        logger.info(f"平台: {platform.platform()}")
        logger.info(f"Python版本: {sys.version}")
        logger.info(f"工作空间: {self.workspace_root}")
        logger.info(f"PAI-DSW环境: {self.is_pai_dsw}")
        
        # GPU信息
        gpu_info = self.get_gpu_info()
        logger.info(f"GPU可用: {gpu_info['available']}")
        if gpu_info["available"]:
            logger.info(f"GPU数量: {gpu_info['count']}")
            for device in gpu_info["devices"]:
                logger.info(f"  GPU {device['id']}: {device['name']}")
    
    def create_symlinks_for_local_dev(self) -> None:
        """为本地开发创建符号链接"""
        if not self.is_pai_dsw:
            # 本地开发时，创建软链接指向实际路径
            local_paths = {
                "data": "./data",
                "models": "./models", 
                "results": "./results"
            }
            
            for name, path in local_paths.items():
                local_path = Path(path)
                if local_path.exists():
                    logger.info(f"本地开发: 使用路径 {local_path}")


# 全局工具实例
pai_dsw_utils = PAIDSWUtils()


# 便捷函数
def setup_pai_dsw_environment():
    """设置PAI-DSW环境"""
    pai_dsw_utils.setup_directories()
    pai_dsw_utils.log_environment_info()
    pai_dsw_utils.create_symlinks_for_local_dev()


def get_pai_dsw_paths():
    """获取PAI-DSW路径信息"""
    return {
        'workspace_path': str(pai_dsw_utils.workspace_root),
        'data_path': str(pai_dsw_utils.get_workspace_path('data')),
        'models_path': str(pai_dsw_utils.get_workspace_path('models')),
        'results_path': str(pai_dsw_utils.get_workspace_path('results')),
        'checkpoints_path': str(pai_dsw_utils.get_workspace_path('checkpoints')),
        'is_pai_dsw': pai_dsw_utils.is_pai_dsw
    }


def check_pai_dsw_data():
    """检查PAI-DSW数据文件"""
    return pai_dsw_utils.check_data_files()


def get_pai_dsw_gpu_info():
    """获取PAI-DSW GPU信息"""
    return pai_dsw_utils.get_gpu_info()
